import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class SimpleTokenizer:
    def __init__(self, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        
    def fit_on_texts(self, texts):
        # Count all words
        for text in texts:
            self.word_counts.update(text.split())
            
        # Get most common words
        most_common = self.word_counts.most_common(self.num_words - 1)  # -1 for OOV token
        
        # Create word to index mapping
        self.word_index = {self.oov_token: 0}
        self.word_index.update({word: idx + 1 for idx, (word, _) in enumerate(most_common)})
        
        # Create index to word mapping
        self.index_word = {v: k for k, v in self.word_index.items()}
        
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                sequence.append(self.word_index.get(word, 0))  # 0 is OOV token
            sequences.append(sequence)
        return sequences

class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class NewsClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, gru_dim, num_classes):
        super(NewsClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Bi-directional GRU
        self.gru = nn.GRU(embed_dim, gru_dim, batch_first=True, bidirectional=True)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(gru_dim * 2)  # Account for bi-directional output
        
        # Fully connected layer
        self.fc = nn.Linear(gru_dim * 2, num_classes)
        
    def forward(self, x):
        # Pass input through embedding layer
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embed_dim)
        
        # Pass through GRU
        gru_out, hidden = self.gru(embedded)  # Shape: (batch_size, seq_length, gru_dim*2)
        
        # Concatenate forward and backward hidden states (last layer of GRU)
        final_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Shape: (batch_size, gru_dim*2)
        
        # Apply BatchNorm and fully connected layer
        normalized = self.bn(final_hidden_state)  # Shape: (batch_size, gru_dim*2)
        output = self.fc(normalized)  # Shape: (batch_size, num_classes)
        
        return torch.softmax(output, dim=1)

def pad_sequences(sequences, maxlen, padding='post'):
    padded = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i, :] = seq[:maxlen]
        else:
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:  # 'pre'
                padded[i, -len(seq):] = seq
    return padded

class NewsClassificationPipeline:
    def __init__(self, data_path, num_words=5000, embed_dim=64, gru_dim=64, batch_size=32, num_epochs=10, feature=['headline']):
        self.data_path = data_path
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.gru_dim = gru_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loss = []
        self.val_loss = []
        self.val_accuracy = 0
        self.feature = feature
        
    def load_data(self):
        self.data = pd.read_json(self.data_path)
        
    def preprocess_data(self):
        # Tokenize text
        self.tokenizer = SimpleTokenizer(num_words=self.num_words, oov_token="<OOV>")
        
        if len(self.feature) > 1:
            # 多個特徵欄位，將其合併為單一文本
            combined_text = self.data[self.feature].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        else:
            # 單一特徵欄位
            combined_text = self.data[self.feature[0]]
        
        # 訓練 Tokenizer
        self.tokenizer.fit_on_texts(combined_text)
        sequences = self.tokenizer.texts_to_sequences(combined_text)
        
        # Pad sequences
        max_length = max(len(x) for x in sequences)
        self.padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.data['category'])
        self.labels_one_hot = np.eye(len(self.label_encoder.classes_))[labels]
        
    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.padded_sequences, self.labels_one_hot, test_size=0.2, random_state=42
        )
        
    def create_dataloaders(self):
        train_dataset = NewsDataset(self.X_train, self.y_train)
        val_dataset = NewsDataset(self.X_val, self.y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
    def initialize_model(self):
        self.model = NewsClassifier(
            vocab_size=self.num_words,
            embed_dim=self.embed_dim,
            gru_dim=self.gru_dim,
            num_classes=len(self.label_encoder.classes_)
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_sequences, batch_labels in self.train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_sequences)
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(batch_labels.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == true_labels).sum().item()
            
            self.train_loss.append(train_loss / len(self.train_loader))
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_sequences, batch_labels in self.val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_sequences)
                    loss = self.criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, true_labels = torch.max(batch_labels.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == true_labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(true_labels.cpu().numpy())
            self.val_accuracy = val_correct / val_total
            
            self.val_loss.append(val_loss / len(self.val_loader))
        
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.tokenizer.word_index,
            'label_encoder_classes': self.label_encoder.classes_
        }, 'news_classifier.pth')
        
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.create_dataloaders()
        self.initialize_model()
        self.train()

if __name__ == "__main__":
    pipeline = NewsClassificationPipeline(data_path='Processed.json' , feature=['headline', 'short_description'])
    pipeline.run()
    # Assuming the pipeline object has attributes `train_loss` and `val_loss` which are lists of loss values
    train_loss = pipeline.train_loss
    val_loss = pipeline.val_loss

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()
