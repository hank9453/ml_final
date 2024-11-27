import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.values if hasattr(texts, 'values') else texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

class NewsClassifier(nn.Module):
    def __init__(self, num_classes, bert_model='distilbert-base-uncased', dropout=0.1, hidden_dim=256, num_layers=1):
        super(NewsClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model)
        self.bilstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=num_layers, 
                              bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        pooled_output = torch.mean(lstm_output, 1)
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

class NewsClassificationPipeline:
    def __init__(self, data_path, bert_model='distilbert-base-uncased', max_length=128, 
                 batch_size=32, num_epochs=5, feature=['headline']):
        self.data_path = data_path
        self.bert_model = bert_model
        self.max_length = max_length
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
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_model)
        
        if len(self.feature) > 1:
            self.texts = self.data[self.feature].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            ).reset_index(drop=True)
        else:
            self.texts = self.data[self.feature[0]].reset_index(drop=True)
        
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.data['category'])
        self.labels_one_hot = np.eye(len(self.label_encoder.classes_))[labels]
        
    def split_data(self):
        texts_array = self.texts.values
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            texts_array, self.labels_one_hot, test_size=0.2, random_state=42
        )
        
    def create_dataloaders(self):
        train_dataset = NewsDataset(
            self.X_train, self.y_train, self.tokenizer, self.max_length
        )
        val_dataset = NewsDataset(
            self.X_val, self.y_val, self.tokenizer, self.max_length
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size
        )
        
    def initialize_model(self):
        self.model = NewsClassifier(
            num_classes=len(self.label_encoder.classes_),
            bert_model=self.bert_model
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-5,
            weight_decay=0.01
        )
        
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(labels.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == true_labels).sum().item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            self.train_loss.append(avg_train_loss)
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, true_labels = torch.max(labels.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == true_labels).sum().item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_accuracy = val_correct / val_total
            self.val_loss.append(avg_val_loss)
            self.val_accuracy = val_accuracy
            
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': self.label_encoder.classes_
        }, 'distilbert_news_classifier.pth')
        
    def run(self):
        print("Loading data...")
        self.load_data()
        print("Preprocessing data...")
        self.preprocess_data()
        print("Splitting data...")
        self.split_data()
        print("Creating dataloaders...")
        self.create_dataloaders()
        print("Initializing model...")
        self.initialize_model()
        print("Starting training...")
        self.train()
        print("Training completed!")

if __name__ == "__main__":
    pipeline = NewsClassificationPipeline(
        data_path='Processed.json',
        bert_model='distilbert-base-uncased',
        batch_size=16,
        num_epochs=5,
        feature=['headline','short_description']
    )
    pipeline.run()
    
    plt.figure(figsize=(10, 5))
    plt.plot(pipeline.train_loss, label='Train Loss')
    plt.plot(pipeline.val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()
