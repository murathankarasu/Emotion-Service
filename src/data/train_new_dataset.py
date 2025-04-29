import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

class DuyguAnalizi(nn.Module):
    def __init__(self, num_labels=6):  # 6 duygu kategorisi için
        super(DuyguAnalizi, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token'ın çıktısı
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def load_and_preprocess_data(file_path):
    """Veri setini yükler ve ön işler"""
    df = pd.read_csv(file_path)
    return df

def prepare_data(df, tokenizer, max_length=128):
    """Veriyi modele uygun formata dönüştürür"""
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    labels = torch.tensor(df['label'].tolist())
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        labels
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset

def train_model(model, train_loader, val_loader, num_epochs=3, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Eğitim
        model.train()
        total_train_loss = 0
        train_progress = tqdm(train_loader, desc='Eğitim')
        
        for batch_input_ids, batch_attention_mask, batch_labels in train_progress:
            input_ids = batch_input_ids.to(device)
            attention_mask = batch_attention_mask.to(device)
            labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Doğrulama
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_input_ids, batch_attention_mask, batch_labels in tqdm(val_loader, desc='Doğrulama'):
                input_ids = batch_input_ids.to(device)
                attention_mask = batch_attention_mask.to(device)
                labels = batch_labels.to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'\nOrtalama eğitim kaybı: {avg_train_loss:.4f}')
        print(f'Ortalama doğrulama kaybı: {avg_val_loss:.4f}')
        print('\nSınıflandırma Raporu:')
        print(classification_report(true_labels, predictions))
    
    return train_losses, val_losses

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri yükleme ve ön işleme
    print("Veri yükleniyor ve ön işleniyor...")
    df = load_and_preprocess_data("data/balanced_text.csv")
    
    # Model ve tokenizer yükleme
    print("\nModel ve tokenizer yükleniyor...")
    model = DuyguAnalizi(num_labels=6).to(device)  # 6 duygu kategorisi
    tokenizer = model.tokenizer
    
    # Veri hazırlama
    print("Veri hazırlanıyor...")
    train_dataset, val_dataset = prepare_data(df, tokenizer)
    
    # DataLoader'ları oluşturma
    print("\nDataLoader'lar hazırlanıyor...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model eğitimi
    print("\nModel eğitimi başlıyor...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=3,
        device=device
    )
    
    # Modeli kaydet
    print("\nModel kaydediliyor...")
    torch.save(model.state_dict(), 'models/duygu_analizi_model.pt')
    print("Model kaydedildi: models/duygu_analizi_model.pt") 