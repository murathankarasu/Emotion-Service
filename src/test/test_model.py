import torch
from transformers import BertTokenizer, BertModel
import pandas as pd

class DuyguAnalizi(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(DuyguAnalizi, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token'ın çıktısı
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def predict_emotion(text, model, tokenizer, device='cpu'):
    # Metni tokenize et
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Modeli değerlendirme moduna al
    model.eval()
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Duygu kategorileri
    emotion_categories = {
        0: "Üzüntü (Sadness)",
        1: "Neşe (Joy)",
        2: "Aşk (Love)",
        3: "Öfke (Anger)",
        4: "Korku (Fear)",
        5: "Şaşkınlık (Surprise)"
    }
    
    return emotion_categories[predicted_class], probabilities[0].tolist()

if __name__ == "__main__":
    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modeli yükle
    model = DuyguAnalizi(num_labels=6).to(device)
    model.load_state_dict(torch.load('models/duygu_analizi_model.pt', map_location=device))
    tokenizer = model.tokenizer
    
    # Test metinleri
    test_texts = [
        "I feel so happy today! Everything is going great!",
        "I'm really scared about what might happen tomorrow.",
        "I love spending time with my family.",
        "This situation makes me so angry!",
        "I'm surprised by how well everything turned out.",
        "I feel so sad and lonely right now."
    ]
    
    # Her metin için tahmin yap
    print("\nDuygu Analizi Sonuçları:")
    print("-" * 50)
    for text in test_texts:
        emotion, probabilities = predict_emotion(text, model, tokenizer, device)
        print(f"\nMetin: {text}")
        print(f"Tahmin Edilen Duygu: {emotion}")
        print("Tüm Kategorilerin Olasılıkları:")
        for i, prob in enumerate(probabilities):
            print(f"{i}: {prob:.4f}")
        print("-" * 50) 