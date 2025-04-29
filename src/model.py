import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class DuyguAnalizi(nn.Module):
    def __init__(self, num_classes=13, max_length=128):
        super(DuyguAnalizi, self).__init__()
        self.max_length = max_length
        # İngilizce BERT modelini kullanıyoruz
        model_name = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Basit bir sınıflandırıcı
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        return output
    
    def predict(self, text):
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            outputs = self(inputs['input_ids'], inputs['attention_mask'])
            predictions = torch.softmax(outputs, dim=1)
            return predictions

class VeriHazırlayıcı:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def metni_hazırla(self, metinler):
        # Tokenize işlemi
        encoded = self.tokenizer(
            metinler,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Token ID'lerini kontrol et
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Geçersiz token ID'lerini [UNK] token'ı ile değiştir
        input_ids[input_ids >= self.tokenizer.vocab_size] = self.tokenizer.unk_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
    def veri_hazırla(self, df):
        # Tokenize işlemi
        inputs = self.metni_hazırla(df['processed_content'].tolist())
        
        # Etiketleri tensor'a çevir
        labels = torch.tensor(df['sentiment_label'].tolist())
        
        return inputs, labels
