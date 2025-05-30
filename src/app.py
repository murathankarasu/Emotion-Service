from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import os
import boto3
from botocore.exceptions import ClientError
from functools import lru_cache

app = Flask(__name__)

def download_model_from_s3():
    s3 = boto3.client('s3',
                     aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                     aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                     region_name=os.environ.get('AWS_REGION', 'us-east-1'))
    
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    model_key = 'emotion-model/duygu_analizi_model.pt'
    local_path = os.path.join(os.path.dirname(__file__), 'models', 'duygu_analizi_model.pt')
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if not os.path.exists(local_path):
        try:
            print(f"Model indiriliyor: s3://{bucket_name}/{model_key}")
            s3.download_file(bucket_name, model_key, local_path)
            print(f"Model başarıyla indirildi: {local_path}")
        except ClientError as e:
            print(f"Model indirme hatası: {e}")
            raise
    else:
        print(f"Model zaten mevcut: {local_path}")
    return local_path

class DuyguAnalizi(torch.nn.Module):
    def __init__(self, num_labels=6):
        super(DuyguAnalizi, self).__init__()
        print("Tokenizer indiriliyor...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("BERT modeli indiriliyor...")
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def predict_emotion(text, model, tokenizer, device='cpu'):
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    emotion_categories = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    
    return emotion_categories[predicted_class], probabilities[0].tolist()

# Grammar correction modeli ve tokenizer'ı yükle
@lru_cache(maxsize=1)
def get_grammar_model():
    grammar_tokenizer = AutoTokenizer.from_pretrained("pszemraj/grammar-synthesis-small")
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/grammar-synthesis-small")
    return grammar_tokenizer, grammar_model

def correct_text(text):
    grammar_tokenizer, grammar_model = get_grammar_model()
    inputs = grammar_tokenizer([text], return_tensors="pt", max_length=128, truncation=True)
    outputs = grammar_model.generate(**inputs, max_length=128)
    corrected = grammar_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return corrected

@lru_cache(maxsize=1)
def get_emotion_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained("AdamCodd/tinybert-emotion-balanced").to(device)
    tokenizer = AutoTokenizer.from_pretrained("AdamCodd/tinybert-emotion-balanced")
    return model, tokenizer, device

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Text not found'}), 400
        # 1. Metni düzelt
        corrected_text = correct_text(text)
        # 2. Duygu analizi
        model, tokenizer, device = get_emotion_model()
        emotion, probabilities = predict_emotion(corrected_text, model, tokenizer, device)
        result = {
            'original_text': text,
            'corrected_text': corrected_text,
            'emotion': emotion,
            'probabilities': {
                'sadness': probabilities[0],
                'joy': probabilities[1],
                'love': probabilities[2],
                'anger': probabilities[3],
                'fear': probabilities[4],
                'surprise': probabilities[5]
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 