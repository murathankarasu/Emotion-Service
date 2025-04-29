from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
import os
import boto3
from botocore.exceptions import ClientError

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
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    emotion_categories = {
        0: "Üzüntü (Sadness)",
        1: "Neşe (Joy)",
        2: "Aşk (Love)",
        3: "Öfke (Anger)",
        4: "Korku (Fear)",
        5: "Şaşkınlık (Surprise)"
    }
    
    return emotion_categories[predicted_class], probabilities[0].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

print("Model başlatılıyor...")
model = DuyguAnalizi(num_labels=6).to(device)
model.eval()

print("Model dosyası S3'ten indiriliyor...")
model_path = download_model_from_s3()
state_dict = torch.load(model_path, map_location=device)
if 'bert.embeddings.position_ids' not in state_dict:
    state_dict['bert.embeddings.position_ids'] = torch.arange(512).unsqueeze(0)
model.load_state_dict(state_dict)
tokenizer = model.tokenizer
print("Model başarıyla yüklendi!")

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Metin bulunamadı'}), 400
        
        emotion, probabilities = predict_emotion(text, model, tokenizer, device)
        
        result = {
            'text': text,
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