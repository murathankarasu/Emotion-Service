import requests
import json

def test_emotion_api(text):
    url = "http://localhost:10000/analyze"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # HTTP hatalarını kontrol et
        
        result = response.json()
        print("\nMetin:", result["text"])
        print("Tahmin Edilen Duygu:", result["emotion"])
        print("\nDuygu Olasılıkları:")
        for emotion, probability in result["probabilities"].items():
            print(f"{emotion}: {probability:.4f}")
        print("-" * 50)
        
    except requests.exceptions.RequestException as e:
        print(f"Hata oluştu: {e}")

# Test edilecek cümleler
test_sentences = [
    "I am so happy to see you!",
    "I feel really sad today.",
    "I love spending time with my family.",
    "I'm so angry about what happened!",
    "I'm scared of the dark.",
    "I'm surprised to see you here!"
]

# Her cümleyi test et
for sentence in test_sentences:
    test_emotion_api(sentence) 