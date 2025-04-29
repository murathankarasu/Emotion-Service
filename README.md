# LoriApp - Duygu Analizi Sistemi

Bu proje, kullanıcıların metin tabanlı gönderilerinden duygu analizi yapan bir sistemdir.

## Proje Yapısı

```
emotion-recognition/
├── dataset/
│   ├── training.csv
│   ├── test.csv
│   └── validation.csv
├── src/
│   ├── models/
│   │   └── emotion_model.py
│   ├── utils/
│   │   └── data_processor.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Modeli eğitin:
```bash
cd src
python train.py
```

## Kullanım

Model eğitildikten sonra, metin tabanlı duygu analizi yapabilirsiniz:

```python
from models.emotion_model import EmotionModel

model = EmotionModel()
model.load_model("./saved_model")

prediction = model.predict("Bugün çok mutluyum")
print(prediction)
```

## Duygu Sınıfları

- 0: Mutlu
- 1: Üzgün
- 2: Öfkeli
- 3: Endişeli
- 4: Şaşkın
- 5: Nötr 