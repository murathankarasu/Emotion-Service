# LoriApp - Emotion Recognition API

This project provides a RESTful API for emotion recognition from English text. The API automatically corrects grammar and spelling mistakes in the input text before performing emotion analysis using a BERT-based model.

## Project Structure

```
emotion-recognition/
├── models/
│   └── duygu_analizi_model.pt
├── src/
│   ├── app.py
│   ├── model.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── ...
└── README.md
```

## Features
- Automatic grammar and spelling correction for English text (using `language_tool_python`)
- Emotion classification into six categories: Sadness, Joy, Love, Anger, Fear, Surprise
- REST API endpoint for easy integration

## Installation

1. Clone the repository and install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

2. Make sure you have the model file (`duygu_analizi_model.pt`) in the `src/models/` directory. If not, it will be downloaded automatically from S3 (AWS credentials required).

3. Start the API server:
   ```bash
   cd src
   python app.py
   ```

## API Usage

### Endpoint
```
POST /analyze
Content-Type: application/json
```

#### Request Body
```json
{
  "text": "I am very hapy today it is a beautful day"
}
```

#### Example Response
```json
{
  "original_text": "I am very hapy today it is a beautful day",
  "corrected_text": "I am very happy today. It is a beautiful day.",
  "emotion": "Joy",
  "probabilities": {
    "sadness": 0.01,
    "joy": 0.95,
    "love": 0.01,
    "anger": 0.01,
    "fear": 0.01,
    "surprise": 0.01
  }
}
```

## Emotion Categories
- 0: Sadness
- 1: Joy
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

## Environment Variables
To download the model from AWS S3, set the following environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (default: us-east-1)
- `S3_BUCKET_NAME`

## License
MIT 