
# English to Amharic Neural Machine Translation

This project is a simple Flask web app for English-to-Amharic machine translation using an attention-based sequence-to-sequence model. It loads SentencePiece tokenizers and pretrained PyTorch checkpoints from the `artifacts/` folder and exposes a lightweight web interface and API for translation.

## Features

- English to Amharic translation
- Attention-based Seq2Seq model
- Basic Seq2Seq model support for comparison
- Flask API endpoints for translation and health checks
- Minimal browser demo interface
- CPU-ready inference for local use and simple deployment

## Project Structure

```text
nmt_deployment/
├── app.py                  # Flask app and web interface
├── nmt_translator.py       # Model loading, tokenizers, decoding logic
├── deployment_config.json  # Model / deployment metadata
├── requirements.txt        # Python dependencies
├── artifacts/
│   ├── en_tokenizer.model
│   ├── am_tokenizer.model
│   ├── attention_seq2seq_best.pt
│   └── basic_seq2seq_best.pt
├── README.md
└── .gitignore
```

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

1. Open a terminal in the project directory.
2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
python app.py
```

5. Open the application in your browser:

```text
http://127.0.0.1:5000
```

## API Endpoints

### GET /
Returns the demo web UI.

### GET /health
Returns model status and deployment metadata.

Example response:

```json
{
  "status": "ok",
  "device": "cpu",
  "source_vocab": 7000,
  "target_vocab": 7000,
  "attention_checkpoint_epoch": 9,
  "basic_model_loaded": true
}
```

### POST /translate
Translates an English sentence into Amharic.

Request body:

```json
{
  "text": "hello world",
  "model": "attention"
}
```

Response:

```json
{
  "text": "hello world",
  "translation": "ም ዓለም",
  "model": "attention",
  "latency_ms": 18.5
}
```

### POST /translate_batch
Translates multiple texts in one request.

Request body:

```json
{
  "texts": ["hello world", "I am going to the university."]
}
```

## Model Details

- Source language: English
- Target language: Amharic
- Model type: Attention-based Seq2Seq + LSTM
- Embedding size: 96
- Hidden size: 192
- Layers: 1
- Dropout: 0.2
- Max decoding length: 32
- Tokenizer: SentencePiece unigram

## Deployment

This project can be deployed to free hosting platforms such as Render or similar Flask-compatible cloud services.

A typical production deployment command is:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

You should add `gunicorn` to the dependency list before deploying to a hosting platform.

## Notes

- The model runs on CPU by default when CUDA is unavailable.
- The project expects the tokenizer and checkpoint files to be present in the `artifacts/` directory.
- Translation quality depends on the trained checkpoint and the model architecture used.

## License

This project is intended for educational and demonstration use.
# machine-translation-project-phase-1
