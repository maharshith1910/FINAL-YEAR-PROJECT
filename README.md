# 🚨 Deep Learning Human Threat Detection — Smart Crime Monitoring

> **Final Review Project | Maharshith | 2-Day Sprint**

A real-time audio threat detection system using CNN and CNN-LSTM models trained on Mel spectrograms to classify distress/scream events and trigger intelligent alerts.

---

## 📁 Project Structure

```
Major Project/
├── data/
│   ├── raw/                  # Raw audio files (.wav)
│   │   ├── scream/
│   │   └── non_scream/
│   ├── processed/            # Extracted Mel spectrogram .npy files
│   └── manifest.csv          # File manifest with labels & splits
├── models/
│   ├── cnn_model.py          # Baseline CNN architecture
│   ├── cnn_lstm_model.py     # CNN-LSTM architecture
│   └── checkpoints/          # Saved .pth model weights
├── training/
│   ├── train.py              # Main training script
│   ├── dataset.py            # Dataset class + augmentation
│   └── evaluate.py           # Evaluation + confusion matrix + curves
├── inference/
│   ├── predictor.py          # Real-time inference engine
│   └── alert_logger.py       # Alert logging to CSV/DB
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── utils/
│   ├── feature_extraction.py # Mel spectrogram extraction
│   └── data_prep.py          # Manifest builder + split
├── outputs/
│   ├── metrics/              # JSON metrics, confusion matrix, curves
│   └── alerts/               # alert_log.csv
├── requirements.txt
└── run_dashboard.sh
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (place audio in data/raw/scream & data/raw/non_scream)
python utils/data_prep.py

# 3. Extract features
python utils/feature_extraction.py

# 4. Train both models
python training/train.py --model cnn --epochs 30
python training/train.py --model cnn_lstm --epochs 30

# 5. Evaluate
python training/evaluate.py --model cnn_lstm

# 6. Launch Dashboard
streamlit run dashboard/app.py
```

---

## 🎯 Architecture

```
Audio Input (mic / file)
        ↓
  Overlapping Windows (1s, 50% overlap)
        ↓
  Log-Mel Spectrogram (128 bands)
        ↓
  ┌─────────────────┐
  │  CNN Feature    │  (4 conv blocks, BatchNorm, MaxPool)
  │  Extractor      │
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │   LSTM Head     │  (2 layers, 256 hidden, dropout)
  └────────┬────────┘
           ↓
  Scream Probability Score
        ↓
  Consecutive-Window Smoother (N=3 threshold)
        ↓
  Alert Engine → CSV Log → Streamlit Dashboard
```

---

## 📊 Evaluation Metrics (reported honestly)

- Accuracy, Precision, Recall, F1, False Positive Rate
- Confusion Matrix (saved as PNG)
- Loss/Accuracy curves per epoch
- Inference latency per window (ms)
- Threshold sweep (Precision-Recall curve)

---

## ⚠️ Ethical Statement

All detections are flagged as **"potential distress event — requires human verification."**
No personal identification is performed. Compliant with Section XIII of the research paper.
