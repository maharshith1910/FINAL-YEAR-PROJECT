# 🛡️ SafeWatch AI — Deep Learning Human Threat Detection

> **Final Year Project · Maharshith & Team · 2026**
> Smart Crime Detection using CNN + CNN-LSTM Scream Detection

![Dashboard Preview](docs/preview.png)

---

## 📚 Documentation & Manuals
- 📖 **[CLI Manual & Command Guide](CLI_RUN_GUIDE.md)**: Complete step-by-step manual for running every CLI command without confusion.
- 🔬 **[Proof Verification & System Manual](RUN_AND_VERIFY.md)**: Comprehensive verification guide for examiners and project evaluators.

## 🚀 Quick Start (One-Shot Verification & Web UI)

```powershell
# 1. Verify all endpoints and features with 1 command:
python verify_system.py

# 2. Launch the Web Dashboard:
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

## 🧠 Architecture

| Model | Input | Key Layers | Purpose |
|-------|-------|-----------|---------|
| **Baseline CNN** | (B, 1, 128, 44) Mel | 4× Conv→BN→ReLU→Pool | Per-frame classification |
| **CNN-LSTM** | (B, T, 1, 128, 44) | CNN extractor + 2-layer LSTM | Temporal pattern detection |

**Feature pipeline:** 22,050 Hz → 1s windows, 0.5s hop → 128-band log-Mel spectrogram → normalised → model

## 📊 Datasets

| Dataset | Source | Size | License |
|---------|--------|------|---------|
| Human Scream Audio | Kaggle (bhavikjikadara) | 862 scream + 2,631 non-scream WAVs | MIT |
| ESC-50 | GitHub (karoldvl) | 2,000 environmental audio clips | CC BY |

## 🗂️ Project Structure

```
FINAL-YEAR-PROJECT/
├── data/
│   ├── raw/scream/          ← Kaggle Screaming WAVs
│   ├── raw/non_scream/      ← Kaggle NotScreaming + ESC-50
│   ├── features/            ← .npy Mel features (auto-created)
│   └── features_manifest.csv
├── models/
│   ├── cnn_model.py         ← Baseline CNN
│   ├── cnn_lstm_model.py    ← CNN + LSTM
│   └── checkpoints/         ← best .pth files
├── training/
│   ├── dataset.py           ← PyTorch datasets
│   ├── train.py             ← training loop
│   └── evaluate.py          ← metrics + plots
├── inference/
│   ├── predictor.py         ← inference engine
│   └── alert_logger.py      ← CSV alert logger
├── utils/
│   ├── data_prep.py         ← manifest + train/val/test split
│   ├── feature_extraction.py← Mel extraction to .npy
│   ├── setup_kaggle.py      ← Kaggle token wizard
│   └── download_datasets.py ← automated downloader
├── dashboard/app.py         ← Streamlit dashboard
├── outputs/metrics/         ← JSON results, PNGs
├── CODEX_PROMPTS.py         ← 3 teammate prompts for Codex/Claude
└── run_all.bat              ← one-click pipeline
```

## ⚡ Step-by-Step Commands

```bash
# Data
python utils/setup_kaggle.py
python utils/download_datasets.py
python utils/data_prep.py
python utils/feature_extraction.py

# Training
python training/train.py --model cnn      --epochs 30 --lr 1e-3
python training/train.py --model cnn_lstm --epochs 30 --lr 1e-3

# Evaluation
python training/evaluate.py --model cnn
python training/evaluate.py --model cnn_lstm

# Dashboard
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

## 📈 Expected Results (ranges — run eval for real numbers)

| Metric | CNN | CNN-LSTM |
|--------|-----|----------|
| Accuracy | ~82–88% | ~85–92% |
| F1 (scream) | ~75–85% | ~80–88% |
| FPR | ~8–15% | ~5–12% |
| Latency/window | ~2–5ms | ~5–15ms |

> ⚠️ These are estimates. **Never report fabricated metrics.** Run `evaluate.py` and use those numbers.

## 🎯 Teammate Prompts

See [`CODEX_PROMPTS.py`](CODEX_PROMPTS.py) for three ready-to-paste Codex prompts:
- **Prompt 1:** Data pipeline (data_prep + feature_extraction)
- **Prompt 2:** Model training (CNN, CNN-LSTM, evaluate)
- **Prompt 3:** Dashboard & inference (predictor + alert_logger)

## ⚠️ Ethical Notice

All detections are flagged as **potential distress events requiring human verification**.
No personal identification is performed. This is an assistive tool for security
personnel, not an autonomous decision-making system.

---

*Deep Learning Human Threat Detection · Smart Crime Detection · 2026*
