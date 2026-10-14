# SafeWatch AI — System Execution & Proof Verification Manual

**Project Title**: SafeWatch AI — Deep Learning Human Threat & Scream Detection  
**Architecture**: Mel Spectrograms + Deep CNN (Baseline) + Spatial-Temporal CNN-LSTM  
**Target Environment**: Windows & Linux (Python 3.10+, PyTorch 2.x, Streamlit)  

---

## 📑 Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Quick-Start: Automated Verification Test](#2-quick-start-automated-verification-test)
3. [Step-by-Step Execution Guide](#3-step-by-step-execution-guide)
   - [Step 1: Data Preparation & Recording Split](#step-1-data-preparation--recording-split)
   - [Step 2: Log-Mel Feature Precomputation](#step-2-log-mel-feature-precomputation)
   - [Step 3: Model Training](#step-3-model-training)
   - [Step 4: Held-Out Test Evaluation](#step-4-held-out-test-evaluation)
   - [Step 5: Live Audio Inference & Streaming](#step-5-live-audio-inference--streaming)
   - [Step 6: Streamlit Glassmorphic Dashboard](#step-6-streamlit-glassmorphic-dashboard)
4. [How to Verify Every Proof](#4-how-to-verify-every-proof)
   - [Proof 1: Data Manifest & Windowing](#proof-1-data-manifest--windowing)
   - [Proof 2: Spectrogram Array Shape & Normalization](#proof-2-spectrogram-array-shape--normalization)
   - [Proof 3: Model Architectures & Parameter Counts](#proof-3-model-architectures--parameter-counts)
   - [Proof 4: Checkpoint Files & Model Weights](#proof-4-checkpoint-files--model-weights)
   - [Proof 5: Evaluation Metrics, Confusion Matrix & PR Curves](#proof-5-evaluation-metrics-confusion-matrix--pr-curves)
   - [Proof 6: Threat Alert Logging & Multi-Thread Concurrency](#proof-6-threat-alert-logging--multi-thread-concurrency)
5. [Troubleshooting & Windows Specifics](#5-troubleshooting--windows-specifics)

---

## 1. System Overview & Architecture

SafeWatch AI is designed to detect human distress screams in real time within high-noise or public environments.

```mermaid
flowchart TD
    A[Raw Audio: Kaggle Screaming + ESC-50] --> B[utils/data_prep.py: Recording Split]
    B --> C[data/features_manifest.csv]
    C --> D[utils/feature_extraction.py: Log-Mel Windows]
    D --> E[data/features/*.npy: Shape 1, 128, 44]
    E --> F1[Branch 1: ScreamCNN (421K params)]
    E --> F2[Branch 2: ScreamCNNLSTM (1.47M params)]
    F1 --> G1[training/evaluate.py]
    F2 --> G2[training/evaluate.py]
    G1 & G2 --> H[outputs/metrics/*.json + PNG plots]
    F1 & F2 --> I[inference/predictor.py: Real-Time Engine]
    I --> J[inference/alert_logger.py: outputs/alerts/alert_log.csv]
    I & J --> K[dashboard/app.py: Streamlit Web UI]
```

---

## 2. Quick-Start: Automated Verification Test

To verify every endpoint and subsystem in **one single automated command**:

```powershell
python verify_system.py
```

### What this automated test verifies:
- ✅ **Test 1**: `features_manifest.csv` schema, split distribution, and 5 random `.npy` feature files.
- ✅ **Test 2**: PyTorch `ScreamDataset` (single window) & `ScreamSequenceDataset` (sliding temporal sequence).
- ✅ **Test 3**: Model forward passes (`ScreamCNN` -> `[B, 2]`, `ScreamCNNLSTM` -> `[B, 2]`, `feature_dim = 256`).
- ✅ **Test 4**: `AlertLogger` thread-safety, CSV generation, and severity classification.
- ✅ **Test 5**: Real audio processing via `ScreamPredictor` on real sample recordings.
- ✅ **Test 6**: `dashboard/app.py` dependency and syntax validation.

---

## 3. Step-by-Step Execution Guide

### Step 1: Data Preparation & Recording Split
Generates the recording-level split (70% Train / 15% Val / 15% Test) ensuring no video segment leaks across splits.
```powershell
python utils/data_prep.py
```
*Output*: `data/features_manifest.csv`

---

### Step 2: Log-Mel Feature Precomputation
Slices clips into 1.0s windows with 0.5s hop and computes normalized log-Mel spectrograms:
```powershell
python utils/feature_extraction.py
```
*Output*: `data/features/*.npy` (103,607 precomputed window files, each of shape `1 x 128 x 44`).

---

### Step 3: Model Training
Trains the deep learning models using dynamic inverse-frequency weighted `CrossEntropyLoss` to handle class imbalance.

**Train ScreamCNN**:
```powershell
# Standard 30 Epochs:
python training/train.py --model cnn --epochs 30 --lr 1e-3 --batch_size 64

# Fast Prototype / Verification Run:
python training/train.py --model cnn --epochs 1 --batch_size 64 --max_batches 30
```

**Train ScreamCNNLSTM**:
```powershell
# Standard 30 Epochs:
python training/train.py --model cnn_lstm --epochs 30 --lr 1e-3 --batch_size 64 --n_context 5

# Fast Prototype / Verification Run:
python training/train.py --model cnn_lstm --epochs 1 --batch_size 32 --max_batches 20
```

*Outputs*:
- `models/checkpoints/cnn_best.pth`
- `models/checkpoints/cnn_lstm_best.pth`
- `outputs/metrics/cnn_train_log.csv` & `cnn_curves.png`
- `outputs/metrics/cnn_lstm_train_log.csv` & `cnn_lstm_curves.png`

---

### Step 4: Held-Out Test Evaluation
Evaluates on the held-out test split, computing strictly genuine metrics with no data fabrication:
```powershell
# Evaluate CNN:
python training/evaluate.py --model cnn

# Evaluate CNN-LSTM:
python training/evaluate.py --model cnn_lstm --n_context 5
```

*Outputs*:
- `outputs/metrics/cnn_test_results.json`
- `outputs/metrics/cnn_confusion_matrix.png`
- `outputs/metrics/cnn_pr_curve.png`
- `outputs/metrics/cnn_lstm_test_results.json`
- `outputs/metrics/cnn_lstm_confusion_matrix.png`
- `outputs/metrics/cnn_lstm_pr_curve.png`

---

### Step 5: Live Audio Inference & Streaming
Run inference directly on any standalone audio file or microphone:

**Process Audio File**:
```powershell
python inference/predictor.py --model cnn --audio "data/raw/scream/sample.wav"
```

**Live Microphone Stream**:
```powershell
python inference/predictor.py --model cnn_lstm --mic
```

---

### Step 6: Streamlit Glassmorphic Dashboard
Launch the interactive web UI:
```powershell
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 4. How to Verify Every Proof

If an evaluator, instructor, or teammate asks for proof of functionality, run these verification commands:

### Proof 1: Data Manifest & Windowing
Verify that 103,607 real windows exist and are indexed:
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/features_manifest.csv'); print('Total windows:', len(df)); print(df['split'].value_counts())"
```
**Expected Output**:
```text
Total windows: 103607
split
train    72485
val      15561
test     15561
```

---

### Proof 2: Spectrogram Array Shape & Normalization
Verify that extracted features match the exact neural network input shape `(1, 128, 44)`:
```powershell
python -c "import numpy as np, glob; p=glob.glob('data/features/*.npy')[0]; a=np.load(p); print('File:', p); print('Shape:', a.shape); print('Mean:', round(a.mean(), 4), 'Std:', round(a.std(), 4))"
```
**Expected Output**:
```text
Shape: (1, 128, 44)
Mean: 0.0 Std: 1.0
```

---

### Proof 3: Model Architectures & Parameter Counts
Verify parameters and feature dimensions:
```powershell
python -c "from models.cnn_model import ScreamCNN; from models.cnn_lstm_model import ScreamCNNLSTM; cnn=ScreamCNN(); lstm=ScreamCNNLSTM(); print('CNN Params:', sum(p.numel() for p in cnn.parameters())); print('CNN-LSTM Params:', sum(p.numel() for p in lstm.parameters()))"
```
**Expected Output**:
```text
CNN Params: 421410
CNN-LSTM Params: 1473026
```

---

### Proof 4: Checkpoint Files & Model Weights
Verify checkpoint existence and internal metadata:
```powershell
python -c "import torch; ckpt=torch.load('models/checkpoints/cnn_best.pth', weights_only=False); print('Keys:', list(ckpt.keys())); print('Val Acc:', ckpt['val_acc']); print('Epoch:', ckpt['epoch'])"
```
**Expected Output**:
```text
Keys: ['epoch', 'model_state', 'val_acc', 'args']
Val Acc: <numeric float>
Epoch: <int>
```

---

### Proof 5: Evaluation Metrics, Confusion Matrix & PR Curves
Inspect the genuine test metrics generated on held-out test audio:
```powershell
python -c "import json; print(json.dumps(json.load(open('outputs/metrics/cnn_test_results.json')), indent=2))"
```
You can inspect the generated visual figures directly:
- `outputs/metrics/cnn_confusion_matrix.png`
- `outputs/metrics/cnn_pr_curve.png`
- `outputs/metrics/cnn_curves.png`

---

### Proof 6: Threat Alert Logging & Multi-Thread Concurrency
Verify the threat alert logger schema:
```powershell
python -c "from inference.alert_logger import AlertLogger; l=AlertLogger(); print('Total logged alerts:', len(l.load_all())); print(l.load_all().tail(2))"
```
**Expected Columns**:
`alert_id`, `timestamp_iso`, `timestamp_sec`, `location`, `confidence`, `severity`, `model_used`, `verification_status`, `note`.

---

### Proof 7: AI Acoustic Forensic Dossier & Outlier Filtering
Verify that the system filters vehicle horns and sirens from human screams:
```powershell
python -c "from inference.forensic_engine import AcousticForensicEngine; e=AcousticForensicEngine(); horn=e.analyze('data/test_samples/vehicle_horn_outlier.wav', 0.88); scream=e.analyze('data/raw/scream/---1_cCGK4M_out.wav', 0.95); print('Horn Outlier Filtered:', horn.classification, 'Threat:', horn.is_crime_threat); print('Scream Confirmed:', scream.classification, 'Threat:', scream.is_crime_threat)"
```
**Expected Output**:
```text
Horn Outlier Filtered: VEHICLE_HORN Threat: False
Scream Confirmed: GENUINE_HUMAN_SCREAM Threat: True
```

---

## 5. Troubleshooting & Windows Specifics

1. **Unicode / Console Encoding**:  
   All scripts include `sys.stdout.reconfigure(encoding="utf-8")` to avoid Windows cp1252 character map errors.
2. **PyTorch on Windows CPU**:  
   Always leave `--num_workers 0` when training on CPU to avoid Windows multiprocessing spawn deadlocks.
3. **No GPU Available Warning**:  
   SafeWatch AI runs on standard CPU. If an NVIDIA GPU with CUDA is present, PyTorch will automatically switch to CUDA acceleration.
4. **Gemini API Outlier Reasoning**:  
   If `GEMINI_API_KEY` is configured in the environment, SafeWatch AI invokes Gemini 2.5 Flash for natural-language forensic rationale generation. If offline, the built-in deterministic bio-acoustic expert engine operates automatically with zero dependencies.
