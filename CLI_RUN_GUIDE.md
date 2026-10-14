# SafeWatch AI — Complete CLI Manual & Command Reference Guide

This guide is your single, definitive reference for running **every command** in SafeWatch AI manually from your terminal. Follow these instructions step-by-step to avoid any confusion or path errors.

---

## 📌 Table of Contents
1. [Terminal Preparation & Environment](#1-terminal-preparation--environment)
2. [Quick Reference Cheat Sheet (Most Common Commands)](#2-quick-reference-cheat-sheet)
3. [Step 1: End-to-End System Health Verification](#step-1-end-to-end-system-health-verification)
4. [Step 2: Launching the Web Dashboard (Streamlit UI)](#step-2-launching-the-web-dashboard-streamlit-ui)
5. [Step 3: Running Inference on Audio Files](#step-3-running-inference-on-audio-files)
6. [Step 4: Running Live Microphone Audio Inference](#step-4-running-live-microphone-audio-inference)
7. [Step 5: Testing Acoustic Outlier & Noise Discrimination](#step-5-testing-acoustic-outlier--noise-discrimination)
8. [Step 6: Data Preparation & Feature Extraction](#step-6-data-preparation--feature-extraction)
9. [Step 7: Training Neural Network Models](#step-7-training-neural-network-models)
10. [Step 8: Model Evaluation on Held-Out Test Set](#step-8-model-evaluation-on-held-out-test-set)
11. [Step 9: Inspecting Generated Threat Logs & Metrics](#step-9-inspecting-generated-threat-logs--metrics)
12. [Troubleshooting & Windows Specific Fixes](#troubleshooting--windows-specific-fixes)

---

## 1. Terminal Preparation & Environment

Always make sure you open **PowerShell** or **Command Prompt (CMD)** and navigate to the project root directory.

### In PowerShell or CMD:
```powershell
cd "c:\Users\Admin\Desktop\MAJOR PROJECT\FINAL-YEAR-PROJECT"
```

> **IMPORTANT**: Always use quotes around the path because `MAJOR PROJECT` contains a space.

### Check Python Installation:
```powershell
python --version
```
*(Recommended: Python 3.10+)*

### Install or Update Required Libraries:
```powershell
pip install -r requirements.txt
```

### (Optional) Enable Gemini Outlier Verification Agent:
If you want the multimodal context agent to generate detailed police dispatch rationale:
```powershell
# In PowerShell:
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# In Windows Command Prompt (CMD):
set GEMINI_API_KEY=your-gemini-api-key-here
```
*(Note: If no API key is provided, the system automatically uses the offline physical bio-acoustic arbiter with zero downtime).*

---

## 2. Quick Reference Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Verify System Health** | `python verify_system.py` |
| **Launch Web Dashboard** | `streamlit run dashboard/app.py` |
| **Run Inference on Audio File** | `python inference/predictor.py --model cnn --audio "data/raw/scream/---1_cCGK4M_out.wav"` |
| **Run Inference on Live Mic** | `python inference/predictor.py --model cnn --mic` |
| **Evaluate CNN on Test Split** | `python training/evaluate.py --model cnn` |
| **Evaluate CNN-LSTM on Test Split** | `python training/evaluate.py --model cnn_lstm --n_context 5` |
| **Train CNN (Fast Verification)** | `python training/train.py --model cnn --epochs 1 --batch_size 64 --max_batches 30` |
| **Train CNN (Full 30 Epochs)** | `python training/train.py --model cnn --epochs 30 --lr 1e-3 --batch_size 64` |
| **Inspect Recent Alert Logs** | `python -c "import pandas as pd; print(pd.read_csv('outputs/alerts/alert_log.csv').tail(5))"` |

---

## Step 1: End-to-End System Health Verification

Before running anything, verify that all features, models, checkpoints, and inference engines are intact:

```powershell
python verify_system.py
```

### Expected Output:
```text
═══════════════════════════════════════════════════════════════
  TEST 1: Data Manifest & Precomputed Feature Proofs
═══════════════════════════════════════════════════════════════
  [PASS] features_manifest.csv exists
  ...
═══════════════════════════════════════════════════════════════
  🎉 ALL 7 SYSTEM-WIDE VERIFICATION TESTS PASSED SUCCESSFULLY!
═══════════════════════════════════════════════════════════════
```

---

## Step 2: Launching the Web Dashboard (Streamlit UI)

To launch the glassmorphic SafeWatch AI web application:

```powershell
streamlit run dashboard/app.py
```

Or run via python module syntax:
```powershell
python -m streamlit run dashboard/app.py --server.port 8501
```

### How to Access:
- Open your browser at: **`http://localhost:8501`**
- Press **`Ctrl + C`** in the terminal to stop the dashboard server when done.

---

## Step 3: Running Inference on Audio Files

The real-time predictor extracts 1.0s sliding windows with a 0.5s hop, applies transient noise gating, passes through the neural network, and runs consecutive-window smoothing.

### 3A. Run Baseline CNN on a Scream Audio:
```powershell
python inference/predictor.py --model cnn --audio "data/raw/scream/---1_cCGK4M_out.wav"
```

### 3B. Run Baseline CNN on a Non-Scream (Ambient) Audio:
```powershell
python inference/predictor.py --model cnn --audio "data/raw/non_scream/--PJHxphWEs_out.wav"
```

### 3C. Customizing Threshold, Consecutive Windows & Location Tag:
```powershell
python inference/predictor.py --model cnn --audio "data/raw/scream/---1_cCGK4M_out.wav" --threshold 0.55 --consec 2 --location "Subway Platform 4"
```

### CLI Arguments Breakdown:
- `--model`: Model architecture (`cnn` or `cnn_lstm`, default: `cnn_lstm`).
- `--audio`: Absolute or relative path to the `.wav` file.
- `--threshold`: Probability threshold $P(\text{scream})$ to trip a candidate window (default: `0.65`).
- `--consec`: Number of consecutive windows above threshold required to confirm an alert (default: `3`).
- `--location`: Human-readable sensor zone tag for forensic audit logging (default: `"CLI Demo"`).

---

## Step 4: Running Live Microphone Audio Inference

To listen live through your computer's default microphone and detect screams in real time:

```powershell
python inference/predictor.py --model cnn --mic
```

With custom sensitivity:
```powershell
python inference/predictor.py --model cnn --mic --threshold 0.55 --consec 2 --location "Guard Station Mic"
```

### Real-Time Terminal Display:
While streaming, your terminal displays a real-time ASCII audio meter:
```text
  [   0.5s] P(scream)=0.012 |                              | Non-Scream
  [   1.0s] P(scream)=0.024 |                              | Non-Scream
  [   1.5s] P(scream)=0.891 |██████████████████████████    | Scream

🚨 ALERT! t=1.50s  P=0.891  [Guard Station Mic]
```
Press **`Ctrl + C`** in the terminal to stop the live microphone stream.

---

## Step 5: Testing Acoustic Outlier & Noise Discrimination

SafeWatch AI features an offline Bio-Acoustic Arbiter that eliminates false alarms from vehicle horns, tire screeches, sirens, and ambient crowd rumble.

### 5A. Test Forensic Engine on a Scream Clip:
```powershell
python -c "from inference.forensic_engine import AcousticForensicEngine; eng=AcousticForensicEngine(); rep=eng.verify_distress('data/raw/scream/---1_cCGK4M_out.wav', cnn_prob=0.88); print('Verdict:', rep.classification); print('Threat Flag:', rep.is_crime_threat); print('Confidence:', rep.threat_confidence); print('Action:', rep.recommended_action); print('Rationale:', rep.forensic_rationale)"
```

### 5B. Test Vehicle Horn Rejection Simulation:
```powershell
python -c "from inference.forensic_engine import AcousticForensicEngine; eng=AcousticForensicEngine(); rep=eng.verify_distress('data/raw/non_scream/--PJHxphWEs_out.wav', cnn_prob=0.35); print('Verdict:', rep.classification); print('Threat Flag:', rep.is_crime_threat); print('Action:', rep.recommended_action)"
```

---

## Step 6: Data Preparation & Feature Extraction

*(Only needed if rebuilding features from raw datasets)*

### 6A. Step 1: Download Datasets (Optional):
```powershell
python utils/download_datasets.py
```

### 6B. Step 2: Build 70/15/15 Recording-Level Split Manifest:
```powershell
python utils/data_prep.py
```
*Creates: `data/features_manifest.csv`*

### 6C. Step 3: Extract Log-Mel Spectrogram Features:
```powershell
python utils/feature_extraction.py
```
*Creates: 103,607 precomputed `.npy` window files in `data/features/`.*

---

## Step 7: Training Neural Network Models

### 7A. Train Baseline CNN:
- **Fast Smoke Test (1 epoch, 30 batches)**:
  ```powershell
  python training/train.py --model cnn --epochs 1 --batch_size 64 --max_batches 30
  ```
- **Full Standard Training (30 epochs)**:
  ```powershell
  python training/train.py --model cnn --epochs 30 --lr 1e-3 --batch_size 64
  ```

### 7B. Train Spatial-Temporal CNN-LSTM:
- **Fast Smoke Test (1 epoch, 20 batches)**:
  ```powershell
  python training/train.py --model cnn_lstm --epochs 1 --batch_size 32 --max_batches 20
  ```
- **Full Standard Training (30 epochs, 5 temporal context windows)**:
  ```powershell
  python training/train.py --model cnn_lstm --epochs 30 --lr 1e-3 --batch_size 64 --n_context 5
  ```

### Training Artifacts Generated:
- `models/checkpoints/cnn_best.pth`
- `models/checkpoints/cnn_lstm_best.pth`
- `outputs/metrics/cnn_train_log.csv` & `outputs/metrics/cnn_curves.png`
- `outputs/metrics/cnn_lstm_train_log.csv` & `outputs/metrics/cnn_lstm_curves.png`

---

## Step 8: Model Evaluation on Held-Out Test Set

Computes unbiased evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Latency) on the strictly un-seen test split.

### 8A. Evaluate Baseline CNN:
```powershell
python training/evaluate.py --model cnn
```

### 8B. Evaluate Spatial-Temporal CNN-LSTM:
```powershell
python training/evaluate.py --model cnn_lstm --n_context 5
```

### Generated Reports & Plots:
- `outputs/metrics/cnn_test_results.json`
- `outputs/metrics/cnn_confusion_matrix.png`
- `outputs/metrics/cnn_pr_curve.png`
- `outputs/metrics/cnn_lstm_test_results.json`
- `outputs/metrics/cnn_lstm_confusion_matrix.png`
- `outputs/metrics/cnn_lstm_pr_curve.png`

---

## Step 9: Inspecting Generated Threat Logs & Metrics

### 9A. View Recent Threat Alert Records:
```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/alerts/alert_log.csv'); print(df.tail(10)[['timestamp', 'model_type', 'threat_score', 'severity', 'classification', 'location']])"
```

### 9B. View Evaluation Test Metrics in Terminal:
```powershell
python -c "import json; data=json.load(open('outputs/metrics/cnn_test_results.json')); print(json.dumps(data, indent=2))"
```

### 9C. Check Precomputed Window Split Counts:
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/features_manifest.csv'); print(df['split'].value_counts())"
```

---

## Troubleshooting & Windows Specific Fixes

### 1. "Port 8501 is already in use"
If Streamlit says port 8501 is blocked:
```powershell
# 1. Find process holding port 8501:
netstat -ano | findstr 8501

# 2. Terminate the process by PID:
taskkill /F /PID <PID_NUMBER>

# 3. Or launch Streamlit on a different port:
streamlit run dashboard/app.py --server.port 8502
```

### 2. "ModuleNotFoundError: No module named 'librosa'" (or similar)
Make sure you are running the python environment where requirements were installed:
```powershell
python -m pip install -r requirements.txt
```

### 3. Microphone Error: `No Default Input Device Available`
- Open Windows **Settings → Privacy & security → Microphone**.
- Ensure **"Microphone access"** and **"Let desktop apps access your microphone"** are toggled **ON**.

### 4. Git Push & Status Check:
```powershell
git status
git add .
git commit -m "Your commit message"
git push origin main
```
