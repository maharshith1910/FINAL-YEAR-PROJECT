"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SAFEWATCH AI — THREE CODEX PROMPTS FOR TEAMMATES                 ║
║    Deep Learning Human Threat Detection · Final Year Project               ║
║    Copy-paste each prompt into your AI IDE (Codex/Claude/Cursor/etc.)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERVIEW
--------
The project is a two-model scream detection system:
  - Branch 1: Baseline CNN on Mel spectrograms
  - Branch 2: CNN-LSTM for temporal context
  - Dataset: Kaggle Human Screaming Detection Dataset (MIT) + ESC-50
  - Pipeline: Data → Features → Train → Evaluate → Dashboard
  - Tech: PyTorch 2.x, torchaudio, librosa, audiomentations, Streamlit

FOLDER STRUCTURE
----------------
FINAL-YEAR-PROJECT/
├── data/
│   ├── raw/
│   │   ├── scream/         ← Kaggle Screaming WAVs
│   │   └── non_scream/     ← Kaggle NotScreaming + ESC-50 WAVs
│   ├── features/           ← .npy Mel spectrogram files (auto-created)
│   └── features_manifest.csv
├── models/
│   ├── cnn_model.py        ← Baseline CNN
│   ├── cnn_lstm_model.py   ← CNN + LSTM
│   └── checkpoints/        ← best .pth files
├── training/
│   ├── dataset.py          ← PyTorch datasets
│   ├── train.py            ← training loop
│   └── evaluate.py         ← metrics + plots
├── inference/
│   ├── predictor.py        ← inference engine
│   └── alert_logger.py     ← CSV alert logger
├── utils/
│   ├── data_prep.py        ← manifest builder + split
│   ├── feature_extraction.py ← Mel extraction
│   ├── setup_kaggle.py     ← Kaggle token wizard
│   └── download_datasets.py ← dataset downloader
├── dashboard/app.py        ← Streamlit dashboard
├── outputs/metrics/        ← JSON results, confusion matrix PNGs
└── requirements.txt

═══════════════════════════════════════════════════════════════════════════════
PROMPT 1 — DATA PIPELINE (Teammate responsible for data prep + features)
═══════════════════════════════════════════════════════════════════════════════

Paste this into Codex/Claude:

---

You are helping build a deep learning scream detection system for a final year
project. The system detects human screams in audio files using Mel spectrograms.
I need you to implement two Python scripts exactly as described below.

CONSTRAINT: Never fabricate data or hardcode fake file paths. If a folder is
empty, the script must print an error and exit cleanly.

═══════════════
Script 1: utils/data_prep.py
═══════════════

Build a recording-level train/val/test split from WAV files and save a CSV manifest.

Requirements:
- Scan  data/raw/scream/  (label=1) and  data/raw/non_scream/  (label=0)
- Split at the RECORDING level (not window level). Group files by filename stem
  prefix — files from the same YouTube video ID share a prefix. Use 70% train /
  15% val / 15% test by file count.
- Use random seed 42 for reproducibility.
- Write  data/features_manifest.csv  with columns:
    filepath, label, split, source
  where source is "scream_kaggle" (for scream/), "non_scream_kaggle" (for
  NotScreaming files), or "esc50" (for files starting with "esc50_").
- Print a summary table showing class balance per split.
- Add argparse: --scream_dir, --non_scream_dir, --out, --seed

Tech: Python stdlib + pandas + scikit-learn (train_test_split).
Target Python: 3.10+

═══════════════
Script 2: utils/feature_extraction.py
═══════════════

Extract log-Mel spectrogram features from every file in the manifest and save
as .npy files. This precomputes features so training is fast.

Requirements:
- Read  data/features_manifest.csv  produced by data_prep.py
- For each WAV file:
    1. Load with librosa (sr=22050, mono=True)
    2. Pad or crop to exactly 10 seconds
    3. Cut into 1-second windows with 0.5-second hop (so ~19 windows per clip)
    4. For each window compute log-Mel spectrogram:
         n_mels=128, n_fft=1024, hop_length=256, fmax=8000
    5. Normalise each window: (x - mean) / (std + 1e-6)
    6. Save as  data/features/<original_stem>_w{i:03d}.npy   (shape 1×128×44)
- Update/create  data/features_manifest.csv  with new columns:
    feature_path, window_index, timestamp_sec
  (one row per window, preserving original label/split/source columns)
- Show a tqdm progress bar.
- Skip files that already have features (to allow resuming).
- Add argparse: --manifest, --feat_dir, --sr, --n_mels, --dry_run

Tech: librosa, numpy, tqdm, pandas.
Handle edge cases: files shorter than 1 window (duplicate-pad), corrupt WAVs
(catch exception, print warning, skip).

IMPORTANT: Add at the bottom:
  if __name__ == "__main__":
      # Verify 5 random feature files load correctly, print shape
      ...

---

═══════════════════════════════════════════════════════════════════════════════
PROMPT 2 — MODEL TRAINING (Teammate responsible for training/train.py)
═══════════════════════════════════════════════════════════════════════════════

Paste this into Codex/Claude:

---

You are helping build a scream detection deep learning system. I need you to
implement training/train.py. The models and evaluation script are already built
and verified.

IMPORTANT RULES:
- NEVER hardcode or fabricate metrics. All numbers must come from actual forward
  passes on real data.
- Use weighted CrossEntropyLoss to handle the ~5:1 non-scream:scream imbalance.

═══════════════
File 1: models/cnn_model.py
═══════════════

Implement ScreamCNN class (nn.Module).

Architecture:
  Input: Tensor of shape (B, 1, 128, 44)  — batch × channel × mel_bins × frames
  - Block 1: Conv2d(1,32,3, padding=1) → BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.25)
  - Block 2: Conv2d(32,64,3,padding=1) → BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.25)
  - Block 3: Conv2d(64,128,3,padding=1)→ BatchNorm → ReLU → MaxPool(2,2) → Dropout(0.25)
  - Block 4: Conv2d(128,256,3,padding=1)→BatchNorm → ReLU → AdaptiveAvgPool(1,1)
  - Flatten → Linear(256,128) → ReLU → Dropout(0.5) → Linear(128, num_classes)
  Forward returns raw logits.
  Add a property: feature_dim = 256

═══════════════
File 2: models/cnn_lstm_model.py
═══════════════

Implement ScreamCNNLSTM class (nn.Module).

Architecture:
  Input: Tensor of shape (B, T, 1, 128, 44)  — batch × time_steps × channel × mel × frames
  - CNN feature extractor: same as ScreamCNN blocks 1-4 but returning 256-dim features
    (reuse the CNN as a sub-module, load pretrained CNN weights if path is given)
  - For each time step t, pass frame through CNN extractor → feature vector (B, 256)
  - Stack T feature vectors → (B, T, 256)
  - LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.3, batch_first=True)
  - Take last hidden state → Linear(256, num_classes)
  Forward returns raw logits.
  Constructor: ScreamCNNLSTM(n_mels=128, num_classes=2, cnn_ckpt_path=None)
    If cnn_ckpt_path is given and exists, load CNN weights into the extractor.

═══════════════
File 3: training/evaluate.py
═══════════════

Evaluation script that runs inference on the test split and saves results.

Requirements:
- argparse: --model {cnn, cnn_lstm}, --manifest, --feat_dir, --ckpt_dir,
  --batch_size, --n_context (for cnn_lstm)
- Load the best checkpoint from  models/checkpoints/{model}_best.pth
- Run inference on the TEST split of  data/features_manifest.csv
- Compute and save to  outputs/metrics/{model}_test_results.json :
    accuracy, precision, recall, f1, false_positive_rate,
    confusion_matrix (2×2 list), avg_latency_ms,
    note: "Computed on held-out test set. No data fabrication."
- Save confusion matrix plot as  outputs/metrics/{model}_confusion_matrix.png
- Save PR curve as                outputs/metrics/{model}_pr_curve.png
- Print a clean summary table to stdout.
- Use sklearn.metrics for all metric calculations.

Class weights for loss: compute from train split label distribution.
  weight[class] = total / (n_classes * count[class])
  Pass as weight= argument to nn.CrossEntropyLoss.

---

═══════════════════════════════════════════════════════════════════════════════
PROMPT 3 — DASHBOARD & INFERENCE (Teammate responsible for the app + demo)
═══════════════════════════════════════════════════════════════════════════════

Paste this into Codex/Claude:

---

You are helping build the inference and dashboard layer of a scream detection
system. The models are already trained. I need two scripts.

RULES:
- If checkpoint is missing: show a clear error in the UI — never raise an
  unhandled exception.
- Never fabricate detections. If model isn't loaded, return empty results.

═══════════════
File 1: inference/predictor.py
═══════════════

Implement ScreamPredictor class.

Constructor args:
  model_type: str          — "cnn" or "cnn_lstm"
  threshold: float         — P(scream) to flag a window (default 0.65)
  consecutive_windows: int — N windows above threshold before alert fires (default 3)
  location_label: str      — location string for logging

Loads checkpoint from  models/checkpoints/{model_type}_best.pth
Raises FileNotFoundError with clear message if missing.

Method: process_file(audio_path: str) → list[WindowResult]
  WindowResult is a dataclass with fields:
    timestamp_sec: float
    scream_prob: float
    is_alert: bool
    label: str           — "SCREAM" / "CLEAR" / "ALERT"

  Processing:
    1. Load WAV with librosa (sr=22050, mono=True)
    2. Cut into 1-second windows with 0.5s hop
    3. For CNN: batch all windows → softmax → P(scream)
    4. For CNN-LSTM: batch as sequences of n_context=5 windows
    5. Apply consecutive-window smoothing for is_alert
    6. Log each ALERT event via AlertLogger

Also export: extract_mel(audio_path, sr=22050, n_mels=128) → np.ndarray
  Returns a single Mel spectrogram (normalised) for one file.

═══════════════
File 2: inference/alert_logger.py
═══════════════

Implement AlertLogger class.

Stores alert records to  outputs/alerts/alert_log.csv

Fields per alert:
  alert_id, timestamp_iso, timestamp_epoch, location, confidence,
  severity (HIGH ≥0.85 / MEDIUM ≥0.65 / LOW otherwise),
  model_used, verification_status (default "PENDING"), note

Methods:
  log_alert(location, confidence, model_used) → alert_id
  load_all() → pd.DataFrame   (returns empty DataFrame if file missing)
  stats() → dict with keys: total, high, medium, pending

Create  outputs/alerts/  directory automatically.
Use file locking (fcntl on Linux / msvcrt on Windows) for concurrent safety.

---

═══════════════════════════════════════════════════════════════════════════════
QUICK REFERENCE — COMMANDS IN ORDER
═══════════════════════════════════════════════════════════════════════════════

# 1. One-time setup
python utils/setup_kaggle.py          # paste kaggle.json token
python utils/download_datasets.py     # ~6 GB total, takes 15-30 min
python utils/data_prep.py             # build manifest, split by recording
python utils/feature_extraction.py    # extract Mel .npy files

# 2. Training (Teammate A)
python training/train.py --model cnn      --epochs 30 --lr 1e-3
python training/train.py --model cnn_lstm --epochs 30 --lr 1e-3

# 3. Evaluation (Teammate A)
python training/evaluate.py --model cnn
python training/evaluate.py --model cnn_lstm

# 4. Dashboard (Teammate B)
streamlit run dashboard/app.py
# → Open http://localhost:8501
# → Sidebar: Load Model → Upload WAV → Inspect Timeline

# OR run everything at once:
run_all.bat

═══════════════════════════════════════════════════════════════════════════════
VIVA TALKING POINTS
═══════════════════════════════════════════════════════════════════════════════

Q: Why Mel spectrograms?
   "Mel scale approximates human auditory perception. Log compression
    reduces dynamic range. 128 bins × 44 frames gives ~1-second coverage
    with adequate time-frequency resolution for scream vs. speech separation."

Q: Why CNN + LSTM?
   "CNN extracts spatial (spectral) features per frame. LSTM captures
    temporal dynamics across frames — screams have distinctive onset patterns
    that single-frame models miss."

Q: Why not 3D CNN?
   "3D CNNs need more data and GPU memory. With ~862 scream clips,
    the CNN+LSTM architecture is better regularised and trains faster,
    which suits our 48-hour deadline. We list 3D CNN as future work."

Q: What's the class imbalance strategy?
   "We use weighted CrossEntropyLoss (weight ∝ 1/class_frequency) and
    report F1/Recall rather than accuracy to fairly evaluate the minority class."

Q: What's the FPR, and why does it matter?
   "False Positive Rate = FP / (FP + TN). In a surveillance context,
    high FPR means alert fatigue — operators stop trusting the system.
    We tune the threshold on the validation set to balance FPR vs. Recall."

Q: Ethical considerations?
   "We never identify individuals. All alerts are labelled 'potential
    distress events requiring human verification'. Our UI explicitly shows
    this disclaimer on every screen."

═══════════════════════════════════════════════════════════════════════════════
HONEST RESULTS REPORTING
═══════════════════════════════════════════════════════════════════════════════

Expected realistic ranges (do NOT hardcode these — use real eval output):
  - Accuracy: 80–92% (depends on augmentation quality and dataset size)
  - F1 (scream class): 75–88%
  - FPR: 5–15% (higher without ESC-50 negatives)

If your numbers are lower, present them with context:
  "The model achieves X% F1 on the scream class. Major failure modes include
   [e.g. laughter / car horns]. Threshold can be adjusted per deployment
   context to trade FPR for Recall."
"""
