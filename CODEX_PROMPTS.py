# Three Codex Prompts for SafeWatch AI
# ========================================
# Copy each prompt below into your IDE Codex (or Copilot Chat / GPT-4).
# Use ONE at a time. Don't paste all three at once.

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 1 — Data Pipeline & Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_1 = """
You are helping build a deep learning audio scream-detection system for a
university final review project. The codebase is at c:/Users/user/Desktop/Major Project/.

TASK: Extend utils/data_prep.py and utils/feature_extraction.py.

Requirements:
1. data_prep.py already builds a manifest.csv from data/raw/scream/ and
   data/raw/non_scream/. Add a --balance flag that downsamples the majority
   class so the train split is balanced (50/50 scream/non-scream).

2. feature_extraction.py already extracts 128-band log-Mel spectrograms.
   Add an --augment flag that, for scream samples ONLY on the train split,
   also saves pitch-shifted (+/- 2 semitones) and time-stretched (0.85x, 1.15x)
   variants. Each variant gets its own .npy file and row in features_manifest.csv.

3. Add a function verify_manifest(path) that checks:
   - No NaN values in any .npy file
   - Class balance within 10% of 50/50 on the train split
   - No source file appears in more than one split
   Print a pass/fail report.

4. NEVER fabricate data or hardcode metrics. If a file is corrupted, log a
   warning and skip it, don't fill in fake values.

Return only complete, runnable Python code for both files. Include docstrings.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 2 — Training Scripts
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_2 = """
You are helping build a deep learning audio scream-detection system.
Codebase: c:/Users/user/Desktop/Major Project/

The following files already exist (read them before writing):
  models/cnn_model.py          — ScreamCNN class
  models/cnn_lstm_model.py     — ScreamCNNLSTM class
  training/dataset.py          — ScreamDataset and ScreamSequenceDataset
  training/train.py            — Training loop with checkpointing
  training/evaluate.py         — Full evaluation with real metrics

TASK: Improve the training and evaluation scripts.

1. In training/train.py:
   a. Add focal loss as an alternative to CrossEntropy (--loss focal flag).
      Focal loss reduces false positives, which matters for a scream detector.
   b. Add early stopping: stop if val_acc doesn't improve for --patience epochs.
   c. Save a training_summary.json at the end with best epoch, best val_acc,
      total training time in seconds, and the args used.

2. In training/evaluate.py:
   a. Add a --threshold sweep that evaluates at 10 thresholds from 0.3 to 0.9
      and saves a precision_recall_threshold.csv with columns:
      threshold, precision, recall, f1, fpr.
   b. Add a failure case analysis: find the 5 windows with highest
      false-positive probability and the 5 with highest false-negative
      probability. Save their npy_paths to a failure_cases.json.

3. CRITICAL: Include this string in train.py and evaluate.py at the top:
   # INTEGRITY: This script never fabricates metrics. All values are computed
   # from real model outputs on real data. Missing checkpoints cause sys.exit(1).

Return only complete runnable Python for the two modified files.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT 3 — Streamlit Dashboard Integration
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_3 = """
You are helping build a Streamlit dashboard for a deep learning scream
detection system. Codebase: c:/Users/user/Desktop/Major Project/

Files that already exist:
  dashboard/app.py             — Full Streamlit app with 4 tabs
  inference/predictor.py       — ScreamPredictor with file + mic inference
  inference/alert_logger.py    — AlertLogger writing to outputs/alerts/alert_log.csv
  outputs/metrics/             — JSON results + PNG plots from training

TASK: Extend dashboard/app.py with these features:

1. In the "Live Detection" tab (tab1):
   a. Add a DEMO MODE button that, if no real checkpoint is found, loads a
      random sample from data/raw/ (any .wav file) and runs the predictor on it,
      clearly marking results as "DEMO — not real detection".
   b. Display a waveform plot (amplitude vs time) of the uploaded file using
      Plotly alongside the timeline.

2. In the "Model Performance" tab (tab2):
   a. Add a live threshold sweep chart: a Plotly line chart reading from
      outputs/metrics/<model>_precision_recall_threshold.csv if it exists.
      Axes: x=threshold, y=precision/recall/f1 as three separate lines.

3. In the "Incident Log" tab (tab3):
   a. Add a "Mark as Verified" and "Mark as False Alarm" button per row.
      These call alert_logger.update_verification(alert_id, new_status).
   b. Add a timeline chart: x=timestamp_iso, y=confidence, color=severity.

4. General rules:
   - Never hardcode any metric values. Load them from JSON files in outputs/metrics/.
   - If a checkpoint is missing, show a clear warning box, not an error crash.
   - All detections must display the note:
     "Potential distress event — requires human verification."
   - Keep all existing CSS and design intact.

Return only the complete modified dashboard/app.py.
"""

print("Codex prompts ready. Use ONE at a time in your IDE Codex.")
print("Prompt 1: Data pipeline  → paste first")
print("Prompt 2: Training       → paste second")
print("Prompt 3: Dashboard      → paste last")
