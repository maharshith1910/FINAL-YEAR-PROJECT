@echo off
title SafeWatch AI — Full Pipeline
color 0B

echo.
echo  ██████████████████████████████████████████████████████
echo   SafeWatch AI  ^|  Deep Learning Threat Detection
echo   Final Year Project — Maharshith ^& Team
echo  ██████████████████████████████████████████████████████
echo.

:: ── STEP 0: Install deps ──────────────────────────────────────────────────
echo [STEP 0/6]  Installing Python dependencies ...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌  pip install failed. Check your Python/pip installation.
    pause & exit /b 1
)
echo  ✅  Dependencies installed.
echo.

:: ── STEP 1: Kaggle setup ─────────────────────────────────────────────────
echo [STEP 1/6]  Kaggle API setup (skip if already done) ...
echo             Run: python utils/setup_kaggle.py
echo             Then re-run this script.
echo.
set /p KAGGLE_READY="Have you set up kaggle.json? (y/n): "
if /i NOT "%KAGGLE_READY%"=="y" (
    python utils/setup_kaggle.py
    echo.
    echo Please re-run run_all.bat after pasting your token.
    pause & exit /b 0
)

:: ── STEP 2: Download datasets ─────────────────────────────────────────────
echo.
echo [STEP 2/6]  Downloading datasets (Kaggle Scream + ESC-50) ...
echo             This may take 10-30 min depending on connection speed.
python utils/download_datasets.py
if errorlevel 1 (
    echo ❌  Dataset download encountered issues. Check output above.
    echo     You can continue if data/raw/scream and data/raw/non_scream are populated.
    pause
)
echo.

:: ── STEP 3: Data preparation ──────────────────────────────────────────────
echo [STEP 3/6]  Preparing data manifest and train/val/test split ...
python utils/data_prep.py
if errorlevel 1 ( echo ❌  data_prep.py failed. & pause & exit /b 1 )
echo  ✅  Manifest created.
echo.

:: ── STEP 4: Feature extraction ────────────────────────────────────────────
echo [STEP 4/6]  Extracting Mel spectrogram features ...
python utils/feature_extraction.py
if errorlevel 1 ( echo ❌  feature_extraction.py failed. & pause & exit /b 1 )
echo  ✅  Features extracted.
echo.

:: ── STEP 5: Training ─────────────────────────────────────────────────────
echo [STEP 5/6]  Training models ...
echo.
echo  [5a] Training Baseline CNN (30 epochs) ...
python training/train.py --model cnn --epochs 30
echo.
echo  [5b] Training CNN-LSTM (30 epochs) ...
python training/train.py --model cnn_lstm --epochs 30
echo.

:: ── STEP 5b: Evaluation ──────────────────────────────────────────────────
echo [5c]  Evaluating models on test set ...
python training/evaluate.py --model cnn
python training/evaluate.py --model cnn_lstm
echo  ✅  Evaluation complete. Results in outputs/metrics/
echo.

:: ── STEP 6: Launch dashboard ─────────────────────────────────────────────
echo [STEP 6/6]  Launching SafeWatch AI Dashboard ...
echo             Open your browser at  http://localhost:8501
echo.
streamlit run dashboard/app.py

pause
