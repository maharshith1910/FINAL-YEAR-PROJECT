@echo off
echo ============================================
echo  SafeWatch AI - Installing dependencies...
echo ============================================
pip install -r requirements.txt
echo.
echo ============================================
echo  STEP 1: Prepare your data first!
echo    Place .wav files in:
echo      data\raw\scream\
echo      data\raw\non_scream\
echo ============================================
pause
echo.
echo  Running data preparation...
python utils/data_prep.py
echo.
echo  Running feature extraction...
python utils/feature_extraction.py
echo.
echo  Training CNN (baseline)...
python training/train.py --model cnn --epochs 30
echo.
echo  Training CNN-LSTM...
python training/train.py --model cnn_lstm --epochs 30
echo.
echo  Evaluating both models...
python training/evaluate.py --model cnn
python training/evaluate.py --model cnn_lstm
echo.
echo ============================================
echo  Launching Dashboard...
echo ============================================
streamlit run dashboard/app.py
