"""
inference/predictor.py
-----------------------
Real-time inference engine.

Handles:
  1. File-based inference: process a .wav file and return per-window scores
  2. Microphone streaming: continuous inference with a ring buffer
  3. Consecutive-window smoothing: only fires alert when N windows in a row
     exceed the threshold (drastically reduces false alarms)

Usage (standalone):
    python inference/predictor.py --model cnn_lstm --audio path/to/file.wav
    python inference/predictor.py --model cnn_lstm --mic        # live mic
"""

import sys
import time
import queue
import threading
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ScreamCNN
from models.cnn_lstm_model import ScreamCNNLSTM
from inference.alert_logger import AlertLogger

# ─── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
WINDOW_SEC     = 1.0
HOP_SEC        = 0.5
N_MELS         = 128
N_FFT          = 2048
HOP_LENGTH     = 512
TARGET_FRAMES  = 44
FMIN, FMAX     = 20, 8000
CHECKPOINT_DIR = Path("models/checkpoints")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alert firing config
DEFAULT_THRESHOLD       = 0.65   # P(scream) must exceed this
DEFAULT_CONSEC_WINDOWS  = 3      # consecutive windows above threshold = alert


# ─── Feature pipeline ──────────────────────────────────────────────────────────

def extract_mel(y: np.ndarray) -> np.ndarray:
    """Convert waveform window → normalised log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad / crop to fixed width
    _, T = log_mel.shape
    if T < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - T)), constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    # Normalise
    mean = log_mel.mean(); std = log_mel.std() + 1e-8
    log_mel = (log_mel - mean) / std
    return log_mel   # (N_MELS, TARGET_FRAMES)


# ─── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_type: str) -> Optional[torch.nn.Module]:
    ckpt_path = CHECKPOINT_DIR / f"{model_type}_best.pth"
    if not ckpt_path.exists():
        print(f"[Predictor] ⚠️  Checkpoint not found: {ckpt_path}")
        print("             Train first: python training/train.py --model", model_type)
        return None

    if model_type == "cnn":
        model = ScreamCNN(n_mels=N_MELS, num_classes=2)
    else:
        model = ScreamCNNLSTM(n_mels=N_MELS, num_classes=2)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"[Predictor] ✅ Loaded {model_type} checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model


# ─── Predictor class ───────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    timestamp_sec: float
    scream_prob:   float
    is_alert:      bool
    label:         str


class ScreamPredictor:
    """
    Stateful predictor with consecutive-window smoothing.

    Args:
        model_type          : 'cnn' or 'cnn_lstm'
        threshold           : P(scream) above which a window is "positive"
        consecutive_windows : how many consecutive positives before an alert fires
        n_context           : context length for CNN-LSTM
        on_alert            : optional callback fn(WindowResult) on alert
        location_label      : descriptive location string for logging
    """

    def __init__(
        self,
        model_type:          str              = "cnn_lstm",
        threshold:           float            = DEFAULT_THRESHOLD,
        consecutive_windows: int              = DEFAULT_CONSEC_WINDOWS,
        n_context:           int              = 5,
        on_alert:            Optional[Callable] = None,
        location_label:      str              = "Unknown Location",
    ):
        self.model_type          = model_type
        self.threshold           = threshold
        self.consec_windows      = consecutive_windows
        self.n_context           = n_context
        self.on_alert            = on_alert
        self.location_label      = location_label

        self.model               = load_model(model_type)
        self.alert_logger        = AlertLogger()

        # Smoothing state
        self._consecutive_count  = 0
        self._window_buffer: list[np.ndarray] = []  # for CNN-LSTM context

    def predict_window(self, mel: np.ndarray) -> float:
        """Score a single Mel spectrogram window. Returns P(scream)."""
        if self.model is None:
            return 0.0

        if self.model_type == "cnn":
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)   # (1,1,H,W)
            with torch.no_grad():
                prob = F.softmax(self.model(x), dim=1)[0, 1].item()

        else:  # cnn_lstm
            self._window_buffer.append(mel)
            if len(self._window_buffer) > self.n_context:
                self._window_buffer.pop(0)

            if len(self._window_buffer) < self.n_context:
                return 0.0   # not enough context yet

            seq = np.stack(self._window_buffer, axis=0)   # (n_context, H, W)
            x   = torch.from_numpy(seq).unsqueeze(1).unsqueeze(0).to(DEVICE)  # (1, n, 1, H, W)
            with torch.no_grad():
                prob = F.softmax(self.model(x), dim=1)[0, 1].item()

        return prob

    def process_window(self, mel: np.ndarray, timestamp_sec: float) -> WindowResult:
        """Score a window and decide whether to fire an alert."""
        prob      = self.predict_window(mel)
        is_pos    = prob >= self.threshold

        if is_pos:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        is_alert  = self._consecutive_count >= self.consec_windows
        label     = "SCREAM DETECTED" if is_alert else ("Suspicious" if is_pos else "Normal")

        result = WindowResult(
            timestamp_sec = timestamp_sec,
            scream_prob   = prob,
            is_alert      = is_alert,
            label         = label,
        )

        if is_alert:
            self._consecutive_count = 0  # reset after firing
            self.alert_logger.log(
                location    = self.location_label,
                confidence  = prob,
                model_used  = self.model_type,
                timestamp   = timestamp_sec,
            )
            if self.on_alert:
                self.on_alert(result)

        return result

    # ── File inference ─────────────────────────────────────────────────────────

    def process_file(self, filepath: str) -> list[WindowResult]:
        """Slice a .wav file into windows and score each one."""
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        results  = []
        win_len  = int(WINDOW_SEC  * SAMPLE_RATE)
        hop_len  = int(HOP_SEC     * SAMPLE_RATE)
        start    = 0

        while start + win_len <= len(y):
            window     = y[start : start + win_len]
            mel        = extract_mel(window)
            ts         = start / SAMPLE_RATE
            result     = self.process_window(mel, ts)
            results.append(result)
            start     += hop_len

        return results

    # ── Microphone streaming ───────────────────────────────────────────────────

    def start_mic_stream(self, duration_sec: Optional[float] = None):
        """Live microphone inference. Press Ctrl+C to stop."""
        try:
            import sounddevice as sd
        except ImportError:
            print("❌ Install sounddevice: pip install sounddevice")
            return

        audio_queue = queue.Queue()
        win_len     = int(WINDOW_SEC * SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            audio_queue.put(indata[:, 0].copy())

        print(f"🎙️  Starting microphone stream (threshold={self.threshold:.2f}, "
              f"consecutive={self.consec_windows} windows)...")
        print("   Press Ctrl+C to stop.\n")

        ring_buffer = np.zeros(win_len * 2, dtype=np.float32)
        elapsed_sec = 0.0
        start_time  = time.time()

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=int(HOP_SEC * SAMPLE_RATE), callback=callback):
            try:
                while True:
                    chunk = audio_queue.get(timeout=5.0)
                    ring_buffer = np.roll(ring_buffer, -len(chunk))
                    ring_buffer[-len(chunk):] = chunk

                    window = ring_buffer[-win_len:]
                    mel    = extract_mel(window)
                    ts     = time.time() - start_time
                    result = self.process_window(mel, ts)

                    bar   = "█" * int(result.scream_prob * 30)
                    print(f"  [{ts:6.1f}s] P(scream)={result.scream_prob:.3f} |{bar:<30}| {result.label}")

                    if duration_sec and (time.time() - start_time) >= duration_sec:
                        break

            except (KeyboardInterrupt, queue.Empty):
                print("\n🛑 Stream stopped.")


# ─── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str,   default="cnn_lstm", choices=["cnn", "cnn_lstm"])
    parser.add_argument("--audio",     type=str,   default=None,  help="Path to .wav file")
    parser.add_argument("--mic",       action="store_true",        help="Use microphone input")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--consec",    type=int,   default=3)
    parser.add_argument("--location",  type=str,   default="CLI Demo")
    args = parser.parse_args()

    predictor = ScreamPredictor(
        model_type          = args.model,
        threshold           = args.threshold,
        consecutive_windows = args.consec,
        location_label      = args.location,
        on_alert            = lambda r: print(f"\n🚨 ALERT! t={r.timestamp_sec:.2f}s  P={r.scream_prob:.3f}\n"),
    )

    if args.mic:
        predictor.start_mic_stream()
    elif args.audio:
        results = predictor.process_file(args.audio)
        alerts  = [r for r in results if r.is_alert]
        print(f"\nProcessed {len(results)} windows. Alerts fired: {len(alerts)}")
        for a in alerts:
            print(f"  🚨 t={a.timestamp_sec:.2f}s  P={a.scream_prob:.3f}")
    else:
        print("Provide --audio <path> or --mic")
