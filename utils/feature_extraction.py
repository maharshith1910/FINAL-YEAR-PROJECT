"""
utils/feature_extraction.py
-----------------------------
Reads manifest.csv, loads each audio file, slices it into 1-second
overlapping windows, converts each window to a 128-band log-Mel
spectrogram, and saves as a .npy array.

Output per file: data/processed/<split>/<label>/<stem>_<idx>.npy
Also writes: data/features_manifest.csv

Usage:
    python utils/feature_extraction.py
    python utils/feature_extraction.py --window_sec 1.0 --hop_sec 0.5 --n_mels 128
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ─── Config defaults ───────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
WINDOW_SEC    = 1.0       # window length in seconds
HOP_SEC       = 0.5       # hop between windows (50% overlap)
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512       # for STFT inside librosa
FMIN          = 20
FMAX          = 8000

MANIFEST_IN   = Path("data/manifest.csv")
PROCESSED_DIR = Path("data/processed")
FEAT_MANIFEST = Path("data/features_manifest.csv")
# ──────────────────────────────────────────────────────────────────────────────


def load_audio(filepath: str, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Load audio file at target sample rate. Returns mono float32 array."""
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return None


def audio_to_melspectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert waveform to log-Mel spectrogram (shape: n_mels × time_frames)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def slice_and_extract(filepath: str, label: int, split: str,
                      window_sec: float, hop_sec: float) -> list[dict]:
    """
    Slice one audio file into windows, extract spectrogram per window.
    Returns list of dicts (npy_path, label, split).
    """
    y = load_audio(filepath)
    if y is None:
        return []

    window_len = int(window_sec * SAMPLE_RATE)
    hop_len    = int(hop_sec * SAMPLE_RATE)
    stem       = Path(filepath).stem

    records = []
    start = 0
    idx   = 0

    while start + window_len <= len(y):
        window = y[start : start + window_len]
        mel    = audio_to_melspectrogram(window)

        # Output directory
        out_dir = PROCESSED_DIR / split / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        npy_path = out_dir / f"{stem}_{idx:04d}.npy"
        np.save(npy_path, mel)

        records.append({
            "npy_path": str(npy_path),
            "label": label,
            "split": split,
            "source_file": filepath,
        })

        start += hop_len
        idx   += 1

    return records


def main(args):
    if not MANIFEST_IN.exists():
        print(f"❌ Manifest not found at {MANIFEST_IN}. Run utils/data_prep.py first.")
        return

    df = pd.read_csv(MANIFEST_IN)
    all_records = []

    print(f"🎵 Extracting features from {len(df)} audio files …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        records = slice_and_extract(
            filepath   = row["filepath"],
            label      = int(row["label"]),
            split      = row["split"],
            window_sec = args.window_sec,
            hop_sec    = args.hop_sec,
        )
        all_records.extend(records)

    feat_df = pd.DataFrame(all_records)
    feat_df.to_csv(FEAT_MANIFEST, index=False)

    print(f"\n✅ Feature extraction complete.")
    print(f"   Windows extracted : {len(feat_df)}")
    print(f"   Features manifest : {FEAT_MANIFEST}")

    # Summary
    summary = feat_df.groupby(["split", "label"]).size().reset_index(name="windows")
    print("\n📊 Window counts per split/label:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mel spectrogram features.")
    parser.add_argument("--window_sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--hop_sec",    type=float, default=HOP_SEC)
    parser.add_argument("--n_mels",     type=int,   default=N_MELS)
    args = parser.parse_args()

    N_MELS = args.n_mels
    main(args)
