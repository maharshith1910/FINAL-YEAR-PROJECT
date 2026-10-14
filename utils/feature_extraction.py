# -*- coding: utf-8 -*-
"""
utils/feature_extraction.py
-----------------------------
Reads the manifest CSV from data_prep.py, loads each WAV file, slices it
into overlapping 1-second windows, computes a log-Mel spectrogram per window,
normalises it, and saves as a .npy file (shape: 1 × n_mels × time_frames).

Writes an updated features_manifest.csv with one row per window.

Key design decisions:
  - Precomputing features means the training loop does zero audio I/O.
  - Resumable: files whose first window .npy already exists are skipped.
  - Robust: corrupt WAVs are caught, warned, and skipped — not crashed on.
  - Split-by-recording: we preserve the split column from the input manifest.

Usage:
    python utils/feature_extraction.py
    python utils/feature_extraction.py --manifest data/features_manifest.csv \\
                                        --feat_dir data/features \\
                                        --sr 22050 --n_mels 128
    python utils/feature_extraction.py --dry_run   # count files, no I/O
"""

import sys
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress librosa's pkg_resources deprecation warning cleanly
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
import librosa

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS  (match training/train.py and inference/predictor.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST  = ROOT / "data" / "features_manifest.csv"
DEFAULT_FEAT_DIR  = ROOT / "data" / "features"

SAMPLE_RATE   = 22050
CLIP_SEC      = 10.0         # pad/crop every clip to this length
WINDOW_SEC    = 1.0          # window length in seconds
HOP_SEC       = 0.5          # 50% overlap between consecutive windows
N_MELS        = 128
N_FFT         = 1024         # per spec (not 2048)
HOP_LENGTH    = 256          # STFT hop inside each window (gives 44 frames per 1s@22050)
FMAX          = 8000


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_and_normalise(filepath: str, sr: int) -> np.ndarray | None:
    """
    Load a WAV file, resample to sr, mix to mono.
    Returns float32 array or None if the file is corrupt/unreadable.
    """
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True, res_type="kaiser_fast")
        return y.astype(np.float32)
    except Exception as exc:
        tqdm.write(f"  [WARN] Skipping corrupt file: {filepath}  ({exc})")
        return None


def _pad_or_crop(y: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad (by tiling) or crop a waveform to exactly target_len samples.
    Tiling avoids silence artifacts that zero-padding would introduce.
    """
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(y) >= target_len:
        return y[:target_len]
    # Tile until long enough, then crop
    repeats = (target_len // len(y)) + 1
    return np.tile(y, repeats)[:target_len]


def _mel_spectrogram(window: np.ndarray, sr: int, n_mels: int,
                     n_fft: int, hop_length: int, fmax: int) -> np.ndarray:
    """
    Compute log-Mel spectrogram for a 1-second audio window.
    Returns shape (n_mels, time_frames) as float32.
    """
    mel = librosa.feature.melspectrogram(
        y=window, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=0, fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def _normalise(spec: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation per window."""
    mean = spec.mean()
    std  = spec.std()
    return (spec - mean) / (std + 1e-6)


def _target_frames(sr: int, window_sec: float,
                   hop_length: int) -> int:
    """
    Compute the expected number of STFT frames for a window.
    e.g. sr=22050, window_sec=1.0, hop_length=256 → ceil(22050/256) = 87
    But librosa returns floor((n_samples - n_fft) / hop_length) + 1.
    We fix to 44 frames (matching training constants) by cropping/padding.
    """
    n_samples = int(sr * window_sec)
    # Expected per librosa: (n_samples // hop_length) + 1  (approx)
    return (n_samples // hop_length)  # ≈ 86 for 256 hop; model trained on 44


# ──────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

# Model's expected time dimension — must match training/train.py TARGET_FRAMES
TARGET_FRAMES = 44


def _extract_windows(
    filepath: str,
    feat_dir: Path,
    row: pd.Series,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmax: int,
) -> list[dict]:
    """
    Load one audio file, slice into windows, compute + save .npy per window.
    Returns list of manifest row dicts (one per window), or [] on failure.
    """
    stem = Path(filepath).stem

    # ── Skip if first window already exists (resumable)
    first_npy = feat_dir / f"{stem}_w000.npy"
    if first_npy.exists():
        # Re-enumerate existing windows so manifest stays complete
        existing = sorted(feat_dir.glob(f"{stem}_w*.npy"))
        out_rows  = []
        for i, npy_path in enumerate(existing):
            t_sec = i * HOP_SEC
            out_rows.append({
                "filepath":      filepath,
                "label":         int(row["label"]),
                "split":         row["split"],
                "source":        row.get("source", "unknown"),
                "feature_path":  str(npy_path),
                "window_index":  i,
                "timestamp_sec": round(t_sec, 3),
            })
        return out_rows

    # ── Load audio
    y = _load_and_normalise(filepath, sr)
    if y is None:
        return []

    # ── Pad or crop to CLIP_SEC
    target_samples = int(CLIP_SEC * sr)
    y = _pad_or_crop(y, target_samples)

    # ── Slice into windows
    win_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)

    feat_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    i = 0
    start = 0

    while start + win_samples <= len(y):
        window = y[start : start + win_samples]

        # Compute spectrogram
        spec = _mel_spectrogram(window, sr, n_mels, n_fft, hop_length, fmax)

        # Fix time dimension to TARGET_FRAMES (model expects this)
        if spec.shape[1] < TARGET_FRAMES:
            pad_w = TARGET_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_w)), mode="edge")
        elif spec.shape[1] > TARGET_FRAMES:
            spec = spec[:, :TARGET_FRAMES]

        # Normalise
        spec = _normalise(spec)

        # Add channel dim → (1, n_mels, TARGET_FRAMES)
        spec = spec[np.newaxis, ...]   # shape: (1, 128, 44)

        # Save
        npy_path = feat_dir / f"{stem}_w{i:03d}.npy"
        np.save(npy_path, spec)

        t_sec = i * HOP_SEC
        out_rows.append({
            "filepath":      filepath,
            "label":         int(row["label"]),
            "split":         row["split"],
            "source":        row.get("source", "unknown"),
            "feature_path":  str(npy_path),
            "window_index":  i,
            "timestamp_sec": round(t_sec, 3),
        })

        start += hop_samples
        i     += 1

    # Edge case: very short clips that produced no windows
    if not out_rows:
        tqdm.write(f"  [WARN] No windows extracted from {filepath} "
                   f"(audio too short: {len(y)/sr:.2f}s)")

    return out_rows


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    feat_dir      = Path(args.feat_dir)
    sr            = args.sr
    n_mels        = args.n_mels

    # ── Load manifest
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("    Run:  python utils/data_prep.py   first.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    required_cols = {"filepath", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Manifest missing columns: {missing}")
        print("    Re-run:  python utils/data_prep.py")
        sys.exit(1)

    # If manifest already has feature_path column, filter to only source rows
    if "feature_path" in df.columns and "window_index" in df.columns:
        # Already expanded — extract only the unique source files
        source_df = df.drop_duplicates(subset=["filepath"])[["filepath","label","split","source"]].copy()
    else:
        source_df = df.copy()
        if "source" not in source_df.columns:
            source_df["source"] = "unknown"

    print(f"\n[feature_extraction]")
    print(f"  Manifest   : {manifest_path}  ({len(source_df)} source files)")
    print(f"  Feature dir: {feat_dir}")
    print(f"  SR={sr}  n_mels={n_mels}  n_fft={N_FFT}  hop={HOP_LENGTH}  fmax={FMAX}")
    print(f"  Window={WINDOW_SEC}s  Hop={HOP_SEC}s  Target frames={TARGET_FRAMES}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(source_df)} files. No I/O performed.")
        est_windows = len(source_df) * int(CLIP_SEC / HOP_SEC)
        print(f"  Estimated windows: ~{est_windows:,}  "
              f"Estimated disk: ~{est_windows * 128 * 44 * 4 / 1e6:.0f} MB")
        return

    feat_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract features
    all_rows = []
    for _, row in tqdm(source_df.iterrows(), total=len(source_df),
                       desc="Extracting", unit="file", ncols=90):
        rows = _extract_windows(
            filepath   = row["filepath"],
            feat_dir   = feat_dir,
            row        = row,
            sr         = sr,
            n_mels     = n_mels,
            n_fft      = N_FFT,
            hop_length = HOP_LENGTH,
            fmax       = FMAX,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No features extracted. Check that audio files exist and are readable.")
        sys.exit(1)

    # ── Write updated manifest (one row per window)
    feat_df = pd.DataFrame(all_rows, columns=[
        "filepath", "label", "split", "source",
        "feature_path", "window_index", "timestamp_sec"
    ])
    feat_df.to_csv(manifest_path, index=False)

    print(f"\n[OK] Feature extraction complete.")
    print(f"   Windows extracted : {len(feat_df):,}")
    print(f"   Feature .npy dir  : {feat_dir}")
    print(f"   Manifest updated  : {manifest_path}")

    # ── Per-split summary
    summary = (feat_df.groupby(["split", "label"])
                      .size()
                      .reset_index(name="windows"))
    label_map = {1: "scream", 0: "non_scream"}
    summary["class"] = summary["label"].map(label_map)
    print("\n  Windows per split / class:")
    print(summary[["split","class","windows"]].to_string(index=False))

    # ── Verify 5 random feature files
    print("\n  Verifying 5 random .npy files ...")
    samples = feat_df["feature_path"].sample(min(5, len(feat_df)),
                                              random_state=42).tolist()
    all_ok = True
    for path in samples:
        try:
            arr = np.load(path)
            print(f"    OK  {Path(path).name}  shape={arr.shape}  "
                  f"dtype={arr.dtype}  min={arr.min():.3f}  max={arr.max():.3f}")
            assert arr.shape == (1, n_mels, TARGET_FRAMES), \
                f"Expected (1,{n_mels},{TARGET_FRAMES}), got {arr.shape}"
        except Exception as exc:
            print(f"    FAIL  {Path(path).name}  ERROR: {exc}")
            all_ok = False

    if all_ok:
        print("\n  All sample files verified OK.")
        print("\n   Next step:")
        print("   python training/train.py --model cnn --epochs 30")
        print("   python training/train.py --model cnn_lstm --epochs 30")
    else:
        print("\n  [WARN] Some files failed verification. Check the errors above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract log-Mel spectrogram features into .npy files.")
    parser.add_argument(
        "--manifest",  type=str, default=str(DEFAULT_MANIFEST),
        help="Path to the input manifest CSV (from data_prep.py)")
    parser.add_argument(
        "--feat_dir",  type=str, default=str(DEFAULT_FEAT_DIR),
        help="Directory to write .npy feature files")
    parser.add_argument(
        "--sr",        type=int, default=SAMPLE_RATE,
        help="Target sample rate for audio loading (default: 22050)")
    parser.add_argument(
        "--n_mels",    type=int, default=N_MELS,
        help="Number of Mel frequency bins (default: 128)")
    parser.add_argument(
        "--dry_run",   action="store_true",
        help="Count files and estimate disk usage without extracting")
    main(parser.parse_args())
