"""
training/dataset.py
--------------------
PyTorch Dataset for loading pre-extracted Mel spectrogram .npy files.
Supports:
  - Single-window mode  → for CNN training
  - Sequence mode       → for CNN-LSTM training (N_CONTEXT consecutive windows)
  - On-the-fly augmentation via audiomentations / numpy transforms
"""

import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ─── Spectrogram augmentation (cheap, numpy-based) ────────────────────────────

def time_mask(mel: np.ndarray, max_mask_pct: float = 0.15) -> np.ndarray:
    """Zero out a random band of time frames (SpecAugment)."""
    _, T = mel.shape
    mask_width = random.randint(1, max(1, int(T * max_mask_pct)))
    start = random.randint(0, T - mask_width)
    mel = mel.copy()
    mel[:, start : start + mask_width] = mel.min()
    return mel


def freq_mask(mel: np.ndarray, max_mask_pct: float = 0.15) -> np.ndarray:
    """Zero out a random band of frequency bins (SpecAugment)."""
    F, _ = mel.shape
    mask_width = random.randint(1, max(1, int(F * max_mask_pct)))
    start = random.randint(0, F - mask_width)
    mel = mel.copy()
    mel[start : start + mask_width, :] = mel.min()
    return mel


def add_gaussian_noise(mel: np.ndarray, std: float = 0.5) -> np.ndarray:
    """Add small Gaussian noise to the spectrogram."""
    return mel + np.random.randn(*mel.shape).astype(np.float32) * std


def augment_spectrogram(mel: np.ndarray) -> np.ndarray:
    """Apply random subset of augmentations."""
    if random.random() < 0.5:
        mel = time_mask(mel)
    if random.random() < 0.5:
        mel = freq_mask(mel)
    if random.random() < 0.3:
        mel = add_gaussian_noise(mel)
    return mel


# ─── Normalisation ─────────────────────────────────────────────────────────────

def normalize(mel: np.ndarray) -> np.ndarray:
    """Normalise to zero mean / unit variance (per sample)."""
    mean = mel.mean()
    std  = mel.std() + 1e-8
    return (mel - mean) / std


def pad_or_crop(mel: np.ndarray, target_frames: int = 44) -> np.ndarray:
    """Ensure consistent time-axis size."""
    _, T = mel.shape
    if T < target_frames:
        pad = target_frames - T
        mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant", constant_values=mel.min())
    else:
        mel = mel[:, :target_frames]
    return mel


# ─── Single-window Dataset (CNN) ───────────────────────────────────────────────

class ScreamDataset(Dataset):
    """
    Loads single spectrogram windows for CNN training.

    Args:
        manifest_path : path to features_manifest.csv
        split         : 'train' | 'val' | 'test'
        augment       : apply augmentation (train only)
        target_frames : number of time frames to pad/crop to
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str           = "train",
        augment: bool        = True,
        target_frames: int   = 44,
        feat_dir: str | Path | None = None,
    ):
        df = pd.read_csv(manifest_path)
        self.df            = df[df["split"] == split].reset_index(drop=True)
        self.augment       = augment and (split == "train")
        self.target_frames = target_frames
        self.feat_dir      = Path(feat_dir) if feat_dir else None
        self.path_col      = "feature_path" if "feature_path" in self.df.columns else "npy_path"

        if len(self.df) == 0:
            raise ValueError(f"No data found for split='{split}' in {manifest_path}. "
                             "Have you run feature_extraction.py?")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, feat_path: str) -> Path:
        p = Path(feat_path)
        if not p.exists() and self.feat_dir:
            fallback = self.feat_dir / p.name
            if fallback.exists():
                return fallback
        return p

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_path = self._resolve_path(row[self.path_col])
        mel = np.load(feat_path)
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0)  # (n_mels, T)
        mel = pad_or_crop(mel, self.target_frames)

        if self.augment:
            mel = augment_spectrogram(mel)

        mel   = normalize(mel)
        mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)  # (1, n_mels, T)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return mel_t, label


# ─── Sequence Dataset (CNN-LSTM) ───────────────────────────────────────────────

class ScreamSequenceDataset(Dataset):
    """
    Loads sequences of N_CONTEXT consecutive spectrogram windows.
    Windows are sorted by source_file/filepath + index; sequences that cross
    source-file boundaries are discarded.

    Returns:
        x     : (N_CONTEXT, 1, n_mels, target_frames)
        label : int (label of the LAST window in the sequence)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str           = "train",
        n_context: int       = 5,
        augment: bool        = True,
        target_frames: int   = 44,
        feat_dir: str | Path | None = None,
    ):
        df   = pd.read_csv(manifest_path)
        self.df            = df[df["split"] == split].reset_index(drop=True)
        self.n_context     = n_context
        self.augment       = augment and (split == "train")
        self.target_frames = target_frames
        self.feat_dir      = Path(feat_dir) if feat_dir else None
        self.path_col      = "feature_path" if "feature_path" in self.df.columns else "npy_path"
        self.file_col      = "filepath" if "filepath" in self.df.columns else "source_file"

        # Build valid sequence start indices (within the same source file)
        self.sequences = self._build_sequences()

    def _resolve_path(self, feat_path: str) -> Path:
        p = Path(feat_path)
        if not p.exists() and self.feat_dir:
            fallback = self.feat_dir / p.name
            if fallback.exists():
                return fallback
        return p

    def _build_sequences(self) -> list[list[int]]:
        sequences = []
        if self.file_col not in self.df.columns:
            return sequences
        
        grouped = self.df.groupby(self.file_col, sort=False).indices
        for _, indices in grouped.items():
            if len(indices) >= self.n_context:
                for i in range(len(indices) - self.n_context + 1):
                    sequences.append(indices[i : i + self.n_context].tolist())

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_indices = self.sequences[idx]
        mels = []

        for i in seq_indices:
            row = self.df.iloc[i]
            feat_path = self._resolve_path(row[self.path_col])
            mel = np.load(feat_path)
            if mel.ndim == 3 and mel.shape[0] == 1:
                mel = mel.squeeze(0)
            mel = pad_or_crop(mel, self.target_frames)
            if self.augment:
                mel = augment_spectrogram(mel)
            mel = normalize(mel)
            mels.append(mel)

        # Stack into (N_CONTEXT, n_mels, T)
        x = np.stack(mels, axis=0)
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)   # (N_CONTEXT, 1, n_mels, T)

        # Label from the last window in the sequence
        label_row = self.df.iloc[seq_indices[-1]]
        label     = torch.tensor(int(label_row["label"]), dtype=torch.long)
        return x, label


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MANIFEST = Path("data/features_manifest.csv")
    if MANIFEST.exists():
        ds = ScreamDataset(MANIFEST, split="train")
        x, y = ds[0]
        print(f"CNN sample shape  : {x.shape}, label={y.item()}")

        ds_seq = ScreamSequenceDataset(MANIFEST, split="train", n_context=5)
        if len(ds_seq) > 0:
            x_s, y_s = ds_seq[0]
            print(f"LSTM sample shape : {x_s.shape}, label={y_s.item()}")
        else:
            print("Not enough samples for a sequence. Extract more features.")
    else:
        print("features_manifest.csv not found. Run feature_extraction.py first.")
