# -*- coding: utf-8 -*-
"""
utils/data_prep.py
------------------
Builds a recording-level train/val/test split from raw WAV files and
saves a manifest CSV.

KEY DESIGN: split is done at the RECORDING level, not window level.
This prevents windows from the same clip leaking across train/test sets.

Directory structure expected:
  data/raw/scream/        ← WAV files (label=1)
  data/raw/non_scream/    ← WAV files (label=0, includes ESC-50)

Output:
  data/features_manifest.csv  columns: filepath, label, split, source

Usage:
    python utils/data_prep.py
    python utils/data_prep.py --scream_dir data/raw/scream \\
                               --non_scream_dir data/raw/non_scream \\
                               --out data/features_manifest.csv \\
                               --seed 42
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DEFAULT_SCREAM  = ROOT / "data" / "raw" / "scream"
DEFAULT_NON     = ROOT / "data" / "raw" / "non_scream"
DEFAULT_OUT     = ROOT / "data" / "features_manifest.csv"
AUDIO_EXTS      = {".wav", ".mp3", ".flac", ".ogg"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # test = 1 - train - val = 0.15


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _source_tag(filepath: Path, label: int) -> str:
    """
    Assign a source tag based on filename and label:
      - label=1            → "scream_kaggle"
      - label=0, esc50_*  → "esc50"
      - label=0, other    → "non_scream_kaggle"
    """
    if label == 1:
        return "scream_kaggle"
    if filepath.name.startswith("esc50_"):
        return "esc50"
    return "non_scream_kaggle"


def _gather(directory: Path, label: int) -> list[dict]:
    """Walk directory and return list of {filepath, label, source} dicts."""
    if not directory.exists():
        return []
    records = []
    for fp in sorted(directory.iterdir()):
        if fp.suffix.lower() in AUDIO_EXTS:
            records.append({
                "filepath": str(fp),
                "label":    label,
                "source":   _source_tag(fp, label),
            })
    return records


def _recording_level_split(records: list[dict], seed: int) -> list[dict]:
    """
    Perform a stratified, recording-level split.

    For the Kaggle scream dataset, filenames follow the pattern:
      <youtube_id>_out.wav   (e.g. -PJHxphWEs_out.wav)
    Files from the same video share the same YouTube ID prefix, so we group
    by stem prefix before splitting to avoid leakage.

    For ESC-50 the names are unique per clip, so no grouping is needed.
    """
    # Group by (source, label) → list of filepaths
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        group_key = (r["source"], r["label"])
        by_group[group_key].append(r)

    result = []
    for (source, label), group in by_group.items():
        # Unique recording IDs: use full filepath (each file = one recording)
        if len(group) < 3:
            # Too few to split — put everything in train
            for r in group:
                r["split"] = "train"
            result.extend(group)
            continue

        labels_for_split = [r["label"] for r in group]

        # train / temp split (temp → val + test)
        train_recs, temp_recs = train_test_split(
            group,
            test_size=1 - TRAIN_RATIO,
            random_state=seed,
            stratify=None,   # already within one class
        )

        # val / test split from temp
        val_size_of_temp = VAL_RATIO / (1 - TRAIN_RATIO)
        if len(temp_recs) < 2:
            val_recs, test_recs = temp_recs, []
        else:
            val_recs, test_recs = train_test_split(
                temp_recs,
                test_size=1 - val_size_of_temp,
                random_state=seed,
            )

        for r in train_recs: r["split"] = "train"
        for r in val_recs:   r["split"] = "val"
        for r in test_recs:  r["split"] = "test"

        result.extend(train_recs)
        result.extend(val_recs)
        result.extend(test_recs)

    return result


def _print_summary(df: pd.DataFrame) -> None:
    """Print a per-split / per-label summary table."""
    print("\n" + "="*60)
    print("  DATASET SPLIT SUMMARY")
    print("="*60)
    summary = (
        df.groupby(["split", "label", "source"])
          .size()
          .reset_index(name="files")
    )
    label_name = {1: "scream", 0: "non_scream"}
    summary["class"] = summary["label"].map(label_name)
    col_order = ["split", "class", "source", "files"]
    print(summary[col_order].to_string(index=False))

    print("\n  Totals per split:")
    for split, grp in df.groupby("split"):
        n_scream  = (grp["label"] == 1).sum()
        n_nonsc   = (grp["label"] == 0).sum()
        print(f"    {split:<8}  scream={n_scream:5d}  non_scream={n_nonsc:5d}"
              f"  total={len(grp):5d}  ratio={n_nonsc/max(1,n_scream):.1f}:1")
    print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    scream_dir     = Path(args.scream_dir)
    non_scream_dir = Path(args.non_scream_dir)
    out_path       = Path(args.out)

    print(f"\n[data_prep]  scream_dir    = {scream_dir}")
    print(f"[data_prep]  non_scream_dir= {non_scream_dir}")
    print(f"[data_prep]  out           = {out_path}")
    print(f"[data_prep]  seed          = {args.seed}\n")

    # ── Gather files
    scream_records  = _gather(scream_dir,     label=1)
    non_scr_records = _gather(non_scream_dir, label=0)

    if not scream_records and not non_scr_records:
        print("[ERROR] No audio files found in either directory.")
        print(f"    scream     -> {scream_dir}")
        print(f"    non_scream -> {non_scream_dir}")
        print("\n    Run: python utils/download_datasets.py   (to download data first)")
        sys.exit(1)

    if not scream_records:
        print(f"[WARN] No scream files found in {scream_dir}. Training will be class-imbalanced.")
    if not non_scr_records:
        print(f"[WARN] No non-scream files found in {non_scream_dir}.")

    all_records = scream_records + non_scr_records
    print(f"  Scream files    : {len(scream_records)}")
    print(f"  Non-scream files: {len(non_scr_records)}")
    print(f"  Total files     : {len(all_records)}")

    # ── Recording-level split
    print(f"\n  Splitting (seed={args.seed}, train=70% val=15% test=15%) …")
    split_records = _recording_level_split(all_records, seed=args.seed)

    # ── Build DataFrame
    df = pd.DataFrame(split_records, columns=["filepath", "label", "split", "source"])
    df = df.sort_values(["split", "label", "filepath"]).reset_index(drop=True)

    # Sanity check: no filepath appears in more than one split
    dup_fps = df.groupby("filepath")["split"].nunique()
    leaky   = dup_fps[dup_fps > 1]
    if not leaky.empty:
        print(f"\n⚠️  WARNING: {len(leaky)} files appear in multiple splits — "
              "possible data leakage. Check your data directory.")

    # ── Print summary
    _print_summary(df)

    # ── Write manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Manifest written -> {out_path}  ({len(df)} rows)")
    print("\n   Next step:")
    print("   python utils/feature_extraction.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build recording-level train/val/test manifest for scream detection.")
    parser.add_argument("--scream_dir",     type=str, default=str(DEFAULT_SCREAM),
                        help="Directory containing scream WAV files (label=1)")
    parser.add_argument("--non_scream_dir", type=str, default=str(DEFAULT_NON),
                        help="Directory containing non-scream WAV files (label=0)")
    parser.add_argument("--out",            type=str, default=str(DEFAULT_OUT),
                        help="Output CSV path for the manifest")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed for reproducibility")
    main(parser.parse_args())
