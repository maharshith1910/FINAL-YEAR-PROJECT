"""
utils/data_prep.py
------------------
Builds the manifest CSV from raw audio files and performs a
recording-level train/val/test split (NOT random clip split, which
causes data leakage).

Directory structure expected:
  data/raw/scream/       ← .wav files of screams
  data/raw/non_scream/   ← .wav files of non-scream sounds

Output:
  data/manifest.csv  with columns: filepath, label, split
"""

import os
import random
import csv
import argparse
from pathlib import Path
from collections import defaultdict

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

CLASS_MAP = {
    "scream": 1,
    "non_scream": 0,
}

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")


def gather_files():
    """Walk raw directory and collect (filepath, label) tuples."""
    records = []
    for class_name, label in CLASS_MAP.items():
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"  [WARN] Directory not found: {class_dir}. Skipping.")
            continue
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            for fp in sorted(class_dir.glob(ext)):
                records.append({"filepath": str(fp), "label": label, "class_name": class_name})
    return records


def recording_level_split(records):
    """
    Split by UNIQUE SOURCE FILE so that windows from the same recording
    never appear in both train and test (prevents data leakage).

    For UrbanSound8K / ESC-50, file names encode the source.
    For custom data, each unique filename is treated as one source.
    """
    random.seed(RANDOM_SEED)

    # Group by class
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)

    all_splits = []
    for class_name, class_records in by_class.items():
        random.shuffle(class_records)
        n = len(class_records)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        for i, rec in enumerate(class_records):
            if i < n_train:
                rec["split"] = "train"
            elif i < n_train + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "test"
            all_splits.append(rec)

    return all_splits


def write_manifest(records):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "class_name", "split"]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n✅ Manifest written to {MANIFEST_PATH}")


def print_stats(records):
    from collections import Counter
    split_class = Counter((r["split"], r["class_name"]) for r in records)
    print("\n📊 Dataset split summary:")
    print(f"{'Split':<10} {'Class':<15} {'Count'}")
    print("-" * 35)
    for (split, cls), count in sorted(split_class.items()):
        print(f"{split:<10} {cls:<15} {count}")


def main():
    parser = argparse.ArgumentParser(description="Build manifest CSV for audio dataset.")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DIR), help="Path to raw audio root dir")
    args = parser.parse_args()

    global RAW_DIR
    RAW_DIR = Path(args.raw_dir)

    print(f"📂 Scanning: {RAW_DIR}")
    records = gather_files()

    if not records:
        print("\n⚠️  No audio files found!")
        print("   Please place .wav files in:")
        print("     data/raw/scream/")
        print("     data/raw/non_scream/")
        print("\n   Suggested datasets:")
        print("     Screams : Search 'human scream audio' on Kaggle / Hugging Face")
        print("     Non-scream: UrbanSound8K (https://urbansounddataset.weebly.com/urbansound8k.html)")
        print("                 ESC-50 (https://github.com/karolpiczak/ESC-50)")
        return

    print(f"  Found {len(records)} audio files across {len(CLASS_MAP)} classes.")
    records = recording_level_split(records)
    print_stats(records)
    write_manifest(records)


if __name__ == "__main__":
    main()
