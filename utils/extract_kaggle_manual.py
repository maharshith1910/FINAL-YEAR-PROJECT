# -*- coding: utf-8 -*-
"""
utils/extract_kaggle_manual.py
-------------------------------
Use this if the Kaggle API gives a 403 error (you need to accept dataset
terms on the website first, OR just download the zip manually).

HOW TO USE:
  1. Go to:
     https://www.kaggle.com/datasets/bhavikjikadara/human-screaming-detection-dataset
  2. Click the Download button (top-right) — saves the zip to your Downloads
  3. Run:  python utils/extract_kaggle_manual.py

The script auto-finds the zip in common locations and extracts it.
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Where to look for the downloaded zip
SEARCH_DIRS = [
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    ROOT / "data" / "_tmp",
]
ZIP_NAMES = [
    "human-screaming-detection-dataset.zip",
    "archive.zip",
]

SCREAM_DIR     = ROOT / "data" / "raw" / "scream"
NON_SCREAM_DIR = ROOT / "data" / "raw" / "non_scream"
TMP_DIR        = ROOT / "data" / "_tmp" / "kaggle_extract"

def find_zip() -> Path | None:
    for d in SEARCH_DIRS:
        for name in ZIP_NAMES:
            candidate = d / name
            if candidate.exists():
                return candidate
    # Also try any zip in Downloads containing "scream" in the name
    dl = Path.home() / "Downloads"
    if dl.exists():
        for f in dl.glob("*.zip"):
            if "scream" in f.name.lower() or "human" in f.name.lower():
                return f
    return None


def main():
    print("=" * 58)
    print("  SafeWatch AI -- Manual Kaggle Extractor")
    print("=" * 58)

    zip_path = find_zip()

    if zip_path is None:
        print("\n[ERROR] Could not find the downloaded zip file.")
        print("\n  Please download it manually:")
        print("  1. Open: https://www.kaggle.com/datasets/bhavikjikadara/human-screaming-detection-dataset")
        print("  2. Click the Download button (top-right of the page)")
        print("  3. Re-run this script once the download is complete")
        print(f"\n  Then place the zip in one of these locations:")
        for d in SEARCH_DIRS:
            print(f"    {d}")
        sys.exit(1)

    print(f"\n  Found zip: {zip_path}")
    print(f"  Size:      {zip_path.stat().st_size / 1e9:.2f} GB")

    # Extract to temp
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    SCREAM_DIR.mkdir(parents=True, exist_ok=True)
    NON_SCREAM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Extracting to {TMP_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        print(f"  Archive contains {len(members)} files")
        z.extractall(TMP_DIR)

    # Walk extracted and sort files
    print("\n  Organising files ...")
    moved_scream = moved_non = 0

    for wav in TMP_DIR.rglob("*.wav"):
        parent_name = wav.parent.name

        # Detect folder name
        if parent_name == "Screaming" or "scream" in parent_name.lower() and "not" not in parent_name.lower():
            dest = SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_scream += 1
        elif parent_name in ("NotScreaming", "Not_Screaming") or "notscream" in parent_name.lower().replace("_","").replace("-",""):
            dest = NON_SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_non += 1
        else:
            # Fallback: check full path
            full = str(wav).lower()
            if "screaming" in full and "not" not in full.split("screaming")[0][-4:]:
                dest = SCREAM_DIR / wav.name
            else:
                dest = NON_SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            if "scream" in str(dest):
                moved_scream += 1
            else:
                moved_non += 1

    print(f"\n  Scream clips moved    : {moved_scream}")
    print(f"  Non-scream clips moved: {moved_non}")

    # Verify
    final_scream = list(SCREAM_DIR.glob("*.wav"))
    final_non    = list(NON_SCREAM_DIR.glob("*.wav"))

    print("\n" + "=" * 58)
    print("  FINAL COUNT")
    print("=" * 58)
    print(f"  data/raw/scream/     : {len(final_scream)} WAV files")
    print(f"  data/raw/non_scream/ : {len(final_non)} WAV files")
    print("=" * 58)

    if len(final_scream) < 100:
        print("\n[WARN] Very few scream files found.")
        print("  The zip folder structure may be different.")
        print(f"  Check inside: {TMP_DIR}")
        print("  Manually copy Screaming/*.wav  ->  data/raw/scream/")
        print("  Manually copy NotScreaming/*.wav -> data/raw/non_scream/")
    else:
        print("\n[OK] Dataset ready!")
        print("\n  Next steps:")
        print("  python utils/data_prep.py")
        print("  python utils/feature_extraction.py")


if __name__ == "__main__":
    main()
