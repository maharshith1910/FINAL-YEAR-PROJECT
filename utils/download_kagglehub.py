# -*- coding: utf-8 -*-
"""
utils/download_kagglehub.py
----------------------------
Downloads the Human Screaming Detection Dataset using kagglehub
(no API key setup needed -- uses browser-based OAuth or cached token).

Dataset: whats2000/human-screaming-detection-dataset
  - 862 Screaming WAVs
  - 2631 NotScreaming WAVs
  - MIT License

Run:
    python utils/download_kagglehub.py
"""

import sys
import shutil
from pathlib import Path

ROOT           = Path(__file__).resolve().parent.parent
SCREAM_DIR     = ROOT / "data" / "raw" / "scream"
NON_SCREAM_DIR = ROOT / "data" / "raw" / "non_scream"

SCREAM_DIR.mkdir(parents=True, exist_ok=True)
NON_SCREAM_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 58)
print("  SafeWatch AI -- kagglehub Downloader")
print("=" * 58)
print("\n  Dataset: whats2000/human-screaming-detection-dataset")
print("  Size: ~4 GB  |  862 scream + 2631 non-scream WAVs")
print("\n  Downloading ... (this may take 5-20 min)\n")

try:
    import kagglehub
except ImportError:
    print("[ERROR] kagglehub not installed.")
    print("  Run: pip install kagglehub")
    sys.exit(1)

# -- Download (kagglehub caches to ~/.cache/kagglehub/...)
try:
    path = kagglehub.dataset_download("whats2000/human-screaming-detection-dataset")
    print(f"\n[OK] Downloaded to: {path}")
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    print("\n  If you see an auth error, run: kaggle auth login")
    print("  Or download manually from:")
    print("  https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset")
    sys.exit(1)

# -- Find and copy WAVs to the correct folders
dataset_path = Path(path)
print(f"\n  Scanning {dataset_path} for WAV files ...")

moved_scream = moved_non = 0

for wav in dataset_path.rglob("*.wav"):
    parent = wav.parent.name

    # Classify by folder name
    if parent == "Screaming":
        dest = SCREAM_DIR / wav.name
        if not dest.exists():
            shutil.copy2(wav, dest)
        moved_scream += 1
    elif parent == "NotScreaming":
        dest = NON_SCREAM_DIR / wav.name
        if not dest.exists():
            shutil.copy2(wav, dest)
        moved_non += 1
    else:
        # Fallback: path-based detection
        path_lower = str(wav).lower().replace("\\", "/")
        if "/screaming/" in path_lower and "/notscreaming/" not in path_lower:
            dest = SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_scream += 1
        else:
            dest = NON_SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_non += 1

# -- Summary
final_scream = list(SCREAM_DIR.glob("*.wav"))
final_non    = list(NON_SCREAM_DIR.glob("*.wav"))

print("\n" + "=" * 58)
print("  DATASET READY")
print("=" * 58)
print(f"  data/raw/scream/     : {len(final_scream)} WAV files")
print(f"  data/raw/non_scream/ : {len(final_non)} WAV files  (incl. ESC-50)")
print(f"  Total                : {len(final_scream) + len(final_non)}")
print("=" * 58)

if len(final_scream) < 100:
    print("\n[WARN] Too few scream files. Check the folder:")
    print(f"  {dataset_path}")
    print("  Look for a 'Screaming' subfolder and copy WAVs manually to:")
    print(f"  {SCREAM_DIR}")
else:
    print("\n[OK] All good!")
    print("\n  Next steps:")
    print("  python utils/data_prep.py")
    print("  python utils/feature_extraction.py")
