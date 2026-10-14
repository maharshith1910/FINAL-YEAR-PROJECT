"""
utils/download_datasets.py
──────────────────────────
Downloads both datasets needed for SafeWatch AI:
  1. Human Screaming Detection Dataset (Kaggle)  → data/raw/scream/  & data/raw/non_scream/
  2. ESC-50 environmental sounds (GitHub)        → data/raw/non_scream/

Usage:
    python utils/download_datasets.py

Requirements:
  - kaggle CLI configured: place kaggle.json in  %USERPROFILE%\.kaggle\kaggle.json
    Get it from: https://www.kaggle.com/settings → API → Create New Token
  - pip install kaggle requests tqdm
"""

import os, sys, shutil, zipfile, tarfile, json
from pathlib import Path
import urllib.request
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

SCREAM_DIR     = ROOT / "data" / "raw" / "scream"
NON_SCREAM_DIR = ROOT / "data" / "raw" / "non_scream"
TMP_DIR        = ROOT / "data" / "_tmp"

for d in [SCREAM_DIR, NON_SCREAM_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, dest: Path, desc: str = ""):
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n❌  kaggle.json NOT FOUND at:", kaggle_json)
        print("    Steps to fix:")
        print("    1. Go to https://www.kaggle.com/settings")
        print("    2. API section → 'Create New Token' → downloads kaggle.json")
        print("    3. Move it to:", kaggle_json)
        print("    4. Run this script again.\n")
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# DATASET 1 — Kaggle Human Screaming Detection Dataset
# ──────────────────────────────────────────────────────────────────────────────

def download_scream_kaggle():
    """
    Dataset: 'bhavikjikadara/human-screaming-detection-dataset'
    862 Screaming + 2631 NotScreaming WAV files (44.1 kHz)
    MIT License
    """
    DATASET_SLUG = "bhavikjikadara/human-screaming-detection-dataset"
    zip_path     = TMP_DIR / "scream_dataset.zip"
    extract_path = TMP_DIR / "scream_kaggle"

    if not check_kaggle_auth():
        return False

    print("\n📥  [1/2] Downloading Human Scream Dataset from Kaggle …")
    print(f"    Dataset: {DATASET_SLUG}")
    print("    This is ~5.7 GB — brew a coffee ☕\n")

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_SLUG,
            path=str(TMP_DIR),
            unzip=False,
            quiet=False,
        )
    except Exception as e:
        print(f"❌  Kaggle download failed: {e}")
        print("    Try manually: kaggle datasets download -d", DATASET_SLUG)
        return False

    # Find the zip
    zips = list(TMP_DIR.glob("*.zip"))
    if not zips:
        print("❌  No zip found in", TMP_DIR)
        return False
    zip_path = zips[0]

    print(f"\n📦  Extracting {zip_path.name} …")
    extract_path.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)

    # Copy files to correct folders
    print("📁  Organising files …")
    moved_scream = moved_non = 0

    # Walk extracted folder for WAV files
    for wav in extract_path.rglob("*.wav"):
        parent = wav.parent.name.lower()
        if "scream" in parent and "not" not in parent:
            dest = SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_scream += 1
        elif "notscreaming" in parent or "not_scream" in parent or "non" in parent:
            dest = NON_SCREAM_DIR / wav.name
            if not dest.exists():
                shutil.copy2(wav, dest)
            moved_non += 1
        else:
            # fallback: try reading folder name
            fname = wav.parent.name
            if fname == "Screaming":
                dest = SCREAM_DIR / wav.name
                if not dest.exists():
                    shutil.copy2(wav, dest)
                moved_scream += 1
            elif fname == "NotScreaming":
                dest = NON_SCREAM_DIR / wav.name
                if not dest.exists():
                    shutil.copy2(wav, dest)
                moved_non += 1

    print(f"  ✅  Scream clips:     {moved_scream}")
    print(f"  ✅  Non-scream clips: {moved_non}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# DATASET 2 — ESC-50 Environmental Sounds
# ──────────────────────────────────────────────────────────────────────────────

def download_esc50():
    """
    ESC-50: 2000 environmental audio clips (5s each), 50 classes.
    License: CC BY (non-commercial).
    We use it for non-scream negative examples.
    """
    ZIP_URL  = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = TMP_DIR / "esc50.zip"

    # Classes to EXCLUDE (too scream-like — would confuse the model)
    EXCLUDE_CLASSES = {
        "crying_baby",  # label 40 — can confuse scream detector
    }

    print("\n📥  [2/2] Downloading ESC-50 Environmental Sounds …")
    print(f"    URL: {ZIP_URL}")
    print("    ~600 MB\n")

    if not zip_path.exists():
        download_url(ZIP_URL, zip_path, desc="ESC-50")
    else:
        print("    (cached zip found, skipping download)")

    extract_path = TMP_DIR / "esc50"
    if not extract_path.exists():
        print("📦  Extracting …")
        extract_path.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

    # Parse meta CSV for label names
    meta_candidates = list(extract_path.rglob("esc50.csv"))
    label_map = {}
    if meta_candidates:
        import csv
        with open(meta_candidates[0]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname  = row.get("filename", "")
                target = int(row.get("target", -1))
                cat    = row.get("category", "unknown").lower()
                label_map[fname] = cat

    # Copy WAV files to non_scream
    moved = 0
    for wav in extract_path.rglob("*.wav"):
        cat = label_map.get(wav.name, "unknown")
        if cat in EXCLUDE_CLASSES:
            continue
        dest = NON_SCREAM_DIR / f"esc50_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)
        moved += 1

    print(f"  ✅  ESC-50 clips added to non_scream: {moved}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def verify():
    scream_files     = list(SCREAM_DIR.glob("*.wav"))
    non_scream_files = list(NON_SCREAM_DIR.glob("*.wav"))

    print("\n" + "="*60)
    print("  📊  DATASET SUMMARY")
    print("="*60)
    print(f"  Scream clips:      {len(scream_files)}")
    print(f"  Non-scream clips:  {len(non_scream_files)}")
    print(f"  Total:             {len(scream_files) + len(non_scream_files)}")
    print(f"  Ratio (neg:pos):   {len(non_scream_files)/max(1,len(scream_files)):.1f}:1")
    print("="*60)

    if len(scream_files) < 100:
        print("\n⚠️  WARNING: Very few scream clips. Check the Kaggle download.")
    if len(non_scream_files) < 100:
        print("\n⚠️  WARNING: Very few non-scream clips. Check the ESC-50 download.")

    if len(scream_files) > 0 and len(non_scream_files) > 0:
        print("\n✅  Ready!  Next step:")
        print("    python utils/data_prep.py")
        print("    python utils/feature_extraction.py")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  SafeWatch AI — Dataset Downloader")
    print("="*60)

    ok1 = download_scream_kaggle()
    ok2 = download_esc50()
    verify()

    if not ok1:
        print("\n⚠️  Kaggle dataset skipped. If you've already downloaded it manually,")
        print("   place files in:")
        print(f"     Screaming WAVs  → {SCREAM_DIR}")
        print(f"     NotScreaming    → {NON_SCREAM_DIR}")
        print("   Then run: python utils/data_prep.py")
