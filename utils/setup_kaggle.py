"""
utils/setup_kaggle.py
─────────────────────
Interactive helper to configure the Kaggle API token.

Run FIRST before download_datasets.py:
    python utils/setup_kaggle.py
"""
import os, json, sys
from pathlib import Path

KAGGLE_DIR  = Path.home() / ".kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"

print("="*58)
print("  SafeWatch AI — Kaggle API Setup")
print("="*58)

if KAGGLE_JSON.exists():
    print(f"\n✅  kaggle.json already exists at:\n   {KAGGLE_JSON}")
    print("    You're ready to download datasets.")
    print("\n    Next: python utils/download_datasets.py")
    sys.exit(0)

print("""
To download the Kaggle dataset, you need a free Kaggle account and API token.

Steps:
  1. Go to  https://www.kaggle.com/settings
  2. Scroll to "API" section → click "Create New Token"
  3. This downloads  kaggle.json  to your Downloads folder
  4. Paste the contents below (it looks like):
     {"username":"yourname","key":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
""")

raw = input("Paste the full contents of kaggle.json here and press Enter:\n> ").strip()

try:
    data = json.loads(raw)
    assert "username" in data and "key" in data
except Exception:
    print("\n❌  That doesn't look like valid kaggle.json content.")
    print("    Expected:  {\"username\":\"...\",\"key\":\"...\"}")
    sys.exit(1)

KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
with open(KAGGLE_JSON, "w") as f:
    json.dump(data, f)
os.chmod(KAGGLE_JSON, 0o600)

print(f"\n✅  kaggle.json written to:\n   {KAGGLE_JSON}")
print("\n    Next step:")
print("    python utils/download_datasets.py")
