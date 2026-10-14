"""
inference/alert_logger.py
--------------------------
Logs threat alerts to a CSV file and provides query helpers for the
Streamlit dashboard.

Each alert row contains:
  alert_id, timestamp_iso, timestamp_sec, location, confidence,
  model_used, severity, verification_status, note
"""

import csv
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

ALERT_LOG_PATH = Path("outputs/alerts/alert_log.csv")
FIELDNAMES = [
    "alert_id",
    "timestamp_iso",
    "timestamp_sec",
    "location",
    "confidence",
    "severity",
    "model_used",
    "verification_status",
    "note",
]


def _severity_from_confidence(conf: float) -> str:
    if conf >= 0.90:
        return "HIGH"
    elif conf >= 0.75:
        return "MEDIUM"
    else:
        return "LOW"


class AlertLogger:
    """
    Thread-safe alert logger.
    Creates the CSV file (with header) on first write if it doesn't exist.
    """

    def __init__(self, log_path: Path = ALERT_LOG_PATH):
        self.log_path = log_path
        self._lock    = Lock()
        self._encoding = "utf-8-sig"   # BOM so Windows Excel reads cleanly
        self._ensure_file()

    def _ensure_file(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding=self._encoding) as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log(
        self,
        location:   str   = "Unknown",
        confidence: float = 0.0,
        model_used: str   = "cnn_lstm",
        timestamp:  float = None,       # seconds since session start (float)
    ) -> dict:
        """Write one alert row. Returns the written dict."""
        if timestamp is None:
            timestamp = time.time()

        alert = {
            "alert_id":           str(uuid.uuid4())[:8].upper(),
            "timestamp_iso":      datetime.now(timezone.utc).isoformat(),
            "timestamp_sec":      round(timestamp, 2),
            "location":           location,
            "confidence":         round(confidence, 4),
            "severity":           _severity_from_confidence(confidence),
            "model_used":         model_used,
            "verification_status": "PENDING",
            "note":               "Potential distress event - requires human verification.",
        }

        with self._lock:
            with open(self.log_path, "a", newline="", encoding=self._encoding) as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(alert)

        return alert

    # ── Read helpers ──────────────────────────────────────────────────────────

    def load_all(self) -> pd.DataFrame:
        """Load the full alert log as a DataFrame."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return pd.DataFrame(columns=FIELDNAMES)
        return pd.read_csv(self.log_path, encoding=self._encoding)

    def update_verification(self, alert_id: str, new_status: str):
        """Update the verification status of an alert (in place)."""
        df = self.load_all()
        if df.empty:
            return
        mask = df["alert_id"] == alert_id
        if mask.any():
            df.loc[mask, "verification_status"] = new_status
            df.to_csv(self.log_path, index=False)

    def stats(self) -> dict:
        """Return high-level stats for the dashboard."""
        df = self.load_all()
        if df.empty:
            return {"total": 0, "high": 0, "medium": 0, "low": 0, "pending": 0}
        return {
            "total":   len(df),
            "high":    int((df["severity"] == "HIGH").sum()),
            "medium":  int((df["severity"] == "MEDIUM").sum()),
            "low":     int((df["severity"] == "LOW").sum()),
            "pending": int((df["verification_status"] == "PENDING").sum()),
        }


# ─── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = AlertLogger()
    a = logger.log(location="Block A", confidence=0.91, model_used="cnn_lstm")
    b = logger.log(location="Block B", confidence=0.76, model_used="cnn_lstm")
    c = logger.log(location="Block A", confidence=0.66, model_used="cnn")
    print("Logged alerts:")
    print(logger.load_all().to_string())
    print("\nStats:", logger.stats())
