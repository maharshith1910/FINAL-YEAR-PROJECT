"""
training/evaluate.py
---------------------
Full evaluation on the held-out test set. Generates:
  - Accuracy, Precision, Recall, F1, FPR
  - Confusion matrix (PNG)
  - Precision-Recall curve / threshold sweep (PNG)
  - Inference latency per window (ms)
  - outputs/metrics/<model>_test_results.json

⚠️  IMPORTANT: This script NEVER fabricates metrics.
    If the checkpoint is missing, it exits with a clear error.

Usage:
    python training/evaluate.py --model cnn
    python training/evaluate.py --model cnn_lstm --n_context 5
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import ScreamDataset, ScreamSequenceDataset
from models.cnn_model import ScreamCNN
from models.cnn_lstm_model import ScreamCNNLSTM

# ─── Paths ─────────────────────────────────────────────────────────────────────
FEAT_MANIFEST  = Path("data/features_manifest.csv")
CHECKPOINT_DIR = Path("models/checkpoints")
METRICS_DIR    = Path("outputs/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

N_MELS        = 128
TARGET_FRAMES = 44
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES   = ["Non-Scream", "Scream"]


def load_checkpoint(model_type: str, n_context: int) -> tuple[torch.nn.Module, dict]:
    ckpt_path = CHECKPOINT_DIR / f"{model_type}_best.pth"
    if not ckpt_path.exists():
        print(f"\n❌ Checkpoint not found: {ckpt_path}")
        print("   Train the model first: python training/train.py --model", model_type)
        sys.exit(1)

    if model_type == "cnn":
        model = ScreamCNN(n_mels=N_MELS, num_classes=2)
    else:
        model = ScreamCNNLSTM(n_mels=N_MELS, num_classes=2)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"✅ Loaded checkpoint: {ckpt_path}  (trained to epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model, ckpt


def build_test_loader(model_type: str, n_context: int, batch_size: int):
    if model_type == "cnn":
        ds = ScreamDataset(FEAT_MANIFEST, split="test", augment=False)
    else:
        ds = ScreamSequenceDataset(FEAT_MANIFEST, split="test", n_context=n_context, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def infer(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run inference. Returns (all_labels, all_preds, all_probs_scream, avg_latency_ms)."""
    all_labels, all_preds, all_probs = [], [], []
    latencies = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            t0     = time.perf_counter()
            logits = model(x)
            t1     = time.perf_counter()
            latencies.append((t1 - t0) * 1000 / x.shape[0])  # ms per sample

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(scream)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y.numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    avg_latency = np.mean(latencies)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), avg_latency


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    plt.tight_layout()
    path = METRICS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Confusion matrix saved → {path}")
    return cm


def plot_pr_curve(y_true, y_probs, model_name: str):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#E74C3C", lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name.upper()}")
    ax.fill_between(recall, precision, alpha=0.1, color="#E74C3C")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = METRICS_DIR / f"{model_name}_pr_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 PR curve saved → {path}")
    return precision, recall, thresholds


def compute_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0
    return fpr


def evaluate(args):
    print(f"\n🔍 Evaluating [{args.model.upper()}] on TEST set")
    if not FEAT_MANIFEST.exists():
        print(f"❌ Features manifest not found. Run feature_extraction.py first.")
        sys.exit(1)

    model, ckpt = load_checkpoint(args.model, args.n_context)
    loader      = build_test_loader(args.model, args.n_context, batch_size=64)

    if len(loader.dataset) == 0:
        print("❌ No test samples found. Check your data and manifest.")
        sys.exit(1)

    print(f"   Test samples: {len(loader.dataset)}")

    y_true, y_pred, y_probs, avg_latency = infer(model, loader)

    # ── Metrics (all real, no fabrication)
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = compute_fpr(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = None

    print(f"\n📈 Test Results for {args.model.upper()}:")
    print(f"   Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision (scream): {precision:.4f}")
    print(f"   Recall    (scream): {recall:.4f}")
    print(f"   F1 Score          : {f1:.4f}")
    print(f"   False Positive Rate: {fpr:.4f}")
    if roc_auc:
        print(f"   ROC-AUC           : {roc_auc:.4f}")
    print(f"   Avg Latency/window: {avg_latency:.2f} ms")

    # ── Plots
    cm = plot_confusion_matrix(y_true, y_pred, args.model)
    plot_pr_curve(y_true, y_probs, args.model)

    # ── Save JSON results
    results = {
        "model":        args.model,
        "checkpoint":   str(CHECKPOINT_DIR / f"{args.model}_best.pth"),
        "test_samples": len(y_true),
        "accuracy":     round(float(acc), 4),
        "precision":    round(float(precision), 4),
        "recall":       round(float(recall), 4),
        "f1":           round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "roc_auc":      round(float(roc_auc), 4) if roc_auc else None,
        "avg_latency_ms": round(float(avg_latency), 2),
        "confusion_matrix": cm.tolist(),
        "note": "All metrics computed on held-out test set. No data fabrication.",
    }

    results_path = METRICS_DIR / f"{args.model}_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  💾 Full results saved → {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scream detection model on test set.")
    parser.add_argument("--model",     type=str, default="cnn_lstm", choices=["cnn", "cnn_lstm"])
    parser.add_argument("--n_context", type=int, default=5)
    args = parser.parse_args()
    evaluate(args)
