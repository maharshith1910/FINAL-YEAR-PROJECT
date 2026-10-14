# -*- coding: utf-8 -*-
"""
training/evaluate.py
---------------------
Full evaluation on the held-out TEST set. Generates:
  - Accuracy, Precision, Recall, F1, FPR, ROC-AUC
  - Confusion matrix PNG
  - Precision-Recall curve PNG
  - Inference latency per window (ms)
  - outputs/metrics/<model>_test_results.json

IMPORTANT: This script NEVER fabricates metrics.
  If the checkpoint is missing it prints an error and exits cleanly.
  All reported numbers come from actual model outputs on real data.

Usage:
    python training/evaluate.py --model cnn
    python training/evaluate.py --model cnn_lstm --n_context 5
    python training/evaluate.py --model cnn --manifest data/features_manifest.csv
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import ScreamDataset, ScreamSequenceDataset
from models.cnn_model import ScreamCNN
from models.cnn_lstm_model import ScreamCNNLSTM

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
FEAT_MANIFEST = ROOT / "data" / "features_manifest.csv"
CKPT_DIR      = ROOT / "models" / "checkpoints"
METRICS_DIR   = ROOT / "outputs" / "metrics"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES   = ["Non-Scream", "Scream"]
N_MELS        = 128


# ── Checkpoint loader ──────────────────────────────────────────────────────────

def load_checkpoint(model_type: str, ckpt_dir: Path) -> torch.nn.Module:
    ckpt_path = ckpt_dir / f"{model_type}_best.pth"
    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found at {ckpt_path}.")
        print(f"  Run training first:  python training/train.py --model {model_type}")
        sys.exit(1)

    model = ScreamCNN(n_mels=N_MELS) if model_type == "cnn" else ScreamCNNLSTM(n_mels=N_MELS)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"[OK] Loaded: {ckpt_path}  (epoch={epoch}, val_acc={val_acc:.4f})")
    return model


# ── Class weights (from train split, same as training) ─────────────────────────

def compute_class_weights(manifest_path: Path) -> torch.Tensor:
    """
    weight[c] = total / (n_classes * count[c])
    Same formula used in training/train.py.
    """
    df      = pd.read_csv(manifest_path)
    train   = df[df["split"] == "train"]
    counts  = train["label"].value_counts().sort_index()   # index: 0, 1
    total   = counts.sum()
    n_cls   = len(counts)
    weights = total / (n_cls * counts.values)
    return torch.tensor(weights, dtype=torch.float32)


# ── DataLoader ─────────────────────────────────────────────────────────────────

def build_test_loader(
    model_type: str, manifest: Path, feat_dir: Path,
    n_context: int, batch_size: int,
) -> DataLoader:
    if model_type == "cnn":
        ds = ScreamDataset(manifest, split="test", augment=False,
                           feat_dir=str(feat_dir))
    else:
        ds = ScreamSequenceDataset(manifest, split="test",
                                   n_context=n_context, augment=False,
                                   feat_dir=str(feat_dir))
    if len(ds) == 0:
        print("[ERROR] No test samples found in manifest. Check feature extraction.")
        sys.exit(1)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module, loader: DataLoader, max_samples: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns (y_true, y_pred, y_probs_scream, avg_latency_ms_per_sample).
    ALL values come from real model outputs -- no fabrication.
    """
    all_labels, all_preds, all_probs = [], [], []
    latencies = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            t0     = time.perf_counter()
            logits = model(x)
            t1     = time.perf_counter()
            latencies.append((t1 - t0) * 1000 / x.shape[0])  # ms / sample

            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            labels = y.numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            if max_samples and len(all_labels) >= max_samples:
                break

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs), float(np.mean(latencies)))


# ── FPR helper ─────────────────────────────────────────────────────────────────

def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return 0.0


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix -- {model_name.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix -> {path}")
    return cm


def plot_pr_curve(y_true, y_probs, model_name: str, out_dir: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#E74C3C", lw=2, label="CNN-LSTM" if "lstm" in model_name else "CNN")
    ax.fill_between(rec, prec, alpha=0.08, color="#E74C3C")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve -- {model_name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{model_name}_pr_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  PR curve        -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    manifest  = Path(args.manifest)
    feat_dir  = Path(args.feat_dir)
    ckpt_dir  = Path(args.ckpt_dir)
    out_dir   = METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard: manifest must exist
    if not manifest.exists():
        print(f"[ERROR] Manifest not found: {manifest}")
        print("  Run: python utils/feature_extraction.py")
        sys.exit(1)

    print(f"\n=== Evaluating [{args.model.upper()}] on TEST set ===")
    print(f"  Device   : {DEVICE}")
    print(f"  Manifest : {manifest}")

    model  = load_checkpoint(args.model, ckpt_dir)
    loader = build_test_loader(args.model, manifest, feat_dir,
                               args.n_context, args.batch_size)
    print(f"  Test samples: {len(loader.dataset)}")
    if args.max_samples:
        print(f"  Evaluating subset: max {args.max_samples} samples")

    y_true, y_pred, y_probs, avg_lat = run_inference(model, loader, max_samples=args.max_samples)

    # ── All metrics from real data (no fabrication)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    fpr  = false_positive_rate(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = None

    # ── Print table
    sep = "-" * 42
    print(f"\n{sep}")
    print(f"  TEST RESULTS  --  {args.model.upper()}")
    print(sep)
    print(f"  Accuracy            : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision (scream)  : {prec:.4f}")
    print(f"  Recall    (scream)  : {rec:.4f}")
    print(f"  F1 Score            : {f1:.4f}")
    print(f"  False Positive Rate : {fpr:.4f}")
    if auc is not None:
        print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Avg Latency/window  : {avg_lat:.2f} ms")
    print(sep)

    # ── Plots
    cm = plot_confusion_matrix(y_true, y_pred, args.model, out_dir)
    plot_pr_curve(y_true, y_probs, args.model, out_dir)

    # ── Save JSON
    results = {
        "model":               args.model,
        "checkpoint":          str(ckpt_dir / f"{args.model}_best.pth"),
        "test_samples":        int(len(y_true)),
        "accuracy":            round(float(acc),  4),
        "precision":           round(float(prec), 4),
        "recall":              round(float(rec),  4),
        "f1":                  round(float(f1),   4),
        "false_positive_rate": round(float(fpr),  4),
        "roc_auc":             round(float(auc),  4) if auc is not None else None,
        "avg_latency_ms":      round(float(avg_lat), 2),
        "confusion_matrix":    cm.tolist(),
        "note": "Computed on held-out test set. No data fabrication.",
    }
    json_path = out_dir / f"{args.model}_test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON    -> {json_path}")
    print(f"\n[OK] Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate scream detection model on held-out test set.")
    parser.add_argument("--model",      type=str, default="cnn",
                        choices=["cnn", "cnn_lstm"],
                        help="Which model to evaluate")
    parser.add_argument("--manifest",   type=str,
                        default=str(FEAT_MANIFEST),
                        help="Path to features_manifest.csv")
    parser.add_argument("--feat_dir",   type=str,
                        default=str(ROOT / "data" / "features"),
                        help="Directory containing .npy feature files")
    parser.add_argument("--ckpt_dir",   type=str,
                        default=str(CKPT_DIR),
                        help="Directory containing best .pth checkpoints")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--n_context",   type=int, default=5,
                        help="Sequence length for cnn_lstm (default: 5)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples to evaluate (default: None, evaluate all)")
    evaluate(parser.parse_args())
