"""
training/train.py
-----------------
Main training script for ScreamCNN and ScreamCNNLSTM.

Usage:
    python training/train.py --model cnn      --epochs 30 --lr 1e-3
    python training/train.py --model cnn_lstm --epochs 30 --lr 1e-3 --n_context 5

Outputs:
    models/checkpoints/<model_name>_best.pth   ← best validation checkpoint
    outputs/metrics/<model_name>_train_log.csv ← per-epoch loss & accuracy
    outputs/metrics/<model_name>_curves.png    ← loss/accuracy plots
"""

import os
import sys
import json
import argparse
import time
import csv
from pathlib import Path

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dataset import ScreamDataset, ScreamSequenceDataset
from models.cnn_model import ScreamCNN
from models.cnn_lstm_model import ScreamCNNLSTM

# ─── Constants ─────────────────────────────────────────────────────────────────
FEAT_MANIFEST  = Path("data/features_manifest.csv")
CHECKPOINT_DIR = Path("models/checkpoints")
METRICS_DIR    = Path("outputs/metrics")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

N_MELS        = 128
TARGET_FRAMES = 44
NUM_CLASSES   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_dataloaders(model_type: str, n_context: int, batch_size: int, num_workers: int = 0):
    if model_type == "cnn":
        train_ds = ScreamDataset(FEAT_MANIFEST, split="train", augment=True)
        val_ds   = ScreamDataset(FEAT_MANIFEST, split="val",   augment=False)
    else:
        train_ds = ScreamSequenceDataset(FEAT_MANIFEST, split="train", n_context=n_context, augment=True)
        val_ds   = ScreamSequenceDataset(FEAT_MANIFEST, split="val",   n_context=n_context, augment=False)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


def build_model(model_type: str) -> nn.Module:
    if model_type == "cnn":
        return ScreamCNN(n_mels=N_MELS, num_classes=NUM_CLASSES)
    elif model_type == "cnn_lstm":
        return ScreamCNNLSTM(n_mels=N_MELS, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer=None, phase="train", epoch=1, total_epochs=1, max_batches=None):
    """Run one epoch. If optimizer is None, runs in eval mode."""
    is_train = (optimizer is not None)
    model.train() if is_train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    total_steps = min(len(loader), max_batches) if max_batches else len(loader)
    pbar = tqdm(loader, total=total_steps, desc=f"  [{phase.capitalize():<5}] Epoch {epoch:02d}/{total_epochs}", leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_acc = compute_accuracy(logits, y)
            total_loss += loss.item()
            total_acc  += batch_acc
            n_batches  += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})
            if max_batches and n_batches >= max_batches:
                break

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def plot_curves(log: list[dict], model_name: str):
    epochs  = [r["epoch"]    for r in log]
    tr_loss = [r["train_loss"] for r in log]
    vl_loss = [r["val_loss"]   for r in log]
    tr_acc  = [r["train_acc"]  for r in log]
    vl_acc  = [r["val_acc"]    for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tr_loss, label="Train Loss", color="#4A90D9")
    ax1.plot(epochs, vl_loss, label="Val Loss",   color="#E74C3C")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, tr_acc, label="Train Acc", color="#4A90D9")
    ax2.plot(epochs, vl_acc, label="Val Acc",   color="#E74C3C")
    ax2.set_title("Accuracy Curves"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    path = METRICS_DIR / f"{model_name}_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Curves saved → {path}")


# ─── Main training loop ────────────────────────────────────────────────────────

def train(args):
    print(f"\n🚀 Training [{args.model.upper()}] on {DEVICE}")
    print(f"   Epochs={args.epochs} | LR={args.lr} | Batch={args.batch_size}")
    if args.max_batches:
        print(f"   Verification mode: max_batches={args.max_batches} per epoch")

    if not FEAT_MANIFEST.exists():
        print(f"❌ Features manifest not found at {FEAT_MANIFEST}.")
        print("   Run: python utils/feature_extraction.py")
        sys.exit(1)

    # ── Class weights for imbalanced loss: total / (n_classes * count)
    df = pd.read_csv(FEAT_MANIFEST)
    train_df = df[df["split"] == "train"]
    total_samples = len(train_df)
    count_0 = max(1, int((train_df["label"] == 0).sum()))
    count_1 = max(1, int((train_df["label"] == 1).sum()))
    w0 = total_samples / (2.0 * count_0)
    w1 = total_samples / (2.0 * count_1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
    print(f"   Train samples: {total_samples:,} (Non-Scream={count_0:,}, Scream={count_1:,})")
    print(f"   Class weights: Non-Scream={w0:.3f}, Scream={w1:.3f}")

    # ── Data
    train_loader, val_loader = build_dataloaders(args.model, args.n_context, args.batch_size, num_workers=args.num_workers)
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model
    model     = build_model(args.model).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop
    best_val_acc = 0.0
    model_name   = args.model
    log          = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        val_max = max(1, args.max_batches // 3) if args.max_batches else None
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, phase="train", epoch=epoch, total_epochs=args.epochs, max_batches=args.max_batches)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      phase="val",   epoch=epoch, total_epochs=args.epochs, max_batches=val_max)
        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_acc":    round(val_acc, 4),
        }
        log.append(row)

        print(f"  Epoch {epoch:03d}/{args.epochs} | "
              f"TrLoss={train_loss:.4f} TrAcc={train_acc:.4f} | "
              f"VlLoss={val_loss:.4f} VlAcc={val_acc:.4f} | {elapsed:.1f}s")

        # ── Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✅ Best checkpoint saved → {ckpt_path} (val_acc={val_acc:.4f})")

    # ── Save training log CSV
    log_path = METRICS_DIR / f"{model_name}_train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"\n  📋 Training log saved → {log_path}")

    # ── Plot curves
    plot_curves(log, model_name)

    print(f"\n🏁 Training complete. Best val acc: {best_val_acc:.4f}")
    print("   Run evaluation: python training/evaluate.py --model", args.model)


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scream detection model.")
    parser.add_argument("--model",       type=str,   default="cnn",    choices=["cnn", "cnn_lstm"],
                        help="Model architecture to train")
    parser.add_argument("--epochs",      type=int,   default=30,
                        help="Number of training epochs")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Batch size")
    parser.add_argument("--n_context",   type=int,   default=5,
                        help="Number of context windows for CNN-LSTM")
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="DataLoader worker count (default: 0 for CPU/Windows safety)")
    parser.add_argument("--max_batches", type=int,   default=None,
                        help="Max batches per epoch for fast verification")
    args = parser.parse_args()
    train(args)
