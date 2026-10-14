# -*- coding: utf-8 -*-
"""
models/cnn_lstm_model.py
-------------------------
CNN-LSTM model: the CNN acts as a per-frame feature extractor;
the LSTM captures temporal patterns across consecutive windows.

Input  : (B, T, 1, 128, 44)  -- batch x time_steps x channel x mel x frames
Output : (B, 2)               -- raw logits [non_scream, scream]

Constructor:
    ScreamCNNLSTM(n_mels=128, num_classes=2, cnn_ckpt_path=None)
    If cnn_ckpt_path is given and exists, CNN weights are loaded.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.cnn_model import ScreamCNN


class ScreamCNNLSTM(nn.Module):
    """
    Architecture:
      For each time step t in the input sequence:
          ScreamCNN.extract_features(frame_t) -> (B, 256)
      Stack T vectors -> (B, T, 256)
      LSTM(input=256, hidden=256, layers=2, dropout=0.3, batch_first=True)
      Take last hidden state -> Linear(256, num_classes)
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        cnn_ckpt_path: str | None = None,
    ):
        super().__init__()

        # CNN feature extractor
        self.cnn = ScreamCNN(n_mels=n_mels, num_classes=num_classes)

        # Optional: load pretrained CNN weights
        if cnn_ckpt_path is not None:
            ckpt_path = Path(cnn_ckpt_path)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state = ckpt.get("model_state", ckpt)
                # Load only CNN-compatible keys (ignore classifier head mismatch)
                missing, unexpected = self.cnn.load_state_dict(state, strict=False)
                print(f"[CNN-LSTM] Loaded CNN weights from {ckpt_path}")
                if missing:
                    print(f"  Missing keys  : {missing[:3]} ...")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected[:3]} ...")
            else:
                print(f"[CNN-LSTM] WARNING: cnn_ckpt_path given but not found: {ckpt_path}")
                print("  Continuing with random CNN weights.")

        self.lstm = nn.LSTM(
            input_size  = self.cnn.feature_dim,   # 256
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = False,
        )

        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1, 128, 44)
        Returns logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Process all T frames through CNN in one batched pass
        x_flat = x.view(B * T, C, H, W)         # (B*T, 1, 128, 44)
        feats  = self.cnn.extract_features(x_flat)  # (B*T, 256)
        feats  = feats.view(B, T, -1)            # (B, T, 256)

        # LSTM over time steps
        lstm_out, _ = self.lstm(feats)           # (B, T, hidden)
        last_out    = lstm_out[:, -1, :]         # (B, hidden) -- last time step

        return self.classifier(last_out)         # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Shape: (B, num_classes)."""
        return F.softmax(self.forward(x), dim=-1)


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ScreamCNNLSTM(n_mels=128, num_classes=2)
    # 4 samples, 5 time steps, 1ch, 128 mel, 44 frames
    dummy = torch.randn(4, 5, 1, 128, 44)
    out   = model(dummy)
    print(f"ScreamCNNLSTM logits shape : {out.shape}")   # (4, 2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters       : {n_params:,}")

    # Test with cnn_ckpt_path that doesn't exist (should warn, not crash)
    model2 = ScreamCNNLSTM(cnn_ckpt_path="models/checkpoints/cnn_best.pth")
    out2   = model2(dummy)
    print(f"ScreamCNNLSTM (no ckpt)    : {out2.shape}")  # still (4, 2)
