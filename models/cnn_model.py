# -*- coding: utf-8 -*-
"""
models/cnn_model.py
--------------------
Baseline 4-block CNN for scream vs non-scream classification.

Input  : (B, 1, 128, 44)   -- batch x channel x mel_bins x frames
Output : (B, 2)             -- raw logits [non_scream, scream]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScreamCNN(nn.Module):
    """
    4-block CNN for log-Mel spectrogram classification.

    Architecture per spec:
      Block 1: Conv2d(1,   32,  3, pad=1) -> BN -> ReLU -> MaxPool(2,2) -> Dropout2d(0.25)
      Block 2: Conv2d(32,  64,  3, pad=1) -> BN -> ReLU -> MaxPool(2,2) -> Dropout2d(0.25)
      Block 3: Conv2d(64,  128, 3, pad=1) -> BN -> ReLU -> MaxPool(2,2) -> Dropout2d(0.25)
      Block 4: Conv2d(128, 256, 3, pad=1) -> BN -> ReLU -> AdaptiveAvgPool(1,1)
      Flatten -> Linear(256,128) -> ReLU -> Dropout(0.5) -> Linear(128, num_classes)

    Returns raw logits.
    """

    feature_dim: int = 256   # used by CNN-LSTM to know embedding size

    def __init__(self, n_mels: int = 128, num_classes: int = 2):
        super().__init__()

        def _block(in_ch, out_ch, pool=True, dropout=0.25):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(p=dropout))
            return nn.Sequential(*layers)

        self.block1 = _block(1,   32,  pool=True,  dropout=0.25)
        self.block2 = _block(32,  64,  pool=True,  dropout=0.25)
        self.block3 = _block(64,  128, pool=True,  dropout=0.25)
        # Block 4: no MaxPool -- use AdaptiveAvgPool instead
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))   # -> (B, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    # ------------------------------------------------------------------
    # Feature extractor (used by CNN-LSTM to get 256-dim embeddings)
    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 44)
        Returns: (B, 256) -- pooled feature vector before classifier
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)          # (B, 256, 1, 1)
        return x.flatten(1)      # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 44)  ->  logits: (B, num_classes)"""
        feat = self.extract_features(x)     # (B, 256)
        # classifier expects (B, 256, 1, 1) via Flatten -- reshape
        feat = feat.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        return self.classifier(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Shape: (B, num_classes)."""
        return F.softmax(self.forward(x), dim=-1)


# ── Sanity check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ScreamCNN(n_mels=128, num_classes=2)
    dummy = torch.randn(8, 1, 128, 44)
    out   = model(dummy)
    feats = model.extract_features(dummy)
    print(f"ScreamCNN  logits shape : {out.shape}")    # (8, 2)
    print(f"ScreamCNN  feature dim  : {feats.shape}")  # (8, 256)
    print(f"feature_dim property    : {model.feature_dim}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters    : {n_params:,}")
