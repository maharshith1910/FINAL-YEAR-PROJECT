"""
models/cnn_model.py
--------------------
Baseline 4-block CNN for scream vs non-scream classification.

Input  : (batch, 1, n_mels, time_frames)  — single-channel spectrogram
Output : (batch, 2)                        — logits [non_scream, scream]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool with optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScreamCNN(nn.Module):
    """
    4-block CNN for log-Mel spectrogram classification.
    Architecture:
        [Conv Block 32] → [Conv Block 64] → [Conv Block 128] → [Conv Block 256]
        → Global Average Pooling → FC(256) → ReLU → Dropout → FC(num_classes)
    """

    def __init__(self, n_mels: int = 128, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  dropout=0.1),
            ConvBlock(32,  64,  dropout=0.2),
            ConvBlock(64,  128, dropout=0.25),
            ConvBlock(128, 256, dropout=0.3),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)          # → (batch, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)


# ─── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ScreamCNN(n_mels=128, num_classes=2)
    dummy = torch.randn(8, 1, 128, 44)   # 8 samples, 1ch, 128 mel, ~1s window
    out   = model(dummy)
    print(f"ScreamCNN output shape: {out.shape}")   # expect (8, 2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")
