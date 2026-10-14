"""
models/cnn_lstm_model.py
-------------------------
CNN-LSTM model: the CNN from cnn_model.py acts as a per-frame feature
extractor; the LSTM captures temporal patterns across consecutive frames.

Input  : (batch, seq_len, 1, n_mels, time_frames_per_window)
           — seq_len windows per sample (e.g. 5 consecutive 1-s windows)
Output : (batch, 2)  — logits

For the dashboard / real-time inference, seq_len = N_CONTEXT windows are
buffered before each prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_model import ScreamCNN


class ScreamCNNLSTM(nn.Module):
    """
    Architecture:
      For each window in the sequence:
          CNN feature extractor → 256-dim vector
      Stack into (batch, seq_len, 256) → LSTM(2 layers, 256 hidden)
      Take last hidden state → FC → num_classes
    """

    def __init__(
        self,
        n_mels: int     = 128,
        num_classes: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()

        # Reuse CNN as feature extractor (without the final classifier)
        cnn_backbone = ScreamCNN(n_mels=n_mels, num_classes=num_classes)
        self.cnn_features = cnn_backbone.features
        self.cnn_gap       = cnn_backbone.gap
        self.cnn_proj      = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, lstm_hidden),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size    = lstm_hidden,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            dropout        = lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional = False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 1, n_mels, time_frames)
        """
        batch, seq_len, C, H, W = x.shape

        # Flatten batch & seq to process all windows through CNN at once
        x_flat = x.view(batch * seq_len, C, H, W)   # (B*S, 1, H, W)
        feat   = self.cnn_features(x_flat)           # (B*S, 256, h', w')
        feat   = self.cnn_gap(feat)                  # (B*S, 256, 1, 1)
        feat   = self.cnn_proj(feat)                 # (B*S, lstm_hidden)

        # Reshape back to sequence
        feat = feat.view(batch, seq_len, -1)         # (B, S, lstm_hidden)

        # LSTM: take final timestep output
        lstm_out, _ = self.lstm(feat)                # (B, S, lstm_hidden)
        last_out    = lstm_out[:, -1, :]             # (B, lstm_hidden)

        return self.classifier(last_out)             # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


# ─── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ScreamCNNLSTM(n_mels=128, num_classes=2)
    # 8 samples, 5 windows, 1ch, 128 mel bands, 44 frames (~1s)
    dummy = torch.randn(8, 5, 1, 128, 44)
    out   = model(dummy)
    print(f"ScreamCNNLSTM output shape: {out.shape}")  # expect (8, 2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")
