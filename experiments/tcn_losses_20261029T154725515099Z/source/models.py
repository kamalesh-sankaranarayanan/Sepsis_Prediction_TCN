"""
models.py — All model architectures for the Sepsis Prediction project.

Implements:
  1. TCN (Temporal Convolutional Network) with causal + dilated convolutions
     and residual connections — both baseline and missingness-aware.
  2. LSTM (Long Short-Term Memory)
  3. GRU (Gated Recurrent Unit)
  4. LogisticRegressionWrapper — sklearn-compatible thin wrapper
  5. XGBoostWrapper            — xgboost-compatible thin wrapper

All sequence models accept (B, T, C) tensors (batch, time, channels)
and produce (B, T) logits (one per time step).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1-D causal dilated convolution (no future information)."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Remove the future-facing padding on the right
        if self.padding:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """
    TCN residual block:
      CausalConv → LayerNorm → ReLU → Dropout →
      CausalConv → LayerNorm → ReLU → Dropout → + skip
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()

        self.conv1  = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.norm1  = nn.LayerNorm(out_channels)
        self.drop1  = nn.Dropout(dropout)

        self.conv2  = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2  = nn.LayerNorm(out_channels)
        self.drop2  = nn.Dropout(dropout)

        self.relu   = nn.ReLU()

        # 1×1 convolution to match channel dimensions if needed
        self.skip   = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out + residual)
        out = self.drop2(out)
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network for per-timestep binary classification.

    Architecture:
      Embedding layer →
      Stack of ResidualBlocks (exponentially increasing dilation) →
      Linear output head → logit per time step

    Parameters
    ----------
    n_input      : number of input features
    num_channels : list of hidden channel sizes, one per residual block
    kernel_size  : convolution kernel size
    dropout      : dropout probability
    """

    def __init__(
        self,
        n_input:      int,
        num_channels: list[int],
        kernel_size:  int  = 3,
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = n_input
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.network    = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, T, C)
        lengths : (B,) actual sequence lengths (unused in forward, kept for API)

        Returns
        -------
        logits  : (B, T)
        """
        # (B, T, C) → (B, C, T)  for Conv1d
        h = x.transpose(1, 2)
        h = self.network(h)
        # (B, C, T) → (B, T, C) → (B, T)
        h = h.transpose(1, 2)
        logits = self.classifier(h).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Bidirectional-capable LSTM for per-timestep binary classification."""

    def __init__(
        self,
        n_input:     int,
        hidden_size: int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_input, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * factor, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        logits = self.classifier(out).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    """GRU for per-timestep binary classification."""

    def __init__(
        self,
        n_input:     int,
        hidden_size: int  = 128,
        num_layers:  int  = 2,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_input, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)
        logits = self.classifier(out).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    n_input:    int,
    cfg:        dict,
) -> nn.Module:
    """
    Build a sequence model by name.

    model_name : 'tcn' | 'tcn_miss' | 'lstm' | 'gru'
    n_input    : number of input features
    cfg        : hyper-parameter dict (from config.py TCN_CFG or RNN_CFG)
    """
    name = model_name.lower()
    if name in ("tcn", "tcn_miss", "tcn_baseline", "tcn_proposed"):
        return TCN(
            n_input      = n_input,
            num_channels = cfg["num_channels"],
            kernel_size  = cfg["kernel_size"],
            dropout      = cfg["dropout"],
        )
    elif name == "lstm":
        return LSTMModel(
            n_input     = n_input,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        )
    elif name == "gru":
        return GRUModel(
            n_input     = n_input,
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            dropout     = cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Quick parameter count
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, D = 4, 48, 40

    for name, n_in in [("tcn", D), ("tcn_miss", 2 * D), ("lstm", D), ("gru", D)]:
        from config import TCN_CFG, RNN_CFG
        cfg = TCN_CFG if "tcn" in name else RNN_CFG
        m   = build_model(name, n_in, cfg)
        x   = torch.randn(B, T, n_in)
        L   = torch.tensor([T] * B)
        out = m(x, L)
        print(f"  {name:12s}  n_input={n_in:3d}  "
              f"output={tuple(out.shape)}  params={count_params(m):,}")

    print("\nModel self-test passed.")
