"""
Shared attention modules for GRU-based models.

Used by:
  - crypto_intraday_model.py (crypto intraday LGB+GRU ensemble)
  - etf_intraday_model.py    (ETF intraday LGB+GRU ensemble)
"""

import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    """Luong (general) attention over GRU hidden states.

    Score:   score(h_last, h_i) = h_last^T * W * h_i
    Weights: softmax(scores)
    Context: weighted sum of all hidden states

    Parameters: hidden_size^2 (e.g. 32x32 = 1,024 extra params).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, gru_outputs: torch.Tensor, last_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_outputs: (batch, seq_len, hidden) — all GRU timestep outputs
            last_hidden: (batch, hidden) — final GRU hidden state

        Returns:
            context: (batch, hidden) — attention-weighted combination
            weights: (batch, seq_len) — attention weights per timestep
        """
        # score_i = last_hidden^T @ W @ h_i  for each timestep i
        # W(gru_outputs): (batch, seq_len, hidden)
        # bmm with last_hidden: (batch, seq_len, hidden) @ (batch, hidden, 1) -> (batch, seq_len, 1)
        scores = torch.bmm(
            self.W(gru_outputs), last_hidden.unsqueeze(2)
        ).squeeze(2)  # (batch, seq_len)

        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # context = sum(weight_i * h_i)
        context = (gru_outputs * weights.unsqueeze(2)).sum(dim=1)  # (batch, hidden)

        return context, weights


class GRUWithAttention(nn.Module):
    """GRU + Luong attention for return regression.

    Drop-in replacement for the old GRUReturnModel that used only the
    final hidden state. This version attends over ALL timesteps, letting
    the model focus on the most informative bars in the lookback window.

    Architecture:
        Input:     (batch, seq_len, n_features)
        GRU:       n_layers, hidden_size, dropout
        Attention: Luong general over all hidden states
        FC head:   hidden -> hidden//2 -> ReLU -> Dropout -> 1

    Output: (batch,) — predicted return (continuous).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = LuongAttention(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            predictions: (batch,)
        """
        out, _ = self.gru(x)           # (batch, seq_len, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        context, _ = self.attention(out, last)  # (batch, hidden)
        return self.head(context).squeeze(-1)   # (batch,)
