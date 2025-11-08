"""Utility head to predict per-pixel keep logits and optional log-variance."""

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn


class KeepUncertaintyHead(nn.Module):
    """Predict per-pixel keep logits (and optional log-variance) from decoder features."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Union[Sequence[int], int] = (128, 64),
        activation: str = "relu",
        predict_log_variance: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.predict_log_variance = predict_log_variance

        act_layer = self._resolve_activation(activation)
        layers: List[nn.Module] = []
        prev_channels = in_channels
        for hidden in hidden_dims:
            if hidden <= 0:
                continue
            layers.append(nn.Conv2d(prev_channels, hidden, kernel_size=1, bias=True))
            layers.append(act_layer())
            prev_channels = hidden
        layers.append(nn.Conv2d(prev_channels, 1, kernel_size=1, bias=True))
        self.keep_head = nn.Sequential(*layers)

        if predict_log_variance:
            self.uncertainty_head = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        else:
            self.uncertainty_head = None

    @staticmethod
    def _resolve_activation(name: str) -> type[nn.Module]:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name in ("silu", "swish"):
            return nn.SiLU
        raise ValueError(f"Unsupported activation '{name}' for KeepUncertaintyHead")

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        keep_logits = self.keep_head(features)
        log_variance = self.uncertainty_head(features) if self.uncertainty_head is not None else None
        return keep_logits, log_variance

