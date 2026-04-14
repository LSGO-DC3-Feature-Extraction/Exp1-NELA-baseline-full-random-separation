from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch

from .original_transformer_model import (
    EmbeddingNet,
    FeatureExtractor as _BaseFeatureExtractor,
    MultiHeadEncoder,
    PositionalEncoding,
)


def _to_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


class FeatureExtractor(_BaseFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._force_train = False

    def set_on_train(self):
        self._force_train = True
        return self

    def set_off_train(self):
        self._force_train = False
        return self

    def get_output_dim(self) -> int:
        return self.embed.linear.out_features

    def _run_forward(self, xs: Any, ys: Any) -> torch.Tensor:
        ref_param = next(self.parameters())
        xs_tensor = _to_tensor(xs, dtype=ref_param.dtype, device=ref_param.device)
        ys_tensor = _to_tensor(ys, dtype=ref_param.dtype, device=ref_param.device)
        if xs_tensor.ndim != 2:
            raise ValueError(f"xs must be 2D (population, dim), got shape {tuple(xs_tensor.shape)}.")
        if ys_tensor.ndim != 1:
            ys_tensor = ys_tensor.reshape(-1)
        if xs_tensor.shape[0] != ys_tensor.shape[0]:
            raise ValueError(
                f"xs and ys must share the population axis, got {xs_tensor.shape[0]} and {ys_tensor.shape[0]}."
            )

        xs_tensor = xs_tensor.unsqueeze(0)
        ys_tensor = ys_tensor.unsqueeze(0)
        ys_norm = (ys_tensor - ys_tensor.min(dim=-1, keepdim=True).values) / (
            ys_tensor.max(dim=-1, keepdim=True).values - ys_tensor.min(dim=-1, keepdim=True).values + 1e-12
        )
        ys_tensor = ys_norm.unsqueeze(-1)

        feature = torch.cat(
            [
                xs_tensor.unsqueeze(-1),
                ys_tensor.repeat(1, 1, xs_tensor.shape[-1]).unsqueeze(-1),
            ],
            dim=-1,
        )
        feature = feature.permute(0, 2, 1, 3)
        hidden = self.embed(feature.float())

        if self.is_mlp:
            return self.mlp(hidden).mean(dim=-3)

        hidden = self.dim_encoder(hidden.reshape(-1, hidden.shape[2], hidden.shape[3])).view(*hidden.shape)
        individual_hidden = hidden.permute(0, 2, 1, 3).reshape(-1, hidden.shape[1], hidden.shape[3])
        if self.use_pe:
            individual_hidden = individual_hidden + self.pe(hidden.shape[1]) * 0.5
        return self.ind_encoder(individual_hidden).view(
            xs_tensor.shape[0],
            xs_tensor.shape[1],
            xs_tensor.shape[2],
            -1,
        ).mean(dim=-2)

    def forward(self, xs: Any, ys: Any) -> torch.Tensor:
        context = contextlib.nullcontext() if self._force_train else torch.no_grad()
        with context:
            return self._run_forward(xs, ys)


Feature_Extractor = FeatureExtractor

__all__ = [
    "EmbeddingNet",
    "FeatureExtractor",
    "Feature_Extractor",
    "MultiHeadEncoder",
    "PositionalEncoding",
]
