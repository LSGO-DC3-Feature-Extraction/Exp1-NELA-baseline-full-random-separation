from __future__ import annotations

from typing import Any

import numpy as np
import torch

ELA_FEATURE_DIM = 21


def get_ela_feature(problem, X, y, random_state=None, ela_conv_nsample=200):
    from ela.ela_feature import get_ela_feature as _get_ela_feature

    return _get_ela_feature(
        problem,
        X,
        y,
        random_state=random_state,
        ela_conv_nsample=ela_conv_nsample,
    )


def get_ela_feature_dim():
    return ELA_FEATURE_DIM


def resolve_state_dim(config, classic_dim: int, use_feature_extractor: bool) -> int:
    if use_feature_extractor:
        return int(config.hidden_dim)
    if getattr(config, "use_ela", False):
        return get_ela_feature_dim()
    return classic_dim


def compute_features(feature_extractor: Any, X: Any, y: Any) -> torch.Tensor:
    features = feature_extractor(X, y)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    return features


def extract_features(feature_extractor: Any, X: Any, y: Any) -> torch.Tensor:
    return compute_features(feature_extractor, X, y).detach()


def features_to_numpy(features: Any) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        return features.detach().cpu().numpy()
    return np.asarray(features)
