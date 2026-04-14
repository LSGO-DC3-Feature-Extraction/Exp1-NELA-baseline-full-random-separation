from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters


def _shallow_copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    return value


def snapshot_config(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        items = config.items()
    else:
        items = vars(config).items()
    return {key: _shallow_copy_value(value) for key, value in items}


def config_from_snapshot(snapshot: Mapping[str, Any], **overrides: Any) -> SimpleNamespace:
    payload = {key: _shallow_copy_value(value) for key, value in snapshot.items()}
    payload.update(overrides)
    return SimpleNamespace(**payload)


def vector2nn(x: Any, net: torch.nn.Module) -> torch.nn.Module:
    params = list(net.parameters())
    expected = sum(param.nelement() for param in params)
    flat_vector = np.asarray(copy.copy(x), dtype=np.float32).reshape(-1)
    if flat_vector.size != expected:
        raise AssertionError("dim of x and net not match!")
    if not params:
        return net

    ref_param = params[0]
    tensor_vector = torch.as_tensor(flat_vector, dtype=ref_param.dtype, device=ref_param.device)
    with torch.no_grad():
        vector_to_parameters(tensor_vector, params)
    return net
