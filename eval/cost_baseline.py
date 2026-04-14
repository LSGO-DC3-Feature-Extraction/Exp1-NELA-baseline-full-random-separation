from __future__ import annotations

from importlib import import_module
from types import MappingProxyType
from typing import Any

from config import normalize_dataset_name

from ._cache import get_cached_baseline

_DATASET_MODULES = {
    "bbob": "._baseline_bbob",
    "bbob-noisy": "._baseline_bbob_noisy",
    "protein_docking": "._baseline_protein",
}

_FEATURE_ALIASES = {
    "RL_DAS_FE": "RL_DAS",
    "LDE_FE": "LDE",
}


def _freeze(value: Any):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _with_feature_aliases(data: dict[str, Any]):
    expanded = dict(data)
    for alias, original in _FEATURE_ALIASES.items():
        if original in expanded:
            expanded[alias] = expanded[original]
    return _freeze(expanded)


def _load_dataset_bundle(dataset: str) -> dict[str, MappingProxyType]:
    dataset = normalize_dataset_name(dataset)
    if dataset not in _DATASET_MODULES:
        supported = ", ".join(sorted(_DATASET_MODULES))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {supported}.")

    module = import_module(_DATASET_MODULES[dataset], __package__)
    return {
        "test": _with_feature_aliases(module.TEST_BASELINE),
        "train": _with_feature_aliases(module.TRAIN_BASELINE),
    }


def _get_dataset_bundle(dataset: str) -> dict[str, MappingProxyType]:
    normalized_dataset = normalize_dataset_name(dataset)
    return get_cached_baseline(normalized_dataset, lambda: _load_dataset_bundle(normalized_dataset))


def get_test_cost_baseline(dataset):
    return _get_dataset_bundle(dataset)["test"]


def get_train_cost_baseline(dataset):
    return _get_dataset_bundle(dataset)["train"]
