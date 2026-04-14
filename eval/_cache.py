from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

_PICKLE_CACHE: dict[Path, Any] = {}
_DATASET_CACHE: dict[str, tuple[Any, Any]] = {}
_BASELINE_CACHE: dict[str, Any] = {}


def load_pickle(path: Path) -> Any:
    resolved = path.resolve()
    if resolved not in _PICKLE_CACHE:
        with resolved.open("rb") as handle:
            _PICKLE_CACHE[resolved] = pickle.load(handle)
    return _PICKLE_CACHE[resolved]


def get_cached_dataset(dataset: str, factory: Callable[[], tuple[Any, Any]]) -> tuple[Any, Any]:
    if dataset not in _DATASET_CACHE:
        _DATASET_CACHE[dataset] = factory()
    return _DATASET_CACHE[dataset]


def get_cached_baseline(dataset: str, factory: Callable[[], Any]) -> Any:
    if dataset not in _BASELINE_CACHE:
        _BASELINE_CACHE[dataset] = factory()
    return _BASELINE_CACHE[dataset]
