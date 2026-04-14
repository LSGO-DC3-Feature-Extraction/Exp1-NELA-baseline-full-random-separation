from __future__ import annotations

from collections.abc import Callable

import numpy as np

AGGREGATOR_REGISTRY: dict[str, Callable[[np.ndarray], float]] = {
    "np.mean": np.mean,
    "np.median": np.median,
    "np.max": np.max,
    "np.sum": np.sum,
}


def get_aggregator(name: str) -> Callable[[np.ndarray], float]:
    try:
        return AGGREGATOR_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(AGGREGATOR_REGISTRY))
        raise ValueError(f"Unsupported aggregation '{name}'. Expected one of: {supported}.") from exc
