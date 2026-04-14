from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any


def timestamp_now() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dump_pickle(obj: Any, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as handle:
        pickle.dump(obj, handle, -1)
    return target


def save_population_results(log_dir: str | Path, run_time: str, results: Any) -> Path:
    return dump_pickle(results, ensure_dir(log_dir) / f"{run_time}.pkl")


def save_best_feature_net(save_dir: str | Path, run_time: str, feature_net: Any) -> Path:
    return dump_pickle(feature_net, ensure_dir(save_dir) / f"{run_time}.pkl")
