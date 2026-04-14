from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

from config import normalize_dataset_name

from ._cache import get_cached_dataset, load_pickle

_BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_SPECS = {
    "bbob": {
        "train_pickle": "dataset/trainset_v2.pkl",
        "test_pickle": "dataset/testset_v2.pkl",
        "bootstrap_modules": ("dataset.bbob",),
        "epochs": {
            "LDE": 70,
            "RL_PSO": 6,
            "RLEPSO": 175,
            "RL_DAS": 650,
            "DE_DDQN": 2,
            "GLEET": 62,
        },
    },
    "bbob-noisy": {
        "train_pickle": "dataset/noisy_trainset.pkl",
        "test_pickle": "dataset/noisy_testset.pkl",
        "bootstrap_modules": ("dataset.bbob",),
        "epochs": {
            "LDE": 40,
            "RL_PSO": 3,
            "RLEPSO": 105,
            "RL_DAS": 400,
            "DE_DDQN": 1,
            "GLEET": 36,
        },
    },
    "protein_docking": {
        "train_pickle": "dataset/pd_trainset.pkl",
        "test_pickle": "dataset/pd_testset.pkl",
        "bootstrap_modules": ("dataset.protein_docking",),
        "epochs": {
            "LDE": 1,
            "RL_PSO": 6,
            "RLEPSO": 24,
            "RL_DAS": 28,
            "DE_DDQN": 3,
            "GLEET": 20,
        },
    },
    "slice-50x20-two": {
        "epochs": {
            "LDE": 70,
            "RL_PSO": 6,
            "RLEPSO": 175,
            "RL_DAS": 650,
            "DE_DDQN": 2,
            "GLEET": 62,
        },
    },
}

_FEATURE_EPOCH_ALIASES = {
    "LDE_FE": "LDE",
    "RL_DAS_FE": "RL_DAS",
}


def _get_dataset_spec(dataset: str) -> dict:
    dataset = normalize_dataset_name(dataset)
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {supported}.") from exc


def get_epoch_dict(dataset):
    spec = _get_dataset_spec(dataset)
    epochs = dict(spec["epochs"])
    for alias, original in _FEATURE_EPOCH_ALIASES.items():
        if original in epochs:
            epochs[alias] = epochs[original]
    return epochs


def _resolve_dataset_file(relative_path: str) -> Path:
    path = _BASE_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def _bootstrap_pickle_modules(module_names: tuple[str, ...]) -> None:
    for module_name in module_names:
        import_module(module_name)


def _bootstrap_pickle_aliases(module_names: tuple[str, ...]) -> None:
    main_module = sys.modules["__main__"]
    for module_name in module_names:
        module = import_module(module_name)
        for symbol_name, symbol_value in vars(module).items():
            if not symbol_name.startswith("_"):
                setattr(main_module, symbol_name, symbol_value)


def construct_problem_set(dataset):
    dataset = normalize_dataset_name(dataset)
    spec = _get_dataset_spec(dataset)

    if dataset == "slice-50x20-two":
        from .slice_dataset import construct_slice_problem_set

        def _load_slice_dataset():
            corpus_root = _BASE_DIR / "prob_instance_slicer" / "sliced_dataset" / "i_50Dx20"
            if not corpus_root.exists():
                raise FileNotFoundError(f"Sliced dataset directory not found: {corpus_root}")
            return construct_slice_problem_set(corpus_root, selected_count=2)

        return get_cached_dataset(dataset, _load_slice_dataset)

    def _load_dataset():
        _bootstrap_pickle_modules(spec["bootstrap_modules"])
        _bootstrap_pickle_aliases(spec["bootstrap_modules"])
        train_path = _resolve_dataset_file(spec["train_pickle"])
        test_path = _resolve_dataset_file(spec["test_pickle"])
        return load_pickle(train_path), load_pickle(test_path)

    return get_cached_dataset(dataset, _load_dataset)
