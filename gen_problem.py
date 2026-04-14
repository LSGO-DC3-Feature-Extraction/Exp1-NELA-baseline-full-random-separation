import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sphere(z: np.ndarray) -> np.ndarray:
    return np.sum(z * z, axis=1)


def _elliptic(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    if dim <= 1:
        return np.sum(z * z, axis=1)
    weights = np.power(1.0e6, np.linspace(0.0, 1.0, dim))
    return np.sum(weights * z * z, axis=1)


def _rastrigin(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    return 10.0 * dim + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z), axis=1)


def _ackley(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    mean_sq = np.mean(z * z, axis=1)
    mean_cos = np.mean(np.cos(2.0 * np.pi * z), axis=1)
    return (
        20.0
        + np.e
        - 20.0 * np.exp(-0.2 * np.sqrt(mean_sq))
        - np.exp(mean_cos)
    )


def _schwefel12(z: np.ndarray) -> np.ndarray:
    prefix = np.cumsum(z, axis=1)
    return np.sum(prefix * prefix, axis=1)


def _griewank(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    denom = np.sqrt(np.arange(1, dim + 1, dtype=float))
    return 1.0 + np.sum(z * z, axis=1) / 4000.0 - np.prod(np.cos(z / denom), axis=1)


def _bent_cigar(z: np.ndarray) -> np.ndarray:
    if z.shape[1] == 1:
        return z[:, 0] * z[:, 0]
    return z[:, 0] * z[:, 0] + 1.0e6 * np.sum(z[:, 1:] * z[:, 1:], axis=1)


def _discus(z: np.ndarray) -> np.ndarray:
    if z.shape[1] == 1:
        return 1.0e6 * z[:, 0] * z[:, 0]
    return 1.0e6 * z[:, 0] * z[:, 0] + np.sum(z[:, 1:] * z[:, 1:], axis=1)


def _different_powers(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    if dim == 1:
        return np.abs(z[:, 0]) ** 2
    exponents = np.linspace(2.0, 6.0, dim)
    return np.sum(np.abs(z) ** exponents, axis=1)


def _zakharov(z: np.ndarray) -> np.ndarray:
    dim = z.shape[1]
    coeff = 0.5 * np.arange(1, dim + 1, dtype=float)
    linear = z @ coeff
    sq = np.sum(z * z, axis=1)
    return sq + linear * linear + linear**4


_BASIC_FUNCTIONS = {
    "sphere": _sphere,
    "elliptic": _elliptic,
    "rastrigin": _rastrigin,
    "ackley": _ackley,
    "schwefel12": _schwefel12,
    "griewank": _griewank,
    "bent_cigar": _bent_cigar,
    "discus": _discus,
    "different_powers": _different_powers,
    "zakharov": _zakharov,
}


_PROBLEM_LIBRARY = {
    1: {"name": "sphere", "kernel": "sphere", "lb": -5.0, "ub": 5.0, "rotation": None},
    2: {"name": "elliptic", "kernel": "elliptic", "lb": -5.0, "ub": 5.0, "rotation": None},
    3: {"name": "rastrigin", "kernel": "rastrigin", "lb": -5.0, "ub": 5.0, "rotation": None},
    4: {"name": "ackley", "kernel": "ackley", "lb": -5.0, "ub": 5.0, "rotation": None},
    5: {"name": "schwefel12", "kernel": "schwefel12", "lb": -5.0, "ub": 5.0, "rotation": None},
    6: {"name": "rotated_sphere", "kernel": "sphere", "lb": -5.0, "ub": 5.0, "rotation": "dense"},
    7: {"name": "rotated_elliptic", "kernel": "elliptic", "lb": -5.0, "ub": 5.0, "rotation": "dense"},
    8: {"name": "rotated_rastrigin", "kernel": "rastrigin", "lb": -5.0, "ub": 5.0, "rotation": "dense"},
    9: {"name": "rotated_ackley", "kernel": "ackley", "lb": -5.0, "ub": 5.0, "rotation": "dense"},
    10: {"name": "griewank", "kernel": "griewank", "lb": -5.0, "ub": 5.0, "rotation": None},
    11: {"name": "rotated_griewank", "kernel": "griewank", "lb": -5.0, "ub": 5.0, "rotation": "dense"},
    12: {"name": "bent_cigar", "kernel": "bent_cigar", "lb": -5.0, "ub": 5.0, "rotation": None},
    13: {"name": "discus", "kernel": "discus", "lb": -5.0, "ub": 5.0, "rotation": None},
    14: {"name": "different_powers", "kernel": "different_powers", "lb": -5.0, "ub": 5.0, "rotation": None},
    15: {"name": "zakharov", "kernel": "zakharov", "lb": -5.0, "ub": 10.0, "rotation": "dense"},
}


def _random_rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


@dataclass
class BaseProblemSpec:
    problem_id: int
    total_dim: int
    seed: int
    name: str
    kernel_name: str
    lb: float
    ub: float
    optimum: float
    params: dict[str, np.ndarray]

    @classmethod
    def create(cls, problem_id: int, total_dim: int, seed: int):
        config = _PROBLEM_LIBRARY[int(problem_id)]
        rng = np.random.default_rng(seed)
        params: dict[str, np.ndarray] = {}
        if config["rotation"] == "dense":
            params["rotation"] = _random_rotation(total_dim, rng)

        return cls(
            problem_id=int(problem_id),
            total_dim=int(total_dim),
            seed=int(seed),
            name=str(config["name"]),
            kernel_name=str(config["kernel"]),
            lb=float(config["lb"]),
            ub=float(config["ub"]),
            optimum=0.0,
            params=params,
        )

    def _transform(self, x: np.ndarray) -> np.ndarray:
        z = x
        rotation = self.params.get("rotation")
        if rotation is not None:
            z = z @ rotation
        return z

    @property
    def dim(self) -> int:
        return self.total_dim

    def evaluate_full(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = self._transform(x)
        return _BASIC_FUNCTIONS[self.kernel_name](z)

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        values = self.evaluate_full(x)
        return values

    def func(self, x):
        return self.eval(x)

    def reset(self):
        return None

    def to_metadata(self) -> dict:
        matrix_files = {name: f"{name}.npy" for name in self.params}
        return {
            "problem_id": self.problem_id,
            "total_dim": self.total_dim,
            "seed": self.seed,
            "name": self.name,
            "kernel_name": self.kernel_name,
            "lb": self.lb,
            "ub": self.ub,
            "optimum": self.optimum,
            "matrix_files": matrix_files,
        }

    @classmethod
    def from_metadata(cls, meta: dict, params: dict[str, np.ndarray]) -> "BaseProblemSpec":
        return cls(
            problem_id=int(meta["problem_id"]),
            total_dim=int(meta["total_dim"]),
            seed=int(meta["seed"]),
            name=str(meta["name"]),
            kernel_name=str(meta["kernel_name"]),
            lb=float(meta["lb"]),
            ub=float(meta["ub"]),
            optimum=float(meta["optimum"]),
            params=params,
        )


@dataclass
class SliceProblem:
    base_problem: BaseProblemSpec
    indices: np.ndarray
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=int)
        self.dim = int(self.indices.size)
        self.lb = float(self.base_problem.lb)
        self.ub = float(self.base_problem.ub)
        self.optimum = float(self.base_problem.optimum)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        embedded = np.full((x.shape[0], self.base_problem.total_dim), self.fill_value, dtype=float)
        embedded[:, self.indices] = x
        return embedded

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        squeeze = False
        if x.ndim == 1:
            if x.shape[0] != self.dim:
                raise ValueError(f"Expected vector of length {self.dim}, got {x.shape[0]}.")
            x = x[None, :]
            squeeze = True
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"Expected shape (n, {self.dim}), got {x.shape}.")
        else:
            raise ValueError(f"Unsupported input shape {x.shape}.")
        values = self.base_problem.evaluate_full(self._embed(x))
        return float(values[0]) if squeeze else values

    def func(self, x):
        return self.eval(x)

    def reset(self):
        return None


def _split_indices(total_dim: int, sliced_dim: int, seed: int) -> list[np.ndarray]:
    if sliced_dim <= 0:
        raise ValueError("sliced_dim must be positive.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_dim)
    return [perm[start : start + sliced_dim].copy() for start in range(0, total_dim, sliced_dim)]


def _func_dir(save_path: str | Path) -> Path:
    return Path(save_path) / "func"


def gen_slice_save(total_dim, sliced_dim, problem_id, save_path, seed, fill_value=0.0) -> None:
    func_dir = _func_dir(save_path)
    func_dir.mkdir(parents=True, exist_ok=True)

    base_problem = BaseProblemSpec.create(problem_id=int(problem_id), total_dim=int(total_dim), seed=int(seed))
    slices = _split_indices(total_dim=int(total_dim), sliced_dim=int(sliced_dim), seed=int(seed))

    for name, value in base_problem.params.items():
        np.save(func_dir / f"{name}.npy", value)

    meta = base_problem.to_metadata()
    slice_meta = {
        "count": len(slices),
        "sliced_dim": int(sliced_dim),
        "fill_value_default": float(fill_value),
        "slices": [
            {
                "slice_id": slice_id,
                "indices": indices.tolist(),
                "dim": int(indices.size),
            }
            for slice_id, indices in enumerate(slices)
        ],
    }

    (func_dir / "problem_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (func_dir / "slices.json").write_text(json.dumps(slice_meta, indent=2), encoding="utf-8")


def load_full_problem(save_path):
    func_dir = _func_dir(save_path)
    meta = json.loads((func_dir / "problem_meta.json").read_text(encoding="utf-8"))

    params = {
        name: np.load(func_dir / filename)
        for name, filename in meta.get("matrix_files", {}).items()
    }
    return BaseProblemSpec.from_metadata(meta, params)


def load_sliced_problem(save_path, id, fill_value=None):
    func_dir = _func_dir(save_path)
    slice_meta = json.loads((func_dir / "slices.json").read_text(encoding="utf-8"))
    base_problem = load_full_problem(save_path)

    slice_id = int(id)
    slices = slice_meta["slices"]
    selected = next(item for item in slices if int(item["slice_id"]) == slice_id)

    resolved_fill_value = (
        float(slice_meta["fill_value_default"]) if fill_value is None else float(fill_value)
    )
    return SliceProblem(
        base_problem=base_problem,
        indices=np.asarray(selected["indices"], dtype=int),
        fill_value=resolved_fill_value,
    )

__all__ = [
    "BaseProblemSpec",
    "SliceProblem",
    "gen_slice_save",
    "load_full_problem",
    "load_sliced_problem",
]
