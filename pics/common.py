from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import pandas as pd


PICS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PICS_ROOT.parent
RECORD_ROOT = PROJECT_ROOT / "record"
DATA_DIR = PICS_ROOT / "data"
OUTPUT_DIR = PICS_ROOT / "output"

SLICE_HEADER_RE = re.compile(r"^\[(?P<phase>\w+)\s+epoch=(?P<epoch>\d+)\s+slice=(?P<slice_id>\d+).*\]$")
SLICE_VALUE_RE = re.compile(r"^fes=(?P<fes>\d+)\tx=\[(?P<x>.*)\]\ty=(?P<y>[-+0-9.eE]+)$")
EPOCH_LINE_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)\telapsed_sec=(?P<elapsed_sec>[-+0-9.eE]+)\tslice_count=(?P<slice_count>\d+)$"
)


@dataclass(frozen=True)
class ExperimentRef:
    problem_id: int
    exp_id: int
    path: Path


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def iter_experiments(record_root: Path = RECORD_ROOT) -> Iterator[ExperimentRef]:
    if not record_root.exists():
        return
    for problem_dir in sorted((p for p in record_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        problem_id = int(problem_dir.name)
        exp_dirs = sorted(
            (p for p in problem_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")),
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        for exp_dir in exp_dirs:
            yield ExperimentRef(problem_id=problem_id, exp_id=int(exp_dir.name.split("_", 1)[1]), path=exp_dir)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_slice_metadata(path: Path) -> dict:
    return load_json(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def read_epoch_history(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if not path.exists():
        return pd.DataFrame(columns=["epoch", "epoch_elapsed_sec", "epoch_slice_count"])
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = EPOCH_LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "epoch_elapsed_sec": float(match.group("elapsed_sec")),
                "epoch_slice_count": int(match.group("slice_count")),
            }
        )
    return pd.DataFrame(rows, columns=["epoch", "epoch_elapsed_sec", "epoch_slice_count"])


def parse_slice_history(path: Path, *, problem_id: int, exp_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_epoch: int | None = None
    current_slice_id: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header_match = SLICE_HEADER_RE.match(line)
        if header_match:
            current_epoch = int(header_match.group("epoch"))
            current_slice_id = int(header_match.group("slice_id"))
            continue
        value_match = SLICE_VALUE_RE.match(line)
        if value_match and current_epoch is not None and current_slice_id is not None:
            x_text = value_match.group("x").strip()
            position = [] if not x_text else [float(item) for item in x_text.split(",")]
            rows.append(
                {
                    "problem_id": problem_id,
                    "exp_id": exp_id,
                    "epoch": current_epoch,
                    "slice_id": current_slice_id,
                    "fes": int(value_match.group("fes")),
                    "y": float(value_match.group("y")),
                    "position": position,
                }
            )
    return rows


def experiment_label(agent: str, slice_length: int, use_fe: bool) -> str:
    fe_label = "FE on" if bool(use_fe) else "FE off"
    return f"{agent} | {slice_length} | {fe_label}"


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def bool_label(value: bool) -> str:
    return "FE on" if value else "FE off"


def problem_output_dir(problem_id: int) -> Path:
    path = OUTPUT_DIR / f"problem_{problem_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["savefig.bbox"] = "tight"
