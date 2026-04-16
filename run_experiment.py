from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from feat_extractor.feature_extractor import Feature_Extractor
from gen_problem import gen_slice_save, load_full_problem, load_sliced_problem
from metabbo.basic_environment import PBO_Env
from metabbo.registry import create_agent, create_optimizer


ROOT = Path(__file__).resolve().parent
EXPERIMENT_PATH = ROOT / "experiment.json"
OUTPUT_TEMPLATE_PATH = ROOT / "output_sample.json"
FEATURE_CHECKPOINT_PATH = ROOT / "feat_extractor" / "checkpoint.pkl"
RECORD_ROOT = ROOT / "record"

DEFAULT_TOTAL_DIM = 1000
DEFAULT_TRAIN_EPOCH = 10
DEFAULT_MAX_FES = 2000
DEFAULT_MAX_LEARNING_STEP = 10
DEFAULT_DEVICE = "cpu"
DEFAULT_LOG_POINTS = 50
DEFAULT_USE_MAX_FES_LIMIT = False
DEFAULT_USE_MAX_LEARNING_STEP_LIMIT = False
DISABLED_MAX_LEARNING_STEP = 10**12
LOG_COLUMNS = ["phase", "epoch", "slice_id", "fes", "best_value", "learn_steps", "elapsed_sec", "message"]


class HistoryWriter:
    def __init__(self, path: Path, title: str):
        self.path = path
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{title}]\n")
            handle.flush()

    def write(self, *, fes, position, value):
        pos_text = ",".join(f"{float(item):.4f}" for item in np.asarray(position, dtype=float).tolist())
        line = f"fes={int(fes)}\tx=[{pos_text}]\ty={float(value):.4f}\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def load_experiments():
    with EXPERIMENT_PATH.open("r", encoding="utf-8") as handle:
        experiments = json.load(handle)
    experiments.sort(key=lambda item: int(item["exp_id"]))
    return experiments


def pick_experiments(experiments, exp_id):
    if exp_id is None:
        return experiments
    return [item for item in experiments if int(item["exp_id"]) == int(exp_id)]


def build_experiment_dir(setting):
    func_dir = RECORD_ROOT / str(int(setting["func_id"]))
    exp_dir = func_dir / f"exp_{int(setting['exp_id'])}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_problem_dirs(exp_dir: Path):
    train_dir = exp_dir / "train_problem"
    test_dir = exp_dir / "test_problem"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, test_dir


def clear_experiment_outputs(exp_dir: Path):
    for path in exp_dir.glob("history_slice_*.txt"):
        path.unlink()
    for name in ("agent.pkl", "log.csv", "output.json", "log.txt", "history_epoch.txt"):
        path = exp_dir / name
        if path.exists():
            path.unlink()


def resolve_seeds(setting):
    base_seed = int(setting["base_seed"])
    return base_seed, base_seed + 1


def build_runtime_config(setting, exp_dir: Path):
    slice_length = int(setting["slice_length"])
    use_max_fes_limit = bool(setting.get("use_max_fes_limit", DEFAULT_USE_MAX_FES_LIMIT))
    use_max_learning_step_limit = bool(
        setting.get("use_max_learning_step_limit", DEFAULT_USE_MAX_LEARNING_STEP_LIMIT)
    )
    if use_max_fes_limit:
        max_fes = int(setting.get("max_fes", DEFAULT_MAX_FES))
    else:
        max_fes = DEFAULT_MAX_FES
    if use_max_learning_step_limit:
        max_learning_step = int(setting.get("max_learning_step", DEFAULT_MAX_LEARNING_STEP))
    else:
        max_learning_step = DISABLED_MAX_LEARNING_STEP
    log_interval = max(1, max_fes // DEFAULT_LOG_POINTS)
    return SimpleNamespace(
        dim=slice_length,
        device=setting.get("device", DEFAULT_DEVICE),
        train_agent=str(setting["agent"]),
        train_optimizer=str(setting["agent"]),
        train_epoch=int(setting.get("train_epoch", DEFAULT_TRAIN_EPOCH)),
        maxFEs=max_fes,
        max_fes=max_fes,
        max_learning_step=max_learning_step,
        use_max_fes_limit=use_max_fes_limit,
        use_max_learning_step_limit=use_max_learning_step_limit,
        log_interval=log_interval,
        n_logpoint=DEFAULT_LOG_POINTS,
        use_ela=False,
        count_ela_fes=False,
        feat_node_dim=2,
        hidden_dim=64,
        n_layers=3,
        feat_n_heads=1,
        feat_ffh=64,
        feat_use_pe=True,
        is_mlp=False,
        agent_save_dir=str(exp_dir),
    )


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_feature_extractor(config):
    with FEATURE_CHECKPOINT_PATH.open("rb") as handle:
        vector = np.asarray(pickle.load(handle), dtype=np.float32).reshape(-1)

    for hidden_dim in range(1, 65):
        for n_layers in range(1, 9):
            for ffh in range(1, 129):
                total = 2 * hidden_dim + 2 * n_layers * (
                    4 * hidden_dim * hidden_dim + 2 * hidden_dim * ffh + ffh + 9 * hidden_dim
                )
                if total != vector.size:
                    continue
                config.hidden_dim = hidden_dim
                config.n_layers = n_layers
                config.feat_ffh = ffh
                config.feat_use_pe = (hidden_dim % 2 == 0)
                feature_extractor = Feature_Extractor(
                    node_dim=config.feat_node_dim,
                    hidden_dim=config.hidden_dim,
                    n_layers=config.n_layers,
                    n_heads=config.feat_n_heads,
                    ffh=config.feat_ffh,
                    use_pe=config.feat_use_pe,
                    is_mlp=config.is_mlp,
                ).to(config.device)
                params = list(feature_extractor.parameters())
                if sum(item.numel() for item in params) != vector.size:
                    continue
                with torch.no_grad():
                    tensor = torch.as_tensor(vector, dtype=params[0].dtype, device=params[0].device)
                    vector_to_parameters(tensor, params)
                feature_extractor.eval()
                return feature_extractor
    raise ValueError(f"Cannot load feature extractor from {FEATURE_CHECKPOINT_PATH}.")


def load_problem_set(problem_dir: Path):
    slice_meta_path = problem_dir / "func" / "slices.json"
    with slice_meta_path.open("r", encoding="utf-8") as handle:
        slice_meta = json.load(handle)
    problems = []
    for item in slice_meta["slices"]:
        slice_id = int(item["slice_id"])
        problems.append((slice_id, load_sliced_problem(problem_dir, slice_id)))
    return problems


def save_agent(agent, exp_dir: Path):
    with (exp_dir / "agent.pkl").open("wb") as handle:
        pickle.dump(agent, handle, -1)


def load_output_template(setting):
    with OUTPUT_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        output = json.load(handle)
    output["setting"] = {
        "exp_id": int(setting["exp_id"]),
        "func_id": int(setting["func_id"]),
        "slice_length": int(setting["slice_length"]),
        "agent": str(setting["agent"]),
        "use_fe": bool(setting["use_fe"]),
        "base_seed": int(setting["base_seed"]),
    }
    output["test_problem_expected_y"] = None
    output["test_problem_optimized_y"] = None
    output["test_problem_slices"] = []
    output["assembled_x"] = []
    return output


def write_output(exp_dir: Path, output):
    with (exp_dir / "output.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)


def format_display_float(value):
    return f"{float(value):.4f}"


def append_log(
    log_rows,
    *,
    phase,
    epoch="",
    slice_id="",
    fes="",
    best_value="",
    learn_steps="",
    elapsed_sec="",
    message="",
):
    log_rows.append(
        {
            "phase": phase,
            "epoch": epoch,
            "slice_id": slice_id,
            "fes": fes,
            "best_value": best_value,
            "learn_steps": learn_steps,
            "elapsed_sec": elapsed_sec,
            "message": message,
        }
    )


def write_log_csv(exp_dir: Path, log_rows):
    with (exp_dir / "log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(log_rows)


def append_epoch_history(exp_dir: Path, *, epoch: int, elapsed_sec: float, slice_count: int):
    line = f"epoch={epoch}\telapsed_sec={elapsed_sec:.4f}\tslice_count={slice_count}\n"
    with (exp_dir / "history_epoch.txt").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def train_agent(agent, feature_extractor, config, train_problems, exp_dir: Path, train_seed: int, log_rows):
    seed_all(train_seed)
    train_records = []
    train_start = time.perf_counter()
    for epoch in range(config.train_epoch):
        epoch_start = time.perf_counter()
        epoch_slice_count = 0
        for slice_id, problem in train_problems:
            optimizer = create_optimizer(config.train_optimizer, config, feature_extractor)
            title = f"train epoch={epoch} slice={slice_id} indices={problem.indices.astype(int).tolist()}"
            optimizer.history_writer = HistoryWriter(exp_dir / f"history_slice_{slice_id}.txt", title)
            env = PBO_Env(problem, optimizer)
            stop, info = agent.train_episode(env)
            epoch_slice_count += 1
            train_records.append(
                {
                    "epoch": epoch,
                    "slice_id": slice_id,
                    "best_value": float(info["gbest"]),
                    "learn_steps": int(info["learn_steps"]),
                    "fes": int(optimizer.fes),
                }
            )
            append_log(
                log_rows,
                phase="train",
                epoch=epoch,
                slice_id=slice_id,
                fes=int(optimizer.fes),
                best_value=format_display_float(info["gbest"]),
                learn_steps=int(info["learn_steps"]),
                elapsed_sec="",
                message=(
                    f"train_seed={train_seed};"
                    f"use_max_fes_limit={config.use_max_fes_limit};"
                    f"use_max_learning_step_limit={config.use_max_learning_step_limit}"
                ),
            )
            if config.use_max_learning_step_limit and stop:
                break
        epoch_elapsed = time.perf_counter() - epoch_start
        append_epoch_history(exp_dir, epoch=epoch, elapsed_sec=epoch_elapsed, slice_count=epoch_slice_count)
        append_log(
            log_rows,
            phase="train_epoch_summary",
            epoch=epoch,
            elapsed_sec=format_display_float(epoch_elapsed),
            message=f"slice_count={epoch_slice_count}",
        )
        if config.use_max_learning_step_limit and stop:
            break
    total_elapsed = time.perf_counter() - train_start
    append_log(
        log_rows,
        phase="train_summary",
        elapsed_sec=format_display_float(total_elapsed),
        message=f"epoch_count={config.train_epoch}",
    )
    return train_records


def test_agent(agent, feature_extractor, config, test_problems, exp_dir: Path, test_seed: int, output, log_rows):
    seed_all(test_seed)
    results = []
    for slice_id, problem in test_problems:
        optimizer = create_optimizer(config.train_optimizer, config, feature_extractor)
        title = f"test slice={slice_id} indices={problem.indices.astype(int).tolist()}"
        optimizer.history_writer = HistoryWriter(exp_dir / f"history_slice_{slice_id}.txt", title)
        env = PBO_Env(problem, optimizer)
        rollout = agent.rollout_episode(env)
        best_position = np.asarray(rollout["best_position"], dtype=float)
        best_value = float(rollout["best_value"])
        results.append(
            {
                "slice_id": slice_id,
                "slice_position": problem.indices.astype(int).tolist(),
                "position_expected": [0.0] * int(problem.dim),
                "position_get": best_position.tolist(),
                "pred_y": best_value,
                "fill_value": float(problem.fill_value),
            }
        )
        append_log(
            log_rows,
            phase="test",
            epoch="",
            slice_id=slice_id,
            fes=int(rollout["fes"]),
            best_value=format_display_float(best_value),
            learn_steps="",
            elapsed_sec="",
            message=f"test_seed={test_seed}",
        )
        output["test_problem_slices"] = sorted(results, key=lambda item: int(item["slice_id"]))
        write_output(exp_dir, output)
    return results


def assemble_full_solution(test_problem_dir: Path, slice_results):
    full_problem = load_full_problem(test_problem_dir)
    assembled_x = np.zeros(full_problem.dim, dtype=float)
    for item in slice_results:
        slice_problem = load_sliced_problem(test_problem_dir, item["slice_id"])
        assembled_x[slice_problem.indices] = np.asarray(item["position_get"], dtype=float)
    total_value = float(full_problem.eval(assembled_x)[0])
    expected_value = float(full_problem.eval(np.zeros(full_problem.dim, dtype=float))[0])
    return assembled_x, expected_value, total_value


def run_one_experiment(setting):
    exp_dir = build_experiment_dir(setting)
    clear_experiment_outputs(exp_dir)
    train_dir, test_dir = build_problem_dirs(exp_dir)
    train_seed, test_seed = resolve_seeds(setting)
    total_dim = int(setting.get("total_dim", DEFAULT_TOTAL_DIM))

    gen_slice_save(total_dim, int(setting["slice_length"]), int(setting["func_id"]), train_dir, train_seed)
    gen_slice_save(total_dim, int(setting["slice_length"]), int(setting["func_id"]), test_dir, test_seed)

    config = build_runtime_config(setting, exp_dir)
    feature_extractor = load_feature_extractor(config) if bool(setting["use_fe"]) else None
    agent = create_agent(config.train_agent, config, feature_extractor)

    output = load_output_template(setting)
    write_output(exp_dir, output)

    train_problems = load_problem_set(train_dir)
    test_problems = load_problem_set(test_dir)

    log_rows = []
    append_log(
        log_rows,
        phase="summary",
        elapsed_sec="",
        message=(
            f"exp_id={int(setting['exp_id'])};func_id={int(setting['func_id'])};"
            f"agent={setting['agent']};use_fe={bool(setting['use_fe'])};"
            f"train_seed={train_seed};test_seed={test_seed};slice_count={len(train_problems)};"
            f"use_max_fes_limit={config.use_max_fes_limit};"
            f"use_max_learning_step_limit={config.use_max_learning_step_limit};"
            f"max_fes={config.max_fes};max_learning_step={config.max_learning_step}"
        ),
    )

    train_records = train_agent(agent, feature_extractor, config, train_problems, exp_dir, train_seed, log_rows)
    save_agent(agent, exp_dir)

    test_results = test_agent(agent, feature_extractor, config, test_problems, exp_dir, test_seed, output, log_rows)
    assembled_x, expected_value, total_value = assemble_full_solution(test_dir, test_results)

    output["test_problem_expected_y"] = expected_value
    output["test_problem_optimized_y"] = total_value
    output["assembled_x"] = assembled_x.tolist()
    output["test_problem_slices"] = sorted(test_results, key=lambda item: int(item["slice_id"]))
    write_output(exp_dir, output)

    append_log(
        log_rows,
        phase="summary",
        elapsed_sec="",
        message=(
            f"train_records={len(train_records)};test_records={len(test_results)};"
            f"expected_y={format_display_float(expected_value)};"
            f"optimized_y={format_display_float(total_value)}"
        ),
    )
    write_log_csv(exp_dir, log_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", "--exp_id", dest="exp_id", type=int, default=None)
    args = parser.parse_args()

    experiments = pick_experiments(load_experiments(), args.exp_id)
    if not experiments:
        raise ValueError("No experiment selected.")

    for setting in experiments:
        run_one_experiment(setting)


if __name__ == "__main__":
    main()
