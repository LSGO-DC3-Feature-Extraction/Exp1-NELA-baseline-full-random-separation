from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    RECORD_ROOT,
    ensure_dirs,
    experiment_label,
    iter_experiments,
    load_json,
    load_slice_metadata,
    parse_slice_history,
    read_csv_rows,
    read_epoch_history,
    safe_float,
    save_dataframe,
)

DEFAULT_FES_ANCHORS = list(range(100, 2001, 100))


def build_experiment_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in iter_experiments(RECORD_ROOT):
        output_path = exp.path / "output.json"
        if not output_path.exists():
            continue

        output = load_json(output_path)
        setting = output.get("setting", {})
        epoch_history = read_epoch_history(exp.path / "history_epoch.txt")
        total_elapsed_sec = None if epoch_history.empty else float(epoch_history["epoch_elapsed_sec"].sum())
        optimized_y = safe_float(output.get("test_problem_optimized_y"))
        expected_y = safe_float(output.get("test_problem_expected_y"))

        row = {
            "problem_id": exp.problem_id,
            "exp_id": int(setting.get("exp_id", exp.exp_id)),
            "agent": str(setting.get("agent", "")),
            "use_fe": bool(setting.get("use_fe", False)),
            "slice_length": int(setting.get("slice_length", 0)),
            "base_seed": int(setting.get("base_seed", 0)),
            "expected_y": expected_y,
            "optimized_y": optimized_y,
            "final_gap": None if optimized_y is None or expected_y is None else optimized_y - expected_y,
            "has_time_data": total_elapsed_sec is not None,
            "total_elapsed_sec": total_elapsed_sec,
        }
        row["exp_label"] = experiment_label(row["agent"], row["slice_length"], row["use_fe"])
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[
            "problem_id",
            "exp_id",
            "agent",
            "use_fe",
            "slice_length",
            "base_seed",
            "expected_y",
            "optimized_y",
            "final_gap",
            "has_time_data",
            "total_elapsed_sec",
            "exp_label",
        ],
    )
    if not df.empty:
        df = df.sort_values(["problem_id", "optimized_y", "exp_id"], na_position="last").reset_index(drop=True)
    return df


def build_epoch_progress() -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []
    for exp in iter_experiments(RECORD_ROOT):
        output_path = exp.path / "output.json"
        log_path = exp.path / "log.csv"
        if not output_path.exists() or not log_path.exists():
            continue

        output = load_json(output_path)
        setting = output.get("setting", {})
        exp_label = experiment_label(
            str(setting.get("agent", "")),
            int(setting.get("slice_length", 0)),
            bool(setting.get("use_fe", False)),
        )
        log_rows = read_csv_rows(log_path)
        train_rows = []
        for row in log_rows:
            if row.get("phase") != "train":
                continue
            epoch = row.get("epoch", "").strip()
            slice_id = row.get("slice_id", "").strip()
            best_value = safe_float(row.get("best_value"))
            if not epoch or not slice_id or best_value is None:
                continue
            train_rows.append(
                {
                    "problem_id": exp.problem_id,
                    "exp_id": int(setting.get("exp_id", exp.exp_id)),
                    "epoch": int(epoch),
                    "slice_id": int(slice_id),
                    "best_value": best_value,
                    "exp_label": exp_label,
                }
            )
        if not train_rows:
            continue

        train_df = pd.DataFrame(train_rows)
        agg_df = (
            train_df.groupby(["problem_id", "exp_id", "epoch", "exp_label"], as_index=False)["best_value"]
            .agg(
                epoch_mean_best="mean",
                epoch_median_best="median",
                epoch_best="min",
                epoch_worst="max",
                epoch_slice_count="count",
            )
            .sort_values(["problem_id", "exp_id", "epoch"])
        )

        epoch_history = read_epoch_history(exp.path / "history_epoch.txt")
        if not epoch_history.empty:
            agg_df = agg_df.merge(epoch_history, on="epoch", how="left")
        else:
            agg_df["epoch_elapsed_sec"] = pd.NA

        all_rows.extend(agg_df.to_dict("records"))

    df = pd.DataFrame(
        all_rows,
        columns=[
            "problem_id",
            "exp_id",
            "epoch",
            "epoch_mean_best",
            "epoch_median_best",
            "epoch_best",
            "epoch_worst",
            "epoch_slice_count",
            "epoch_elapsed_sec",
            "exp_label",
        ],
    )
    if not df.empty:
        df = df.sort_values(["problem_id", "exp_id", "epoch"]).reset_index(drop=True)
    return df


def build_slice_progress() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in iter_experiments(RECORD_ROOT):
        output_path = exp.path / "output.json"
        if not output_path.exists():
            continue
        output = load_json(output_path)
        setting = output.get("setting", {})
        exp_label = experiment_label(
            str(setting.get("agent", "")),
            int(setting.get("slice_length", 0)),
            bool(setting.get("use_fe", False)),
        )

        for history_path in sorted(exp.path.glob("history_slice_*.txt")):
            parsed = parse_slice_history(
                history_path,
                problem_id=exp.problem_id,
                exp_id=int(setting.get("exp_id", exp.exp_id)),
            )
            for row in parsed:
                row["exp_label"] = exp_label
                rows.append(row)

    df = pd.DataFrame(rows, columns=["problem_id", "exp_id", "epoch", "slice_id", "fes", "y", "exp_label"])
    if df.empty:
        df["is_final_point"] = pd.Series(dtype="boolean")
        return df

    df = df.sort_values(["problem_id", "exp_id", "epoch", "slice_id", "fes"]).reset_index(drop=True)
    final_mask = (
        df.groupby(["problem_id", "exp_id", "epoch", "slice_id"])["fes"].transform("max") == df["fes"]
    )
    df["is_final_point"] = final_mask
    return df


def _validate_and_build_index_map(slices_meta: dict) -> tuple[dict[int, list[int]], int]:
    slice_items = slices_meta.get("slices", [])
    index_map: dict[int, list[int]] = {}
    covered_indices: list[int] = []
    expected_dim_total = 0
    for item in slice_items:
        slice_id = int(item["slice_id"])
        indices = [int(index) for index in item["indices"]]
        index_map[slice_id] = indices
        covered_indices.extend(indices)
        expected_dim_total += int(item.get("dim", len(indices)))

    unique_indices = sorted(set(covered_indices))
    if len(unique_indices) != len(covered_indices):
        raise ValueError("Slice indices overlap; full vector reconstruction is ambiguous.")
    if len(unique_indices) != expected_dim_total:
        raise ValueError("Slice dimensions do not match the number of covered indices.")
    if not unique_indices:
        raise ValueError("No slice indices were found.")
    total_dim = len(unique_indices)
    expected_indices = list(range(total_dim))
    if unique_indices != expected_indices:
        raise ValueError("Slice indices do not cover a contiguous [0, total_dim) range.")
    return index_map, total_dim


def _get_state_at_anchor(slice_track: pd.DataFrame, fes_anchor: int) -> np.ndarray | None:
    available = slice_track[slice_track["fes"] <= fes_anchor]
    if available.empty:
        return None
    last_row = available.iloc[-1]
    return np.asarray(last_row["position"], dtype=float)


def build_vector_progress() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp in iter_experiments(RECORD_ROOT):
        output_path = exp.path / "output.json"
        slices_meta_path = exp.path / "test_problem" / "func" / "slices.json"
        if not output_path.exists() or not slices_meta_path.exists():
            continue

        output = load_json(output_path)
        setting = output.get("setting", {})
        exp_id = int(setting.get("exp_id", exp.exp_id))
        exp_label = experiment_label(
            str(setting.get("agent", "")),
            int(setting.get("slice_length", 0)),
            bool(setting.get("use_fe", False)),
        )
        slices_meta = load_slice_metadata(slices_meta_path)
        try:
            index_map, total_dim = _validate_and_build_index_map(slices_meta)
        except ValueError as exc:
            print(f"Skip problem={exp.problem_id} exp={exp_id}: {exc}")
            continue

        slice_tracks: dict[int, pd.DataFrame] = {}
        missing_slice = False
        for slice_id in sorted(index_map):
            history_path = exp.path / f"history_slice_{slice_id}.txt"
            if not history_path.exists():
                print(f"Skip problem={exp.problem_id} exp={exp_id}: missing {history_path.name}")
                missing_slice = True
                break
            parsed = parse_slice_history(history_path, problem_id=exp.problem_id, exp_id=exp_id)
            if not parsed:
                print(f"Skip problem={exp.problem_id} exp={exp_id}: empty history for slice {slice_id}")
                missing_slice = True
                break
            slice_df = pd.DataFrame(parsed).sort_values(["fes"]).reset_index(drop=True)
            slice_tracks[slice_id] = slice_df
        if missing_slice:
            continue

        prev_vector: np.ndarray | None = None
        start_vector: np.ndarray | None = None
        cumulative_path_length = 0.0
        for fes_anchor in DEFAULT_FES_ANCHORS:
            full_vector = np.empty(total_dim, dtype=float)
            full_vector[:] = np.nan
            anchor_ok = True
            for slice_id, indices in index_map.items():
                state = _get_state_at_anchor(slice_tracks[slice_id], fes_anchor)
                if state is None:
                    anchor_ok = False
                    break
                if state.size != len(indices):
                    anchor_ok = False
                    break
                full_vector[np.asarray(indices, dtype=int)] = state
            if not anchor_ok or np.isnan(full_vector).any():
                print(f"Skip problem={exp.problem_id} exp={exp_id} anchor={fes_anchor}: incomplete vector state")
                continue

            if start_vector is None:
                start_vector = full_vector.copy()

            vector_l2_norm = float(np.linalg.norm(full_vector, ord=2))
            step_path_length = 0.0 if prev_vector is None else float(np.linalg.norm(full_vector - prev_vector, ord=2))
            cumulative_path_length += step_path_length
            start_to_current = float(np.linalg.norm(full_vector - start_vector, ord=2))
            if math.isclose(start_to_current, 0.0, abs_tol=1e-12):
                path_length_ratio = 1.0
            else:
                path_length_ratio = cumulative_path_length / start_to_current

            rows.append(
                {
                    "problem_id": exp.problem_id,
                    "exp_id": exp_id,
                    "fes_anchor": fes_anchor,
                    "vector_l2_norm": vector_l2_norm,
                    "step_path_length": step_path_length,
                    "cumulative_path_length": cumulative_path_length,
                    "path_length_ratio": path_length_ratio,
                    "exp_label": exp_label,
                }
            )
            prev_vector = full_vector

    df = pd.DataFrame(
        rows,
        columns=[
            "problem_id",
            "exp_id",
            "fes_anchor",
            "vector_l2_norm",
            "step_path_length",
            "cumulative_path_length",
            "path_length_ratio",
            "exp_label",
        ],
    )
    if not df.empty:
        df = df.sort_values(["problem_id", "exp_id", "fes_anchor"]).reset_index(drop=True)
    return df


def main() -> None:
    ensure_dirs()
    experiment_summary = build_experiment_summary()
    epoch_progress = build_epoch_progress()
    slice_progress = build_slice_progress()
    vector_progress = build_vector_progress()

    save_dataframe(experiment_summary, DATA_DIR / "experiment_summary.csv")
    save_dataframe(epoch_progress, DATA_DIR / "epoch_progress.csv")
    save_dataframe(slice_progress, DATA_DIR / "slice_progress.csv")
    save_dataframe(vector_progress, DATA_DIR / "vector_progress.csv")

    print(f"Saved {len(experiment_summary)} experiment rows to {DATA_DIR / 'experiment_summary.csv'}")
    print(f"Saved {len(epoch_progress)} epoch rows to {DATA_DIR / 'epoch_progress.csv'}")
    print(f"Saved {len(slice_progress)} slice rows to {DATA_DIR / 'slice_progress.csv'}")
    print(f"Saved {len(vector_progress)} vector rows to {DATA_DIR / 'vector_progress.csv'}")


if __name__ == "__main__":
    main()
