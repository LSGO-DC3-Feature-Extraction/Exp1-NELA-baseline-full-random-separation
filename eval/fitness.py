from __future__ import annotations

import numpy as np

from ._registry import get_aggregator

EPS = 1e-8


def _coerce_series(results, cost_baseline, key: str) -> tuple[np.ndarray, np.ndarray]:
    if key not in cost_baseline:
        raise KeyError(f"Missing baseline for problem '{key}'.")
    result_values = np.asarray(results[key], dtype=float)
    baseline_values = np.asarray(cost_baseline[key], dtype=float)
    if result_values.shape != baseline_values.shape:
        raise ValueError(
            f"Result length mismatch for problem '{key}': "
            f"{result_values.shape} vs {baseline_values.shape}."
        )
    return result_values, baseline_values


def calculate_task_performance(results, cost_baseline):
    avg_result = {}
    for key in results:
        result_values, baseline_values = _coerce_series(results, cost_baseline, key)
        safe_result = np.maximum(result_values, EPS)
        safe_baseline = np.maximum(baseline_values, EPS)
        gains = (np.log10(safe_result) - np.log10(safe_baseline)) / (np.log10(safe_baseline) + 9.0)
        avg_result[key] = float(np.mean(gains))
    return avg_result


def calculate_compare_performance(results, cost_baseline):
    avg_result = {}
    for key in results:
        result_values, baseline_values = _coerce_series(results, cost_baseline, key)
        better_mask = (result_values <= baseline_values) | (
            (result_values <= EPS) & (baseline_values <= EPS)
        )
        avg_result[key] = float(np.mean(np.where(better_mask, -1.0, 0.0)))
    return avg_result


def calculate_z_performance(results, cost_baseline):
    avg_result = {}
    for key in results:
        result_values, baseline_values = _coerce_series(results, cost_baseline, key)
        raw_mean = float(np.mean(baseline_values))
        mean_baseline = max(raw_mean, EPS)
        sigma_baseline = max(float(np.std(baseline_values)), EPS)
        z_scores = (result_values - mean_baseline) / sigma_baseline
        if raw_mean <= EPS:
            z_scores = np.where(result_values <= EPS, 0.0, z_scores)
        avg_result[key] = float(np.mean(z_scores))
    return avg_result


FITNESS_MODE_REGISTRY = {
    "cont": calculate_task_performance,
    "comp": calculate_compare_performance,
    "z-score": calculate_z_performance,
}


def calculate_per_task_perf(raw_data, fitness_mode, cost_baseline):
    try:
        calculator = FITNESS_MODE_REGISTRY[fitness_mode]
    except KeyError as exc:
        supported = ", ".join(sorted(FITNESS_MODE_REGISTRY))
        raise ValueError(f"Unsupported fitness mode '{fitness_mode}'. Expected one of: {supported}.") from exc
    return calculator(raw_data, cost_baseline)


def calculate_aggregate_performance(task_performance_results, agent_list, in_task_agg, out_task_agg):
    if len(task_performance_results) != len(agent_list):
        raise ValueError(
            f"task_performance_results length {len(task_performance_results)} does not match "
            f"agent_list length {len(agent_list)}."
        )

    in_task_aggregator = get_aggregator(in_task_agg)
    out_task_aggregator = get_aggregator(out_task_agg)
    final_results = {"task_performance_results": task_performance_results}
    final_score_list = []
    per_task_scores = {}

    for task_id, task_result in enumerate(task_performance_results):
        task_perf = task_result["task_perf"]
        scores = np.asarray(list(task_perf.values()), dtype=float)
        per_score = float(in_task_aggregator(scores))
        per_task_scores[f"task-{task_id}"] = per_score
        final_score_list.append(per_score)

    final_results["per_task_scores"] = per_task_scores
    final_results["final_score"] = float(out_task_aggregator(np.asarray(final_score_list, dtype=float)))
    return final_results
