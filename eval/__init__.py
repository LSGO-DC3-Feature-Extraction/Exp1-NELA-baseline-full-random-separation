from .cost_baseline import get_test_cost_baseline, get_train_cost_baseline
from .fitness import calculate_aggregate_performance, calculate_per_task_perf
from .utils import get_epoch_dict

__all__ = [
    "calculate_aggregate_performance",
    "calculate_per_task_perf",
    "build_feature_embedder",
    "construct_problem_set",
    "evaluate_fixed_feature_net",
    "evaluate_loaded_feature_checkpoint",
    "eval_net_population",
    "get_epoch_dict",
    "load_feature_checkpoint",
    "resolve_feature_checkpoint",
    "get_test_cost_baseline",
    "get_train_cost_baseline",
    "vector2nn",
]


def __getattr__(name):
    if name in {
        "build_feature_embedder",
        "construct_problem_set",
        "evaluate_fixed_feature_net",
        "evaluate_loaded_feature_checkpoint",
        "eval_net_population",
        "load_feature_checkpoint",
        "resolve_feature_checkpoint",
        "vector2nn",
    }:
        from .evaluator import (
            build_feature_embedder,
            construct_problem_set,
            evaluate_fixed_feature_net,
            evaluate_loaded_feature_checkpoint,
            eval_net_population,
            load_feature_checkpoint,
            resolve_feature_checkpoint,
            vector2nn,
        )

        return {
            "build_feature_embedder": build_feature_embedder,
            "construct_problem_set": construct_problem_set,
            "evaluate_fixed_feature_net": evaluate_fixed_feature_net,
            "evaluate_loaded_feature_checkpoint": evaluate_loaded_feature_checkpoint,
            "eval_net_population": eval_net_population,
            "load_feature_checkpoint": load_feature_checkpoint,
            "resolve_feature_checkpoint": resolve_feature_checkpoint,
            "vector2nn": vector2nn,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
