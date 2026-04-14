from __future__ import annotations

from pathlib import Path

import numpy as np

from config import refresh_config
from feat_extractor.feature_extractor import Feature_Extractor

from ._cache import load_pickle
from ._compat import config_from_snapshot, snapshot_config, vector2nn as _vector2nn
from ._io import save_best_feature_net, save_population_results, timestamp_now
from .cost_baseline import get_test_cost_baseline, get_train_cost_baseline
from .fitness import calculate_aggregate_performance, calculate_per_task_perf
from .utils import construct_problem_set as _construct_problem_set

try:
    import ray
except ModuleNotFoundError as exc:
    _RAY_IMPORT_ERROR = exc

    class _RayProxy:
        @staticmethod
        def remote(*args, **kwargs):
            def decorator(fn):
                def _missing(*_args, **_kwargs):
                    raise ModuleNotFoundError("ray is required to use eval.evaluator") from _RAY_IMPORT_ERROR

                _missing.remote = _missing
                return _missing

            return decorator

        @staticmethod
        def put(*_args, **_kwargs):
            raise ModuleNotFoundError("ray is required to use eval.evaluator") from _RAY_IMPORT_ERROR

        @staticmethod
        def get(*_args, **_kwargs):
            raise ModuleNotFoundError("ray is required to use eval.evaluator") from _RAY_IMPORT_ERROR

    ray = _RayProxy()


def vector2nn(x, net):
    return _vector2nn(x, net)


def construct_problem_set(dataset="bbob"):
    return _construct_problem_set(dataset)


def build_feature_extractor(config):
    return Feature_Extractor(
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        is_mlp=getattr(config, "is_mlp", False),
    )


def build_feature_embedder(feature_net, config):
    return vector2nn(feature_net, build_feature_extractor(config))


def resolve_feature_checkpoint(load_path=None, records_root="records") -> Path:
    if load_path:
        path = Path(load_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Feature checkpoint not found: {path}")
        return path

    records_dir = Path(records_root)
    candidates = sorted(
        records_dir.rglob("*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.parent.name == "save_model":
            return candidate
    raise FileNotFoundError(f"No feature checkpoint found under {records_dir.resolve()}")


def load_feature_checkpoint(load_path=None, records_root="records"):
    checkpoint_path = resolve_feature_checkpoint(load_path=load_path, records_root=records_root)
    return checkpoint_path, load_pickle(checkpoint_path)


def _compact_rollout_summary(results):
    flattened = [float(value) for values in results.values() for value in values]
    return {
        "problem_count": len(results),
        "trial_count": len(flattened),
        "mean_final_cost": float(np.mean(flattened)) if flattened else None,
    }


def _resolve_eval_targets(config, target_prefix: str):
    agent_list = list(getattr(config, f"{target_prefix}_agent_list"))
    epoch_list = list(getattr(config, f"{target_prefix}_epoch_list"))
    if len(agent_list) != len(epoch_list):
        raise ValueError(
            f"{target_prefix}_agent_list length {len(agent_list)} does not match "
            f"{target_prefix}_epoch_list length {len(epoch_list)}."
        )
    return agent_list, epoch_list


@ray.remote(num_cpus=1, num_gpus=0)
def _evaluate_task(
    feature_idx,
    task_id,
    feature_net,
    agent_name,
    train_epoch,
    trainset,
    testset,
    config_snapshot,
    seed,
):
    from trainer import Trainer

    config = config_from_snapshot(
        config_snapshot,
        train_agent=f"{agent_name}_Agent",
        train_optimizer=f"{agent_name}_Optimizer",
        train_epoch=train_epoch,
    )
    feature_embedder = build_feature_embedder(feature_net, config)
    trainer = Trainer(config, trainset, testset, seed, feature_embedder)
    results = trainer.train(pick_best=True)
    baseline = get_test_cost_baseline(config.dataset)[agent_name]
    payload = {
        "task_perf": calculate_per_task_perf(
            results,
            fitness_mode=config.fitness_mode,
            cost_baseline=baseline,
        ),
    }
    if getattr(config, "eval_return_raw_data", False):
        payload["raw_data"] = results
    else:
        payload["summary"] = _compact_rollout_summary(results)
    return {
        "feature_idx": feature_idx,
        "task_id": task_id,
        "task_result": payload,
    }


def _aggregate_feature_results(raw_results, agent_list, in_task_agg, out_task_agg, n_features):
    grouped_results = {feature_idx: {} for feature_idx in range(n_features)}
    for item in raw_results:
        grouped_results[item["feature_idx"]][item["task_id"]] = item["task_result"]

    aggregated_results = []
    for feature_idx in range(n_features):
        ordered_results = [
            grouped_results[feature_idx][task_id]
            for task_id in range(len(agent_list))
        ]
        aggregated_results.append(
            calculate_aggregate_performance(
                ordered_results,
                agent_list,
                in_task_agg,
                out_task_agg,
            )
        )
    return aggregated_results


def _advance_eval_counter(config):
    call_index = getattr(config, "_eval_call_count", 0) + 1
    setattr(config, "_eval_call_count", call_index)
    return call_index


def _should_persist(config, call_index):
    if not getattr(config, "eval_save_population_results", False):
        return False
    interval = max(1, int(getattr(config, "eval_save_interval", 1)))
    return call_index % interval == 0


def _maybe_save_population_snapshot(config, run_time, results, call_index):
    if _should_persist(config, call_index):
        save_population_results(config.log_dir, run_time, results)


def _maybe_save_best_feature_net(config, run_time, feature_nets, final_scores):
    if not getattr(config, "eval_save_best_feature_net", True):
        return None
    best_idx = int(np.argmin(final_scores))
    best_score = float(final_scores[best_idx])
    previous_best = getattr(config, "_best_eval_score", None)
    should_save = True
    if getattr(config, "eval_save_only_if_improved", True) and previous_best is not None:
        should_save = best_score < previous_best
    if should_save:
        setattr(config, "_best_eval_score", best_score)
        return save_best_feature_net(config.save_dir, run_time, feature_nets[best_idx])
    return None


def _evaluate_feature_nets(feature_nets, config, agent_list, epoch_list, train_set, test_set, seed=0):
    config = refresh_config(config)
    config_snapshot = snapshot_config(config)
    train_ref = ray.put(train_set)
    test_ref = ray.put(test_set)
    task_refs = []
    for feature_idx, feature_net in enumerate(feature_nets):
        feature_ref = ray.put(feature_net)
        for task_id, (agent_name, train_epoch) in enumerate(zip(agent_list, epoch_list)):
            task_refs.append(
                _evaluate_task.remote(
                    feature_idx,
                    task_id,
                    feature_ref,
                    agent_name,
                    train_epoch,
                    train_ref,
                    test_ref,
                    config_snapshot,
                    seed,
                )
            )
    raw_results = ray.get(task_refs)
    return _aggregate_feature_results(
        raw_results,
        agent_list,
        config.in_task_agg,
        config.out_task_agg,
        len(feature_nets),
    )


def evaluate_fixed_feature_net(
    feature_net,
    config,
    *,
    target_prefix="test",
    train_set=None,
    test_set=None,
    seed=0,
):
    config = refresh_config(config)
    if train_set is None or test_set is None:
        train_set, test_set = construct_problem_set(config.dataset)
    agent_list, epoch_list = _resolve_eval_targets(config, target_prefix)
    return _evaluate_feature_nets(
        [feature_net],
        config,
        agent_list,
        epoch_list,
        train_set,
        test_set,
        seed=seed,
    )[0]


def eval_net_population(feature_nets, config):
    config = refresh_config(config)
    train_set, test_set = construct_problem_set(config.dataset)
    agent_list, epoch_list = _resolve_eval_targets(config, "train")
    results = _evaluate_feature_nets(
        list(feature_nets),
        config,
        agent_list,
        epoch_list,
        train_set,
        test_set,
    )

    run_time = timestamp_now()
    call_index = _advance_eval_counter(config)
    _maybe_save_population_snapshot(config, run_time, results, call_index)

    final_scores = [result["final_score"] for result in results]
    if getattr(config, "eval_verbose", True):
        print(final_scores)

    _maybe_save_best_feature_net(config, run_time, feature_nets, final_scores)
    return final_scores


def evaluate_loaded_feature_checkpoint(
    config,
    *,
    load_path=None,
    records_root="records",
    target_prefix="test",
    train_set=None,
    test_set=None,
    seed=0,
):
    checkpoint_path, feature_net = load_feature_checkpoint(load_path=load_path, records_root=records_root)
    result = evaluate_fixed_feature_net(
        feature_net,
        config,
        target_prefix=target_prefix,
        train_set=train_set,
        test_set=test_set,
        seed=seed,
    )
    return checkpoint_path, feature_net, result
