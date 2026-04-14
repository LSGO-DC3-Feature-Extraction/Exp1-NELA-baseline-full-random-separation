from pathlib import Path
from types import SimpleNamespace
import copy
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1 if sys.path else 0, str(PROJECT_ROOT))

from config import refresh_config
from eval.cost_baseline import get_train_cost_baseline
from eval.fitness import calculate_aggregate_performance, calculate_per_task_perf
from metabbo.basic_environment import PBO_Env
from metabbo.registry import create_agent, create_optimizer


def _snapshot_config(config):
    snapshot = {}
    for key, value in vars(config).items():
        if isinstance(value, list):
            snapshot[key] = list(value)
        elif isinstance(value, tuple):
            snapshot[key] = tuple(value)
        elif isinstance(value, dict):
            snapshot[key] = dict(value)
        elif isinstance(value, set):
            snapshot[key] = set(value)
        else:
            snapshot[key] = value
    return snapshot


def _align_training_results_with_baseline(raw_data, baseline):
    aligned = {}
    for problem_key, values in raw_data.items():
        result_values = list(values)
        baseline_values = baseline.get(problem_key)
        if baseline_values is None:
            aligned[problem_key] = result_values
            continue
        target_len = len(baseline_values)
        if len(result_values) == target_len:
            aligned[problem_key] = result_values
            continue
        if len(result_values) == 1 and target_len > 1:
            aligned[problem_key] = result_values * target_len
            continue
        raise ValueError(
            f"Cannot align training result length {len(result_values)} with "
            f"baseline length {target_len} for problem '{problem_key}'."
        )
    return aligned


class Trainer(object):
    def __init__(self, config, train_set, test_set, seed, fe=None, run_logger=None):
        self.config = refresh_config(config)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.agent = create_agent(self.config.train_agent, self.config, fe)
        self.optimizer = create_optimizer(self.config.train_optimizer, self.config, fe)
        self.train_set = train_set
        self.test_set = test_set
        self.fe = fe
        self.run_logger = run_logger

    def _make_eval_snapshot(self):
        return SimpleNamespace(
            agent=copy.deepcopy(self.agent),
            optimizer=copy.deepcopy(self.optimizer),
        )

    def _build_checkpoint_payload(self, epoch, tag):
        return {
            "epoch": epoch,
            "tag": tag,
            "config": _snapshot_config(self.config),
            "agent": self.agent,
            "optimizer": self.optimizer,
            "has_feature_extractor": self.fe is not None,
        }

    def _save_checkpoint(self, epoch, tag):
        save_dir = getattr(self.config, "save_checkpoint_dir", None)
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f"epoch-{epoch}.pkl")
        with open(checkpoint_path, "wb") as handle:
            pickle.dump(self._build_checkpoint_payload(epoch, tag), handle, -1)

    def _save_trival_snapshot(self):
        save_dir = os.path.join("records", "trival_saved")
        os.makedirs(save_dir, exist_ok=True)
        if not self.config.use_ela:
            filename = f"{self.config.train_agent}_{self.config.train_epoch}_{self.config.dataset}.pkl"
        else:
            filename = (
                f"{self.config.train_agent}_{self.config.train_epoch}"
                f"_ela_count{self.config.count_ela_fes}_{self.config.dataset}.pkl"
            )
        with open(os.path.join(save_dir, filename), "wb") as handle:
            pickle.dump(
                self._build_checkpoint_payload(self.config.train_epoch, "trival"),
                handle,
                -1,
            )

    def train(self, pick_best=False, trival=False, save_checkpoint=False):
        print("start training")
        exceed_max_lt = False
        epoch = 0
        train_start = time.time()
        if pick_best:
            best_agent = None
            best_perf = math.inf
        if trival:
            cost_trival_train = []

        while not exceed_max_lt:
            epoch_start = time.time()
            cost_one_episode = {}
            with tqdm(range(self.train_set.N), desc=f"Training {self.config.train_agent} Epoch {epoch}") as pbar:
                for problem in self.train_set:
                    problem_start = time.time()
                    problem_key = str(problem)
                    env = PBO_Env(problem, self.optimizer)
                    _, info = self.agent.train_episode(env)
                    cost_one_episode[problem_key] = [info["gbest"]]
                    if self.run_logger is not None:
                        self.run_logger.log_epoch_task(
                            epoch=epoch,
                            problem_key=problem_key,
                            best_value=float(info["gbest"]),
                            epoch_duration_sec=time.time() - epoch_start,
                            problem_duration_sec=time.time() - problem_start,
                            payload={
                                "learn_steps": int(info.get("learn_steps", -1)),
                                "normalizer": float(info.get("normalizer", 0.0)),
                                "task_best_value": float(info["gbest"]),
                            },
                        )
                    pbar.update(1)
            if pick_best:
                agent_name = self.config.train_agent[:-len("_Agent")]
                train_baseline = get_train_cost_baseline(self.config.dataset)[agent_name]
                aligned_cost_one_episode = _align_training_results_with_baseline(
                    cost_one_episode,
                    train_baseline,
                )
                task_perf = calculate_per_task_perf(
                    raw_data=aligned_cost_one_episode,
                    fitness_mode=self.config.fitness_mode,
                    cost_baseline=train_baseline,
                )
                perf = calculate_aggregate_performance(
                    task_performance_results=[
                        {"raw_data": aligned_cost_one_episode, "task_perf": task_perf}
                    ],
                    agent_list=[agent_name],
                    in_task_agg=self.config.in_task_agg,
                    out_task_agg=self.config.out_task_agg,
                )["final_score"]
                print(f"perf: {perf}\n task_perf: {task_perf}")
                if perf <= best_perf:
                    best_agent = self._make_eval_snapshot()
                    best_perf = perf

            if self.run_logger is not None and cost_one_episode:
                epoch_best = min(values[0] for values in cost_one_episode.values())
                self.run_logger.log_epoch_summary(
                    epoch=epoch,
                    epoch_duration_sec=time.time() - epoch_start,
                    best_value=float(epoch_best),
                    payload={
                        "task_count": len(cost_one_episode),
                    },
                )

            if save_checkpoint:
                self._save_checkpoint(epoch, "train")

            if trival and epoch > self.config.train_epoch - 3:
                cost_trival_train.append(cost_one_episode)

            epoch += 1

            if epoch > self.config.train_epoch:
                exceed_max_lt = True
                if trival:
                    print_result = {}
                    for problem_key in cost_trival_train[0].keys():
                        print_result[problem_key] = [
                            cost_trival_train[index][problem_key][0]
                            for index in range(len(cost_trival_train))
                        ]
                    print(f"Agent: {self.config.train_agent}, train_result: {print_result}")

        if trival:
            self._save_trival_snapshot()

        record = self.rollout(agent=self if not pick_best else best_agent)
        if self.run_logger is not None:
            flat_results = {
                problem_key: min(values) if values else None
                for problem_key, values in record.items()
            }
            self.run_logger.finalize(
                status="success",
                summary={
                    "train_duration_sec": time.time() - train_start,
                    "pipeline_dataset": self.config.dataset,
                    "agent_name": self.config.train_agent,
                    "optimizer_name": self.config.train_optimizer,
                    "train_epoch": self.config.train_epoch,
                    "train_problem_count": self.train_set.N,
                    "test_problem_count": self.test_set.N,
                    "per_task_final_best": flat_results,
                    "best_final_value": min(value for value in flat_results.values() if value is not None),
                },
            )
        return record

    def rollout(self, agent):
        print("start testing")
        cost_record = {}
        for problem in self.test_set:
            problem_key = str(problem)
            cost_record[problem_key] = []
            for seed in range(5):
                torch.manual_seed(seed)
                np.random.seed(seed)
                env = PBO_Env(problem, agent.optimizer)
                best_found_obj = agent.agent.rollout_episode(env)["cost"][-1]
                cost_record[problem_key].append(best_found_obj)
                if self.run_logger is not None:
                    self.run_logger.log_test_result(
                        problem_key=problem_key,
                        seed=seed,
                        best_value=float(best_found_obj),
                    )
        return cost_record
