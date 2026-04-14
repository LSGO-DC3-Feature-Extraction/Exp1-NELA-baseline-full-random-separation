from __future__ import annotations

import copy
import pickle

import ray
import numpy as np
import torch
from torch.nn.utils import vector_to_parameters
from tqdm import tqdm

from feat_extractor.feature_extractor import Feature_Extractor
from metabbo.basic_environment import PBO_Env
from metabbo.registry import create_agent, create_optimizer



class ProblemSet:
    def __init__(self, problems):
        self.problems = list(problems)
        self.N = len(self.problems)

    def __iter__(self):
        return iter(self.problems)

    def __len__(self):
        return self.N


class TrainConfig:
    def __init__(
        self,
        *,
        dim,
        device,
        train_agent,
        train_optimizer,
        train_epoch,
        max_fes,
        max_learning_step,
        log_interval,
        n_logpoint,
        feat_node_dim,
        hidden_dim,
        n_layers,
        feat_n_heads,
        feat_ffh,
        feat_use_pe,
        is_mlp,
        agent_save_dir,
    ):
        self.dim = int(dim)
        self.device = device
        self.train_agent = train_agent
        self.train_optimizer = train_optimizer
        self.train_epoch = int(train_epoch)
        self.maxFEs = int(max_fes)
        self.max_learning_step = int(max_learning_step)
        self.log_interval = int(log_interval)
        self.n_logpoint = int(n_logpoint)
        self.use_ela = False
        self.count_ela_fes = False
        self.feat_node_dim = int(feat_node_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.feat_n_heads = int(feat_n_heads)
        self.feat_ffh = int(feat_ffh)
        self.feat_use_pe = bool(feat_use_pe)
        self.is_mlp = bool(is_mlp)
        self.agent_save_dir = agent_save_dir


def _seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _problem_key(problem):
    if hasattr(problem, "dim"):
        return f"{problem.__class__.__name__}(dim={int(problem.dim)})"
    return str(problem)


def _algo_names(algo_name: str):
    base = algo_name
    if base.endswith("_Agent"):
        base = base[:-6]
    if base.endswith("_Optimizer"):
        base = base[:-10]
    return f"{base}_Agent", f"{base}_Optimizer"


def _load_feature_extractor(checkpoint_path, config):
    with open(checkpoint_path, "rb") as f:
        vector = np.asarray(pickle.load(f), dtype=np.float32).reshape(-1)

    specs = [
        dict(
            node_dim=config.feat_node_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.feat_n_heads,
            ffh=config.feat_ffh,
            use_pe=config.feat_use_pe,
            is_mlp=config.is_mlp,
        )
    ]

    # fallback for current checkpoint format in this repo
    target = int(vector.size)
    for hidden_dim in range(1, 65):
        for n_layers in range(1, 9):
            for ffh in range(1, 129):
                total = 2 * hidden_dim + 2 * n_layers * (
                    4 * hidden_dim * hidden_dim + 2 * hidden_dim * ffh + ffh + 9 * hidden_dim
                )
                if total == target:
                    specs.append(
                        dict(
                            node_dim=2,
                            hidden_dim=hidden_dim,
                            n_layers=n_layers,
                            n_heads=1,
                            ffh=ffh,
                            use_pe=(hidden_dim % 2 == 0),
                            is_mlp=False,
                        )
                    )

    for spec in specs:
        try:
            fe = Feature_Extractor(**spec).to(config.device)
            params = list(fe.parameters())
            total = sum(p.numel() for p in params)
            if total != vector.size:
                continue
            with torch.no_grad():
                vector_to_parameters(torch.as_tensor(vector, dtype=params[0].dtype, device=params[0].device), params)
            fe.eval()
            config.feat_node_dim = spec["node_dim"]
            config.hidden_dim = spec["hidden_dim"]
            config.n_layers = spec["n_layers"]
            config.feat_n_heads = spec["n_heads"]
            config.feat_ffh = spec["ffh"]
            config.feat_use_pe = spec["use_pe"]
            config.is_mlp = spec["is_mlp"]
            return fe, vector
        except Exception:
            continue

    raise ValueError(f"Cannot restore feature extractor from checkpoint: {checkpoint_path}")


def _rollout_once(agent, optimizer_name, config, problem, seed, feature_extractor):
    _seed(seed)
    optimizer = create_optimizer(optimizer_name, config, feature_extractor)
    env = PBO_Env(problem, optimizer)
    rollout = agent.rollout_episode(env)
    costs = [float(x) for x in rollout["cost"]]
    return {
        "problem_key": _problem_key(problem),
        "seed": int(seed),
        "final_cost": float(costs[-1]),
        "cost": costs,
        "episode_return": float(rollout["return"]),
    }


@ray.remote
def _ray_rollout_worker(agent_bytes, optimizer_name, config, problem, seed, feature_vector):
    agent = pickle.loads(agent_bytes)
    feature_extractor = None
    if feature_vector is not None:
        feature_vector = np.array(feature_vector, dtype=np.float32, copy=True)
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
        with torch.no_grad():
            vector_to_parameters(
                torch.as_tensor(feature_vector, dtype=params[0].dtype, device=params[0].device),
                params,
            )
        feature_extractor.eval()
    return _rollout_once(agent, optimizer_name, config, problem, seed, feature_extractor)


def train(
    train_set,
    test_set,
    algo_name,
    *,
    checkpoint_path="./feat_extractor/checkpoint.pkl",
    seed=0,
    use_feat_extractor=True,
    train_epoch=2,
    max_fes=2000,
    max_learning_step=10,
    device=None,
    log_interval=None,
    n_logpoint=50,
    test_seeds=(0, 1, 2, 3, 4),
    use_ray=True,
    ray_num_cpus=4,
    ray_test_parallelism=None,
    ray_init_kwargs=None,
    feat_node_dim=2,
    hidden_dim=64,
    n_layers=3,
    feat_n_heads=1,
    feat_ffh=None,
    feat_use_pe=True,
    is_mlp=False,
    show_progress=True,
    train_progress_desc=None,
    test_progress_desc=None,
):
    train_set = train_set if hasattr(train_set, "N") else ProblemSet(train_set)
    test_set = test_set if hasattr(test_set, "N") else ProblemSet(test_set)
    agent_name, optimizer_name = _algo_names(algo_name)
    dim = int(next(iter(train_set)).dim)
    device = device or "cpu"
    log_interval = max(1, log_interval or (int(max_fes) // max(1, int(n_logpoint))))

    config = TrainConfig(
        dim=dim,
        device=device,
        train_agent=agent_name,
        train_optimizer=optimizer_name,
        train_epoch=int(train_epoch),
        max_fes=int(max_fes),
        max_learning_step=int(max_learning_step),
        log_interval=int(log_interval),
        n_logpoint=int(n_logpoint),
        feat_node_dim=int(feat_node_dim),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        feat_n_heads=int(feat_n_heads),
        feat_ffh=int(hidden_dim if feat_ffh is None else feat_ffh),
        feat_use_pe=bool(feat_use_pe),
        is_mlp=bool(is_mlp),
        agent_save_dir="records/metabbo_train/agent",
    )

    feature_extractor = None
    feature_vector = None
    if use_feat_extractor:
        feature_extractor, feature_vector = _load_feature_extractor(checkpoint_path, config)

    _seed(seed)
    agent = create_agent(agent_name, config, feature_extractor)
    if hasattr(agent, "update_setting"):
        agent.update_setting(config)

    train_history = []
    stop = False
    train_bar = None
    if show_progress:
        train_bar = tqdm(total=config.max_learning_step, desc=train_progress_desc or f"Training {agent_name}")

    last_learn_steps = 0
    for epoch in range(config.train_epoch):
        if stop:
            break
        for problem in train_set:
            optimizer = create_optimizer(optimizer_name, config, feature_extractor)
            env = PBO_Env(problem, optimizer)
            _, info = agent.train_episode(env)
            learn_steps = int(info.get("learn_steps", 0))
            train_history.append(
                {
                    "epoch": epoch,
                    "problem_key": _problem_key(problem),
                    "best_value": float(info["gbest"]),
                    "episode_return": float(info["return"]),
                    "learn_steps": learn_steps,
                }
            )
            if train_bar is not None:
                delta = max(0, learn_steps - last_learn_steps)
                if delta:
                    train_bar.update(min(delta, config.max_learning_step - train_bar.n))
                last_learn_steps = max(last_learn_steps, learn_steps)
                train_bar.set_postfix(
                    epoch=epoch,
                    problem=_problem_key(problem),
                    best=f"{float(info['gbest']):.4g}",
                    learn_steps=learn_steps,
                )
            if learn_steps >= config.max_learning_step:
                stop = True
                break
    if train_bar is not None:
        train_bar.close()

    test_results = {}
    if use_ray:
        if not ray.is_initialized():
            kwargs = dict(ray_init_kwargs or {})
            if ray_num_cpus is not None:
                kwargs.setdefault("num_cpus", int(ray_num_cpus))
            ray.init(**kwargs)

        agent_bytes = pickle.dumps(agent, protocol=pickle.HIGHEST_PROTOCOL)
        refs = []
        for problem in test_set:
            for one_seed in test_seeds:
                refs.append(
                    _ray_rollout_worker.remote(
                        agent_bytes,
                        optimizer_name,
                        config,
                        problem,
                        int(one_seed),
                        feature_vector,
                    )
                )

        ray_bar = None
        if show_progress:
            ray_bar = tqdm(total=len(refs), desc=test_progress_desc or "Testing (Ray)")

        if ray_test_parallelism and ray_test_parallelism > 0:
            pending = refs
            active = pending[: int(ray_test_parallelism)]
            pending = pending[int(ray_test_parallelism) :]
            records = []
            while active:
                done, active = ray.wait(active, num_returns=1)
                done_records = ray.get(done)
                records.extend(done_records)
                if ray_bar is not None:
                    last = done_records[-1]
                    ray_bar.update(len(done_records))
                    ray_bar.set_postfix(seed=last["seed"], final_cost=f"{last['final_cost']:.4g}")
                if pending:
                    active.append(pending.pop(0))
        else:
            records = ray.get(refs)
            if ray_bar is not None:
                for item in records:
                    ray_bar.update(1)
                    ray_bar.set_postfix(seed=item["seed"], final_cost=f"{item['final_cost']:.4g}")
        if ray_bar is not None:
            ray_bar.close()
    else:
        records = []
        test_bar = None
        if show_progress:
            test_bar = tqdm(total=len(test_set), desc=test_progress_desc or "Testing")
        for problem in test_set:
            problem_records = []
            for one_seed in test_seeds:
                problem_records.append(
                    _rollout_once(
                        copy.deepcopy(agent),
                        optimizer_name,
                        config,
                        problem,
                        int(one_seed),
                        feature_extractor,
                    )
                )
            records.extend(problem_records)
            if test_bar is not None:
                test_bar.update(1)
                test_bar.set_postfix(problem=_problem_key(problem))
        if test_bar is not None:
            test_bar.close()

    for item in records:
        test_results.setdefault(item["problem_key"], []).append(item)
    for value in test_results.values():
        value.sort(key=lambda x: x["seed"])

    return {
        "train_history": train_history,
        "test_results": test_results,
        "resolved_agent_name": agent_name,
        "resolved_optimizer_name": optimizer_name,
        "config": config,
        "feature_extractor_enabled": feature_extractor is not None,
        "ray_enabled": bool(use_ray),
    }


__all__ = ["ProblemSet", "train"]
