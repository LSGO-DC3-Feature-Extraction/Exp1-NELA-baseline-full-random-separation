from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class RuntimeBinding:
    module_name: str
    symbol_name: str
    constructor_mode: str = "feature_flag"

    def resolve_class(self):
        module = import_module(self.module_name)
        return getattr(module, self.symbol_name)

    def build(self, config, feature_extractor):
        runtime_arg = (
            feature_extractor
            if self.constructor_mode == "feature_instance"
            else feature_extractor is not None
        )
        return self.resolve_class()(config, runtime_arg)


AGENT_REGISTRY = {
    "LDE": RuntimeBinding("metabbo.lde_agent", "LDE_Agent"),
    "LDE_Agent": RuntimeBinding("metabbo.lde_agent", "LDE_Agent"),
    "LDE_FE": RuntimeBinding(
        "finetune_agent_opt.lde_fe_agent",
        "LDE_FE_Agent",
        constructor_mode="feature_instance",
    ),
    "LDE_FE_Agent": RuntimeBinding(
        "finetune_agent_opt.lde_fe_agent",
        "LDE_FE_Agent",
        constructor_mode="feature_instance",
    ),
    "RL_PSO": RuntimeBinding("metabbo.rl_pso_agent", "RL_PSO_Agent"),
    "RL_PSO_Agent": RuntimeBinding("metabbo.rl_pso_agent", "RL_PSO_Agent"),
    "RLEPSO": RuntimeBinding("metabbo.rlepso_agent", "RLEPSO_Agent"),
    "RLEPSO_Agent": RuntimeBinding("metabbo.rlepso_agent", "RLEPSO_Agent"),
    "RL_DAS": RuntimeBinding("metabbo.rl_das_agent", "RL_DAS_Agent"),
    "RL_DAS_Agent": RuntimeBinding("metabbo.rl_das_agent", "RL_DAS_Agent"),
    "RL_DAS_FE": RuntimeBinding(
        "finetune_agent_opt.rl_das_fe_agent",
        "RL_DAS_FE_Agent",
        constructor_mode="feature_instance",
    ),
    "RL_DAS_FE_Agent": RuntimeBinding(
        "finetune_agent_opt.rl_das_fe_agent",
        "RL_DAS_FE_Agent",
        constructor_mode="feature_instance",
    ),
    "DE_DDQN": RuntimeBinding("metabbo.deddqn_agent", "DE_DDQN_Agent"),
    "DE_DDQN_Agent": RuntimeBinding("metabbo.deddqn_agent", "DE_DDQN_Agent"),
    "GLEET": RuntimeBinding("metabbo.gleet_agent", "GLEET_Agent"),
    "GLEET_Agent": RuntimeBinding("metabbo.gleet_agent", "GLEET_Agent"),
}

OPTIMIZER_REGISTRY = {
    "LDE": RuntimeBinding("metabbo.lde_optimizer", "LDE_Optimizer", constructor_mode="feature_instance"),
    "LDE_Optimizer": RuntimeBinding("metabbo.lde_optimizer", "LDE_Optimizer", constructor_mode="feature_instance"),
    "LDE_FE": RuntimeBinding(
        "finetune_agent_opt.lde_fe_optimizer",
        "LDE_FE_Optimizer",
        constructor_mode="feature_instance",
    ),
    "LDE_FE_Optimizer": RuntimeBinding(
        "finetune_agent_opt.lde_fe_optimizer",
        "LDE_FE_Optimizer",
        constructor_mode="feature_instance",
    ),
    "RL_PSO": RuntimeBinding("metabbo.rl_pso_optimizer", "RL_PSO_Optimizer", constructor_mode="feature_instance"),
    "RL_PSO_Optimizer": RuntimeBinding("metabbo.rl_pso_optimizer", "RL_PSO_Optimizer", constructor_mode="feature_instance"),
    "RLEPSO": RuntimeBinding("metabbo.rlepso_optimizer", "RLEPSO_Optimizer", constructor_mode="feature_instance"),
    "RLEPSO_Optimizer": RuntimeBinding("metabbo.rlepso_optimizer", "RLEPSO_Optimizer", constructor_mode="feature_instance"),
    "RL_DAS": RuntimeBinding("metabbo.rl_das_optimizer", "RL_DAS_Optimizer", constructor_mode="feature_instance"),
    "RL_DAS_Optimizer": RuntimeBinding("metabbo.rl_das_optimizer", "RL_DAS_Optimizer", constructor_mode="feature_instance"),
    "RL_DAS_FE": RuntimeBinding(
        "finetune_agent_opt.rl_das_fe_optimizer",
        "RL_DAS_FE_Optimizer",
        constructor_mode="feature_instance",
    ),
    "RL_DAS_FE_Optimizer": RuntimeBinding(
        "finetune_agent_opt.rl_das_fe_optimizer",
        "RL_DAS_FE_Optimizer",
        constructor_mode="feature_instance",
    ),
    "DE_DDQN": RuntimeBinding("metabbo.deddqn_optimizer", "DE_DDQN_Optimizer", constructor_mode="feature_instance"),
    "DE_DDQN_Optimizer": RuntimeBinding("metabbo.deddqn_optimizer", "DE_DDQN_Optimizer", constructor_mode="feature_instance"),
    "GLEET": RuntimeBinding("metabbo.gleet_optimizer", "GLEET_Optimizer", constructor_mode="feature_instance"),
    "GLEET_Optimizer": RuntimeBinding("metabbo.gleet_optimizer", "GLEET_Optimizer", constructor_mode="feature_instance"),
}


def _resolve(name: str, registry: dict[str, RuntimeBinding], registry_name: str) -> RuntimeBinding:
    try:
        return registry[name]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported {registry_name} '{name}'. Expected one of: {supported}."
        ) from exc


def resolve_agent_class(name: str):
    return _resolve(name, AGENT_REGISTRY, "agent").resolve_class()


def resolve_optimizer_class(name: str):
    return _resolve(name, OPTIMIZER_REGISTRY, "optimizer").resolve_class()


def create_agent(name: str, config: Any, feature_extractor=None):
    return _resolve(name, AGENT_REGISTRY, "agent").build(config, feature_extractor)


def create_optimizer(name: str, config: Any, feature_extractor=None):
    return _resolve(name, OPTIMIZER_REGISTRY, "optimizer").build(config, feature_extractor)
