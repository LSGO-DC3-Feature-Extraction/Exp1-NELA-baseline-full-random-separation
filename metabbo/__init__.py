"""Modified MetaBBO package with shared utilities and runtime registries."""

__all__ = [
    "AGENT_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "create_agent",
    "create_optimizer",
    "resolve_agent_class",
    "resolve_optimizer_class",
]


def __getattr__(name):
    if name in __all__:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
