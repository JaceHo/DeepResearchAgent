from importlib import import_module

_MODULE_MAP = {
    "ReformulatorTool": ".reformulator",
    "AICapabilityDebateTool": ".ai_capability_debate",
    "HumanCapabilityDebateTool": ".human_capability_debate",
    "FutureCapabilityDebateTool": ".future_capability_debate",
}


def __getattr__(name: str):
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "ReformulatorTool",
    "AICapabilityDebateTool",
    "HumanCapabilityDebateTool",
    "FutureCapabilityDebateTool",
]
