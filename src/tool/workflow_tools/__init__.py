from importlib import import_module

_MODULE_MAP = {
    "BrowserTool": ".browser",
    "DeepResearcherTool": ".deep_researcher",
    "DeepAnalyzerTool": ".deep_analyzer",
    "ReporterTool": ".reporter",
    "ToolGeneratorTool": ".tool_generator",
    "SkillGeneratorTool": ".skill_generator",
    "TodoTool": ".todo",
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
    "BrowserTool",
    "DeepResearcherTool",
    "DeepAnalyzerTool",
    "ReporterTool",
    "ToolGeneratorTool",
    "SkillGeneratorTool",
    "TodoTool",
]
