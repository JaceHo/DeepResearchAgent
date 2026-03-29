from importlib import import_module

_MODULE_MAP = {
    "BashTool": ".bash",
    "PythonInterpreterTool": ".python_interpreter",
    "DoneTool": ".done",
    "WebFetcherTool": ".web_fetcher",
    "WebSearcherTool": ".web_searcher",
    "MdifyTool": ".mdify",
    "LeetCodeTool": ".leetcode",
    "FileReaderTool": ".file_reader",
    "FileEditorTool": ".file_editor",
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
    "BashTool",
    "PythonInterpreterTool",
    "DoneTool",
    "WebFetcherTool",
    "WebSearcherTool",
    "MdifyTool",
    "LeetCodeTool",
    "FileReaderTool",
    "FileEditorTool",
]
