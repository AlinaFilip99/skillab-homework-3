"""Tool Registry."""
import inspect
from typing import Any, Callable, get_type_hints

import pandas as pd
from pydantic import BaseModel

_REGISTRY: dict[str, dict] = {}


def register_tool(func: Callable) -> Callable:
    """Decorator care înregistrează un tool."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    has_inputs = "inputs" in hints and hints["inputs"] == list[pd.DataFrame]
    params_model = None

    for param_name, param in sig.parameters.items():
        if param_name == "inputs":
            continue
        if isinstance(param.annotation, type) and issubclass(param.annotation, BaseModel):
            params_model = param.annotation
            break

    if params_model is None:
        raise TypeError(f"{func.__name__}: requires a BaseModel params parameter")

    docstring = (func.__doc__ or "").strip()
    if not docstring:
        raise ValueError(f"{func.__name__}: requires a docstring")

    _REGISTRY[func.__name__] = {
        "func": func,
        "has_inputs": has_inputs,
        "params_model": params_model,
        "description": docstring.split("\n")[0],
    }
    return func


def call_tool(name: str, inputs: list[pd.DataFrame] | None = None, **params) -> Any:
    """Execută un tool."""
    if name not in _REGISTRY:
        raise KeyError(f"Tool '{name}' not found")

    t = _REGISTRY[name]
    validated_params = t["params_model"](**params)

    if t["has_inputs"]:
        return t["func"](inputs or [], validated_params)
    return t["func"](validated_params)


def get_tools_catalog() -> str:
    """Generează catalog pentru LLM."""
    if not _REGISTRY:
        return "No tools available."

    lines = []
    for name, t in _REGISTRY.items():
        if t["has_inputs"]:
            lines.append(f"- {name} (inputs: list): {t['description']}")
        else:
            lines.append(f"- {name}: {t['description']}")

        params = [
            f"{f}: {info.description or '?'}"
            for f, info in t["params_model"].model_fields.items()
        ]
        if params:
            lines.append(f"  Params: {'; '.join(params)}")

    return "\n".join(lines)
