import inspect
from typing import Any, Dict

import tools


TOOL_MAP = {
    "create_event": tools.create_event,
    "send_message": tools.send_message,
    "add_task": tools.add_task,
    "set_mode": tools.set_mode,
}


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'."

    fn = TOOL_MAP[name]
    sig = inspect.signature(fn)

    if not isinstance(args, dict):
        return "Error: tool arguments must be a JSON object."

    required = [
        pname
        for pname, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
    ]
    missing = [p for p in required if p not in args]
    if missing:
        return f"Error: missing required argument(s): {', '.join(missing)}."

    safe_args = {}
    for key in sig.parameters:
        if key in args:
            safe_args[key] = str(args[key])

    try:
        return fn(**safe_args)
    except TypeError as err:
        return f"Error: invalid arguments for '{name}': {err}."
    except Exception as err:  # pragma: no cover
        return f"Error: tool execution failed: {err}."
