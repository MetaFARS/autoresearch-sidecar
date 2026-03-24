from __future__ import annotations

import difflib
import json
from typing import Any


JsonDict = dict[str, Any]


def copy_messages(messages: list[JsonDict]) -> list[JsonDict]:
    return [{"role": item.get("role", ""), "content": item.get("content", "")} for item in messages]


def copy_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def emit_trace(trace: list[JsonDict] | None, event: str, **payload: Any) -> None:
    if trace is None:
        return
    item: JsonDict = {"event": event}
    item.update(payload)
    trace.append(item)


def trace_to_json(trace: list[JsonDict]) -> str:
    return json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True)


def compare_traces(left: list[JsonDict], right: list[JsonDict]) -> tuple[bool, str]:
    if left == right:
        return True, "oracle traces are identical"
    left_text = trace_to_json(left).splitlines()
    right_text = trace_to_json(right).splitlines()
    diff = "\n".join(difflib.unified_diff(left_text, right_text, fromfile="left_trace", tofile="right_trace", lineterm=""))
    return False, diff or "oracle traces differ"
