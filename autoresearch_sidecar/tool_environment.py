from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping

from .agent_trace import JsonDict, emit_trace


ToolHandler = Callable[[str], str]


@dataclass(frozen=True)
class ToolSpec:
    signature: str
    description: str

    def render(self) -> str:
        return f"{self.signature}: {self.description}"


@dataclass(frozen=True)
class ToolCatalog:
    tool_specs: Mapping[str, ToolSpec]


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    argument: str
    raw_text: str


@dataclass(frozen=True)
class ToolTurn:
    assistant_message: str
    invocation: ToolInvocation


class ToolHost:
    TOOL_CALL_PATTERN = r"""
        <tool>\s*
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)\(
        \s*
        (?P<quote>["'])
        (?P<argument>(?:\\.|(?!(?P=quote)).)*?)
        (?P=quote)
        \s*
        \)\s*
        </tool>
    """
    TOOL_CALL_RE = re.compile(
        TOOL_CALL_PATTERN,
        re.DOTALL | re.VERBOSE,
    )

    def __init__(
        self,
        tool_specs: Mapping[str, ToolSpec],
        handlers: Mapping[str, ToolHandler],
        trace: list[JsonDict] | None = None,
    ) -> None:
        self.tool_specs = dict(tool_specs)
        self.handlers = dict(handlers)
        self.trace = trace

    def render_protocol(self, allowed_tools: tuple[str, ...]) -> str:
        lines = ["Allowed tools for this phase:"]
        for name in allowed_tools:
            spec = self.tool_specs[name]
            lines.append(f"- {spec.render()}")
        lines.extend(
            (
                "",
                "Tool protocol:",
                "- Emit exactly one tool call wrapped as <tool>...</tool><stop>.",
                "- Only one tool call is allowed per assistant turn.",
                "- After the host injects <result>...</result>, continue the same phase.",
                "- When investigation is complete, stop calling tools and emit the required phase output.",
            )
        )
        return "\n".join(lines)

    def parse(self, raw_text: str) -> ToolInvocation | None:
        match = self.TOOL_CALL_RE.search(raw_text)
        if not match:
            return None
        return ToolInvocation(
            name=match.group("name"),
            argument=match.group("argument"),
            raw_text=match.group(0).strip(),
        )

    def parse_turn(self, raw_text: str) -> ToolTurn | None:
        normalized = raw_text
        if "</tool>" in normalized and "<stop>" not in normalized:
            normalized = f"{normalized}<stop>"

        matches = list(self.TOOL_CALL_RE.finditer(normalized))
        if not matches:
            return None
        if len(matches) > 1:
            raise RuntimeError("Assistant emitted multiple tool calls in a single turn.")
        invocation_match = matches[0]
        invocation = ToolInvocation(
            name=invocation_match.group("name"),
            argument=invocation_match.group("argument"),
            raw_text=invocation_match.group(0).strip(),
        )

        invocation_end = normalized.find(invocation.raw_text)
        if invocation_end < 0:
            return None
        invocation_end += len(invocation.raw_text)

        stop_end = normalized.find("<stop>", invocation_end)
        if stop_end < 0:
            return None
        stop_end += len("<stop>")

        return ToolTurn(
            assistant_message=normalized[:stop_end].strip(),
            invocation=invocation,
        )

    def execute(
        self,
        tool_name: str,
        argument: str,
        allowed_tools: tuple[str, ...],
        *,
        role_name: str,
        phase_name: str,
    ) -> str:
        emit_trace(
            self.trace,
            "tool_request",
            role=role_name,
            phase=phase_name,
            tool=tool_name,
            argument=argument,
        )
        if tool_name not in allowed_tools:
            result = f"Error: tool '{tool_name}' is not allowed in this phase."
            emit_trace(
                self.trace,
                "tool_result",
                role=role_name,
                phase=phase_name,
                tool=tool_name,
                argument=argument,
                result=result,
            )
            return result
        handler = self.handlers.get(tool_name)
        if handler is None:
            result = f"Error: unknown tool '{tool_name}'."
            emit_trace(
                self.trace,
                "tool_result",
                role=role_name,
                phase=phase_name,
                tool=tool_name,
                argument=argument,
                result=result,
            )
            return result
        try:
            result = handler(argument)
        except Exception as exc:
            result = f"Error: {tool_name}({argument!r}) failed with {exc!r}."
            emit_trace(
                self.trace,
                "tool_result",
                role=role_name,
                phase=phase_name,
                tool=tool_name,
                argument=argument,
                result=result,
            )
            return result
        rendered = result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False)
        emit_trace(
            self.trace,
            "tool_result",
            role=role_name,
            phase=phase_name,
            tool=tool_name,
            argument=argument,
            result=rendered,
        )
        return rendered
