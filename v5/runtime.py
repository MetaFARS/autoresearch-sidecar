
from __future__ import annotations

import asyncio
import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import requests

from compiler import CompiledForeachStep, CompiledHostStep, CompiledProgram, CompiledRole, CompiledRoleStep


def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": item.get("role", ""), "content": item.get("content", "")} for item in messages]


def _copy_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _trace_emit(trace: list[dict[str, Any]] | None, event: str, **payload: Any) -> None:
    if trace is None:
        return
    item = {"event": event}
    item.update(payload)
    trace.append(item)


def trace_to_json(trace: list[dict[str, Any]]) -> str:
    return json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True)


def compare_traces(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[bool, str]:
    if left == right:
        return True, "oracle traces are identical"
    left_text = trace_to_json(left).splitlines()
    right_text = trace_to_json(right).splitlines()
    diff = "\n".join(
        difflib.unified_diff(left_text, right_text, fromfile="generic_trace", tofile="residual_trace", lineterm="")
    )
    return False, diff or "oracle traces differ"


class KappaClient:
    def __init__(self, base_url: str, api_key: str, model: str, debug: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.debug = debug

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=300).json()
        if "choices" not in response:
            raise RuntimeError(f"API error: {response}")
        return response["choices"][0]["message"]["content"]


@dataclass
class ToolCall:
    name: str
    argument: str
    raw_text: str


class ToolHost:
    TOOL_CALL_RE = re.compile(r"<tool>\s*(\w+)\([\"\'](.*?)[\"\']\)\s*</tool>")

    def __init__(self, world: Any, trace: list[dict[str, Any]] | None = None, debug: bool = False):
        self.world = world
        self.trace = trace
        self.debug = debug
        self.handlers: dict[str, Callable[[str], str]] = {
            "read_meta": self.world.read_meta,
            "read_code": self.world.read_code,
            "read_stdout": self.world.read_stdout,
            "read_stderr": self.world.read_stderr,
        }

    def parse_tool_call(self, raw: str) -> ToolCall | None:
        match = self.TOOL_CALL_RE.search(raw)
        if not match:
            return None
        return ToolCall(name=match.group(1), argument=match.group(2), raw_text=match.group(0).strip())

    def execute(self, tool_name: str, argument: str, *, role_name: str, phase_name: str) -> str:
        if tool_name not in self.handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        _trace_emit(
            self.trace,
            "tool_request",
            role=role_name,
            phase=phase_name,
            tool=tool_name,
            argument=argument,
        )
        result = self.handlers[tool_name](argument)
        _trace_emit(
            self.trace,
            "tool_result",
            role=role_name,
            phase=phase_name,
            tool=tool_name,
            argument=argument,
            result=result,
        )
        return result


class RoleRunner:
    TOOL_BLOCK_RE = re.compile(r"(.*?<tool>\s*\w+\([\"\'].*?[\"\']\)\s*</tool><stop>)", re.DOTALL)

    def __init__(
        self,
        client: KappaClient,
        tool_host: ToolHost,
        trace: list[dict[str, Any]] | None = None,
        debug: bool = False,
    ):
        self.client = client
        self.tool_host = tool_host
        self.trace = trace
        self.debug = debug

    async def run(self, role: CompiledRole, initial_state: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_sync, role, initial_state)

    def _run_sync(self, role: CompiledRole, initial_state: dict[str, Any]) -> dict[str, Any]:
        local_state = dict(initial_state)
        stop_tokens = ["<stop>"]
        for phase in role.phases:
            history = [
                {"role": "system", "content": phase.system_prompt},
                {"role": "user", "content": self._render_user_prompt(phase.reads, phase.field_descriptions, local_state)},
            ]
            while True:
                _trace_emit(
                    self.trace,
                    "oracle_request",
                    role=role.name,
                    phase=phase.name,
                    messages=_copy_messages(history),
                    stop=list(stop_tokens),
                )
                raw = self.client.complete(history, stop=stop_tokens)
                _trace_emit(self.trace, "oracle_response", role=role.name, phase=phase.name, raw=raw)
                if "</tool>" in raw and "<stop>" not in raw:
                    raw += "<stop>"
                match = self.TOOL_BLOCK_RE.search(raw)
                if match:
                    if not phase.allow_tools:
                        raise PermissionError(f"Phase {phase.name} does not allow tools.")
                    assistant_message = match.group(1).strip()
                    history.append({"role": "assistant", "content": assistant_message})
                    invocation = self.tool_host.parse_tool_call(assistant_message)
                    if invocation is None:
                        raise ValueError(f"Malformed tool call: {assistant_message}")
                    if invocation.name not in phase.allowed_tools:
                        raise PermissionError(f"Phase {phase.name} cannot call {invocation.name}")
                    result = self.tool_host.execute(
                        invocation.name,
                        invocation.argument,
                        role_name=role.name,
                        phase_name=phase.name,
                    )
                    history.append({"role": "user", "content": f"<result>\n{result}\n</result>"})
                    if self.debug:
                        print(f"\n[tool] {invocation.name}({invocation.argument!r})\n{result}\n", flush=True)
                    continue

                parsed = phase.contract.parser(raw)
                validated = phase.contract.validator(parsed)
                self._write_phase_output(local_state, phase.writes, validated)
                _trace_emit(
                    self.trace,
                    "phase_output",
                    role=role.name,
                    phase=phase.name,
                    writes=list(phase.writes),
                    value=_copy_value(validated),
                )
                if self.debug:
                    print(f"\n[phase {phase.name} output]\n{validated!r}\n", flush=True)
                break
        return local_state

    def _render_user_prompt(
        self,
        reads: tuple[str, ...],
        field_descriptions: dict[str, str],
        state: dict[str, Any],
    ) -> str:
        lines = ["Phase state bindings:"]
        for field_name in reads:
            description = field_descriptions.get(field_name, "")
            value = state.get(field_name)
            if description:
                lines.append(f"\n[{field_name}] {description}\n{self._serialize(value)}")
            else:
                lines.append(f"\n[{field_name}]\n{self._serialize(value)}")
        lines.append("\nProduce the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    def _write_phase_output(self, state: dict[str, Any], writes: tuple[str, ...], value: Any) -> None:
        if len(writes) == 1:
            state[writes[0]] = value
            return
        if not isinstance(value, dict):
            raise ValueError(f"Phase writes {writes!r} require a dict output, got {type(value)!r}")
        for key in writes:
            if key not in value:
                raise ValueError(f"Phase output is missing expected field {key!r}")
            state[key] = value[key]


class RuntimeSession:
    def __init__(
        self,
        program: CompiledProgram,
        world: Any,
        client: KappaClient,
        trace: list[dict[str, Any]] | None = None,
        debug: bool = False,
    ):
        self.program = program
        self.world = world
        self.client = client
        self.trace = trace
        self.debug = debug
        self.role_runner = RoleRunner(client, ToolHost(world, trace=trace, debug=debug), trace=trace, debug=debug)

    async def run_workflow(self, memory: dict[str, Any] | None = None) -> dict[str, Any]:
        state = dict(memory or {})
        for step in self.program.workflow.steps:
            await self._run_step(step, state)
        return state

    async def _run_step(self, step: Any, state: dict[str, Any]) -> None:
        if isinstance(step, CompiledHostStep):
            args = step.arg_getter(state)
            result = step.fn(self.world, state, args)
            if asyncio.iscoroutine(result):
                result = await result
            if step.save_as is not None:
                state[step.save_as] = result
            if self.debug:
                print(f"[host-step] {step.name} -> {step.save_as}: {result!r}")
            return

        if isinstance(step, CompiledRoleStep):
            bindings = step.binding_getter(state)
            result = await self.role_runner.run(step.role, bindings)
            state[step.save_as] = result
            if self.debug:
                print(f"[role-step] {step.name} -> {step.save_as}: keys={sorted(result)}")
            return

        if isinstance(step, CompiledForeachStep):
            items = step.items_getter(state)
            if not isinstance(items, list):
                raise TypeError(f"Foreach step {step.name} expected a list, got {type(items)!r}")
            for item in items:
                state[step.item_name] = item
                for child in step.body:
                    await self._run_step(child, state)
            return

        raise TypeError(f"Unknown compiled step type: {type(step)!r}")
