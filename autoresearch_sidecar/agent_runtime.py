from __future__ import annotations

import json
from typing import Any

import requests

from .tool_environment import ToolHost
from .agent_trace import JsonDict, copy_messages, copy_value, emit_trace
from .workflow_spec import PhaseSpec, RoleSpec, StateDict


class ChatCompletionClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[JsonDict], stop: list[str] | None = None) -> str:
        payload: JsonDict = {
            "model": self.model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"API error: {data}")
        message = data["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError(f"Model returned non-text content: {message}")


class RoleRunner:
    def __init__(
        self,
        client: ChatCompletionClient,
        tool_host: ToolHost,
        trace: list[JsonDict] | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.client = client
        self.tool_host = tool_host
        self.trace = trace
        self.debug_mode = debug_mode

    def run(self, role: RoleSpec, initial_state: StateDict) -> StateDict:
        missing = [name for name in role.required_inputs if name not in initial_state]
        if missing:
            raise ValueError(f"Missing required inputs for role {role.name}: {missing}")

        state = dict(initial_state)
        for phase in role.phases:
            raw_output = self._run_phase(role, phase, state)
            parsed = phase.output.parser(raw_output)
            if phase.output.validator is not None:
                parsed = phase.output.validator(parsed)
            state[phase.writes] = parsed
            emit_trace(
                self.trace,
                "phase_output",
                role=role.name,
                phase=phase.name,
                writes=[phase.writes],
                value=copy_value(parsed),
            )
        return state

    def _run_phase(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        history: list[JsonDict] = [
            {"role": "system", "content": self._render_system_prompt(role, phase)},
            {"role": "user", "content": self._render_user_prompt(role, phase, state)},
        ]

        tool_rounds = 0
        while True:
            stop_tokens = ["<stop>"] if phase.allow_tools else None
            emit_trace(
                self.trace,
                "oracle_request",
                role=role.name,
                phase=phase.name,
                messages=copy_messages(history),
                stop=list(stop_tokens) if stop_tokens is not None else None,
            )
            raw = self.client.complete(history, stop=stop_tokens)
            emit_trace(self.trace, "oracle_response", role=role.name, phase=phase.name, raw=raw)
            turn = self.tool_host.parse_turn(raw) if phase.allow_tools else None
            if turn is None:
                history.append({"role": "assistant", "content": raw})
                if self.debug_mode:
                    print(f"\n[phase {phase.name} output]\n{raw}\n", flush=True)
                return raw

            if tool_rounds >= phase.max_tool_rounds:
                raise RuntimeError(f"Phase {phase.name} exceeded its tool budget ({phase.max_tool_rounds}).")

            history.append({"role": "assistant", "content": turn.assistant_message})
            invocation = turn.invocation
            tool_result = self.tool_host.execute(
                invocation.name,
                invocation.argument,
                phase.allowed_tools,
                role_name=role.name,
                phase_name=phase.name,
            )
            history.append({"role": "user", "content": f"<result>\n{tool_result}\n</result>"})
            tool_rounds += 1

            if self.debug_mode:
                print(f"\n[tool {invocation.name}({invocation.argument!r})]\n{tool_result}\n", flush=True)

    def _render_system_prompt(self, role: RoleSpec, phase: PhaseSpec) -> str:
        lines = [
            role.system_context.strip(),
            f"Role: {role.name}",
            f"Role purpose: {role.purpose}",
            f"Current phase: {phase.name}",
            f"Phase purpose: {phase.purpose}",
            "Phase instructions:",
            phase.instructions.strip(),
        ]
        if role.invariants:
            lines.append("Role invariants:")
            lines.extend(f"- {item}" for item in role.invariants)
        if phase.allow_tools:
            lines.append(self.tool_host.render_protocol(phase.allowed_tools))
        lines.extend(("Output contract:", phase.output.instructions.strip()))
        return "\n".join(lines)

    def _render_user_prompt(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        lines = ["Phase state bindings:"]
        for field_name in phase.reads:
            description = role.field_descriptions.get(field_name, "")
            lines.append(f"\n[{field_name}] {description}" if description else f"\n[{field_name}]")
            lines.append(self._serialize(state.get(field_name)))
        lines.append("\nEmit the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
