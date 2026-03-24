from __future__ import annotations

from typing import cast

from autoresearch_sidecar.agent_runtime import ChatCompletionClient, RoleRunner
from autoresearch_sidecar.tool_environment import ToolHost, ToolSpec
from autoresearch_sidecar.workflow_spec import OutputSpec, PhaseSpec, RoleSpec


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        return next(self._responses)


def _build_role() -> RoleSpec:
    return RoleSpec(
        name="planner",
        purpose="Collect experiment notes.",
        system_context="ctx",
        required_inputs=("snapshot",),
        field_descriptions={"snapshot": "Snapshot."},
        phases=(
            PhaseSpec(
                name="investigate",
                purpose="Inspect one node.",
                reads=("snapshot",),
                writes="experiment_notes",
                instructions="Use the tool if needed.",
                output=OutputSpec(
                    instructions="Return text.",
                    parser=lambda raw: raw.strip(),
                    validator=lambda value: value,
                ),
                allow_tools=True,
                allowed_tools=("read_note",),
            ),
        ),
    )


def test_tool_host_parse_preserves_escaped_and_multiline_arguments() -> None:
    tool_host = ToolHost(
        {"read_note": ToolSpec(signature="read_note(node_id: str) -> str", description="Read note.")},
        {"read_note": lambda node_id: f"note:{node_id}"},
    )

    escaped = tool_host.parse(r'<tool>read_note("node-\"1\"")</tool><stop>')
    assert escaped is not None
    assert escaped.name == "read_note"
    assert escaped.argument == r'node-\"1\"'

    multiline = tool_host.parse('<tool>read_note("line-1\nline-2")</tool><stop>')
    assert multiline is not None
    assert multiline.argument == "line-1\nline-2"


def test_role_runner_accepts_multiline_args_without_explicit_stop() -> None:
    trace: list[dict[str, object]] = []
    client = FakeClient(['<tool>read_note("line-1\nline-2")</tool>', "grounded note"])
    tool_host = ToolHost(
        {"read_note": ToolSpec(signature="read_note(node_id: str) -> str", description="Read note.")},
        {"read_note": lambda node_id: f"note:{node_id}"},
        trace=trace,
    )
    runner = RoleRunner(cast(ChatCompletionClient, client), tool_host, trace=trace)

    state = runner.run(_build_role(), {"snapshot": "seed"})

    assert state["experiment_notes"] == "grounded note"
    assert trace[2]["argument"] == "line-1\nline-2"
    assert trace[3]["result"] == "note:line-1\nline-2"


def test_role_runner_executes_first_tool_block_when_multiple_are_present() -> None:
    seen_args: list[str] = []
    trace: list[dict[str, object]] = []
    client = FakeClient(['<tool>read_note("first")</tool><stop><tool>read_note("second")</tool><stop>', "done"])

    def read_note(node_id: str) -> str:
        seen_args.append(node_id)
        return f"note:{node_id}"

    tool_host = ToolHost(
        {"read_note": ToolSpec(signature="read_note(node_id: str) -> str", description="Read note.")},
        {"read_note": read_note},
        trace=trace,
    )
    runner = RoleRunner(cast(ChatCompletionClient, client), tool_host, trace=trace)

    state = runner.run(_build_role(), {"snapshot": "seed"})

    assert state["experiment_notes"] == "done"
    assert seen_args == ["first"]
    assert trace[2]["argument"] == "first"
