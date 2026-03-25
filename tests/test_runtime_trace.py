from __future__ import annotations

from typing import cast

from autoresearch_sidecar.agent_runtime import ChatCompletionClient, RoleRunner
from autoresearch_sidecar.tool_environment import ToolHost, ToolSpec
from autoresearch_sidecar.workflow_spec import CommitSpec, OutputSpec, PhaseSpec, RoleSpec


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        return next(self._responses)


def test_role_runner_emits_v5_trace_events() -> None:
    trace: list[dict[str, object]] = []
    client = FakeClient(['<tool>read_note("node-1")</tool><stop>', "grounded note"])
    tool_host = ToolHost(
        {"read_note": ToolSpec(signature="read_note(node_id: str) -> str", description="Read note.")},
        {"read_note": lambda node_id: f"note:{node_id}"},
        trace=trace,
    )
    runner = RoleRunner(cast(ChatCompletionClient, client), tool_host, trace=trace)
    role = RoleSpec(
        name="planner",
        purpose="Collect experiment notes.",
        system_context="ctx",
        required_inputs=("snapshot",),
        field_descriptions={"snapshot": "Snapshot."},
        phases=(
            PhaseSpec(
                name="investigate",
                purpose="Inspect one node.",
                dominant_mode="observe",
                reads=("snapshot",),
                instructions="Use the tool if needed.",
                commit=CommitSpec(
                    writes="experiment_notes",
                    output=OutputSpec(
                        instructions="Return text.",
                        parser=lambda raw: raw.strip(),
                        validator=lambda value: value,
                    ),
                ),
                allow_tools=True,
                allowed_tools=("read_note",),
            ),
        ),
    )

    state = runner.run(role, {"snapshot": "seed"})

    assert state["experiment_notes"] == "grounded note"
    assert [event["event"] for event in trace] == [
        "oracle_request",
        "oracle_response",
        "tool_request",
        "tool_result",
        "oracle_request",
        "oracle_response",
        "phase_output",
    ]
    assert trace[0]["role"] == "planner"
    assert trace[0]["phase"] == "investigate"
    assert trace[0]["stop"] == ["<stop>"]
    messages = cast(list[dict[str, object]], trace[0]["messages"])
    assert "Dominant cognitive mode: observe" in str(messages[0]["content"])
    assert trace[2]["tool"] == "read_note"
    assert trace[2]["argument"] == "node-1"
    assert trace[3]["result"] == "note:node-1"
    assert trace[-1]["writes"] == ["experiment_notes"]
    assert trace[-1]["value"] == "grounded note"
