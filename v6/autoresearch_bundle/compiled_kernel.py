from __future__ import annotations

"""Task-family residual kernel with trace instrumentation emitted for autoresearch_program.

This kernel bakes in the task-family workflow and role runners while preserving
a uniform oracle-trace event format so it can be compared directly against the
generic runtime.
"""

import asyncio
import difflib
import json
import re
from typing import Any, Callable

import requests

from .world_adapter import AutoresearchProgramWorldAdapter, adapt_world

PROGRAM_NAME = 'autoresearch_program'
PROGRAM_DESCRIPTION = "CompiledProgram<autoresearch_program>\n\nRoles:\n- planner: inputs=['snapshot'] outputs=['proposals']\n    phase investigate: reads=['snapshot'] writes=['research_notes'] tools=['read_meta', 'read_code', 'read_stdout', 'read_stderr']\n    phase emit_proposals: reads=['snapshot', 'research_notes'] writes=['proposals'] tools=[]\n- implementer: inputs=['illustration', 'node_id', 'parent_id', 'snapshot', 'tldr'] outputs=['python_source']\n    phase inspect_parent: reads=['snapshot', 'node_id', 'parent_id', 'tldr', 'illustration'] writes=['implementation_plan'] tools=['read_meta', 'read_code', 'read_stdout', 'read_stderr']\n    phase emit_code: reads=['snapshot', 'node_id', 'parent_id', 'tldr', 'illustration', 'implementation_plan'] writes=['python_source'] tools=[]\n\nWorkflow slots:\n- current_node\n- implementer_run\n- new_node_ids\n- pending_node_ids\n- planner_run\n- snapshot"

def capture_snapshot(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> str:
    return world.snapshot()


def materialize_proposals(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> list[str]:
    proposals = args["proposals"]
    node_ids: list[str] = []
    for item in proposals:
        node = world.add_idea(
            parent_id=item.get("parent_id"),
            tldr=item.get("tldr", ""),
            illustration=item.get("illustration", ""),
        )
        node_ids.append(node.node_id)
    return node_ids


def list_unimplemented_nodes(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> list[str]:
    return [node.node_id for node in world.pending_nodes() if not world.has_code(node.node_id)]


def load_node_record(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> dict[str, object]:
    return world.get_node_record(args["node_id"])


def write_generated_source(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> None:
    world.write_code(args["node_id"], args["source"])


async def run_pending_experiments(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> None:
    await world.run_pending_experiments()


def parse_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError("Expected non-empty text output.")
    return text


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*(.*?)\s*```$", re.DOTALL)


def parse_code(raw: str) -> str:
    text = raw.strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    if not text:
        raise ValueError("Expected non-empty source code.")
    return text


_JSON_LIST_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_DICT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_json_list(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_LIST_RE.search(text)
        if match is None:
            raise
        return json.loads(match.group(1))


def parse_json_dict(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_DICT_RE.search(text)
        if match is None:
            raise
        return json.loads(match.group(1))


def identity(value: Any) -> Any:
    return value


def validate_nonempty_text(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty string.")
    return value


_REQUIRED_PROPOSAL_KEYS = {"parent_id", "tldr", "illustration"}


def validate_proposal_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Planner output must be a list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Proposal {index} must be an object.")
        missing = _REQUIRED_PROPOSAL_KEYS - set(item)
        if missing:
            raise ValueError(f"Proposal {index} is missing keys: {sorted(missing)}")
        parent_id = item.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"Proposal {index} has invalid parent_id: {parent_id!r}")
        tldr = item.get("tldr")
        illustration = item.get("illustration")
        if not isinstance(tldr, str) or not tldr.strip():
            raise ValueError(f"Proposal {index} has empty tldr.")
        if not isinstance(illustration, str) or not illustration.strip():
            raise ValueError(f"Proposal {index} has empty illustration.")
        normalized.append(
            {
                "parent_id": parent_id,
                "tldr": tldr.strip(),
                "illustration": illustration.strip(),
            }
        )
    return normalized


_METRIC_HINT_RE = re.compile(r"Metric\s*:")


def validate_python_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Generated source must be a non-empty string.")
    source = value.strip()
    compile(source, "<generated.py>", "exec")
    if not _METRIC_HINT_RE.search(source):
        raise ValueError("Generated code must contain a 'Metric:' print path.")
    return source


PARSERS = {
    "text": parse_text,
    "code": parse_code,
    "json_list": parse_json_list,
    "json_dict": parse_json_dict,
}

VALIDATORS = {
    None: identity,
    "nonempty_text": validate_nonempty_text,
    "proposal_list": validate_proposal_list,
    "python_source": validate_python_source,
}


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
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
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


class ToolCall:
    def __init__(self, name: str, argument: str, raw_text: str):
        self.name = name
        self.argument = argument
        self.raw_text = raw_text


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


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _render_user_prompt(reads: list[str], field_descriptions: dict[str, str], state: dict[str, Any]) -> str:
    lines = ["Phase state bindings:"]
    for field_name in reads:
        description = field_descriptions.get(field_name, "")
        value = state.get(field_name)
        if description:
            lines.append(f"\n[{field_name}] {description}\n{_serialize(value)}")
        else:
            lines.append(f"\n[{field_name}]\n{_serialize(value)}")
    lines.append("\nProduce the required phase output now.")
    return "\n".join(lines)


def _write_phase_output(state: dict[str, Any], writes: list[str], value: Any) -> None:
    if len(writes) == 1:
        state[writes[0]] = value
        return
    if not isinstance(value, dict):
        raise ValueError(f"Phase writes {writes!r} require a dict output, got {type(value)!r}")
    for key in writes:
        if key not in value:
            raise ValueError(f"Phase output is missing expected field {key!r}")
        state[key] = value[key]


def _execute_phase_sync(
    client: KappaClient,
    tool_host: ToolHost,
    *,
    role_name: str,
    phase_name: str,
    system_prompt: str,
    reads: list[str],
    field_descriptions: dict[str, str],
    allow_tools: bool,
    allowed_tools: list[str],
    parser_name: str,
    validator_name: str | None,
    writes: list[str],
    state: dict[str, Any],
    trace: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> None:
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _render_user_prompt(reads, field_descriptions, state)},
    ]
    tool_block_re = re.compile(r"(.*?<tool>\s*\w+\([\"\'].*?[\"\']\)\s*</tool><stop>)", re.DOTALL)
    stop_tokens = ["<stop>"]

    while True:
        _trace_emit(
            trace,
            "oracle_request",
            role=role_name,
            phase=phase_name,
            messages=_copy_messages(history),
            stop=list(stop_tokens),
        )
        raw = client.complete(history, stop=stop_tokens)
        _trace_emit(trace, "oracle_response", role=role_name, phase=phase_name, raw=raw)
        if "</tool>" in raw and "<stop>" not in raw:
            raw += "<stop>"
        match = tool_block_re.search(raw)
        if match:
            if not allow_tools:
                raise PermissionError(f"Phase {phase_name} does not allow tools.")
            assistant_message = match.group(1).strip()
            history.append({"role": "assistant", "content": assistant_message})
            invocation = tool_host.parse_tool_call(assistant_message)
            if invocation is None:
                raise ValueError(f"Malformed tool call: {assistant_message}")
            if invocation.name not in allowed_tools:
                raise PermissionError(f"Phase {phase_name} cannot call {invocation.name}")
            result = tool_host.execute(invocation.name, invocation.argument, role_name=role_name, phase_name=phase_name)
            history.append({"role": "user", "content": f"<result>\n{result}\n</result>"})
            if debug:
                print(f"\n[tool] {invocation.name}({invocation.argument!r})\n{result}\n", flush=True)
            continue

        parsed = PARSERS[parser_name](raw)
        validated = VALIDATORS[validator_name](parsed)
        _write_phase_output(state, writes, validated)
        _trace_emit(
            trace,
            "phase_output",
            role=role_name,
            phase=phase_name,
            writes=list(writes),
            value=_copy_value(validated),
        )
        if debug:
            print(f"\n[phase {phase_name} output]\n{validated!r}\n", flush=True)
        return


async def _execute_phase(
    client: KappaClient,
    tool_host: ToolHost,
    *,
    role_name: str,
    phase_name: str,
    system_prompt: str,
    reads: list[str],
    field_descriptions: dict[str, str],
    allow_tools: bool,
    allowed_tools: list[str],
    parser_name: str,
    validator_name: str | None,
    writes: list[str],
    state: dict[str, Any],
    trace: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> None:
    await asyncio.to_thread(
        _execute_phase_sync,
        client,
        tool_host,
        role_name=role_name,
        phase_name=phase_name,
        system_prompt=system_prompt,
        reads=reads,
        field_descriptions=field_descriptions,
        allow_tools=allow_tools,
        allowed_tools=allowed_tools,
        parser_name=parser_name,
        validator_name=validator_name,
        writes=writes,
        state=state,
        trace=trace,
        debug=debug,
    )


def describe_program() -> str:
    return PROGRAM_DESCRIPTION



async def run_planner_role(
    client: KappaClient,
    tool_host: ToolHost,
    initial_state: dict[str, Any],
    trace: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    state = dict(initial_state)
    await _execute_phase(
        client,
        tool_host,
        role_name='planner',
        phase_name='investigate',
        system_prompt='The research system manages a hierarchical tree of iterative experiments.\nEach experiment is encapsulated as an Idea object with the following fields:\n- node_id (str): unique ID for the experiment node.\n- parent_id (str | None): ID of the base experiment this extends.\n- illustration (str): technical explanation of the hypothesis and design.\n- tldr (str): concise summary of the proposal.\n- metric (float | None): the primary quantitative result. Larger is always better.\n- exit_code (int | None): 0 for successful execution, non-zero for crashes.\n- status (IdeaStatus): one of pending, running, success, failed.\nRole name: planner\nRole purpose: Propose the next batch of autoresearch ideas from the current tree state.\nState schema:\n- snapshot (input): The current serialized research tree snapshot.\n- research_notes (working): Findings collected from tool-assisted investigation.\n- proposals (output): A list of proposal objects with parent_id, tldr, illustration.\nCurrent phase: investigate\nPhase purpose: Inspect prior experiments and collect explicit research notes.\nPhase instructions:\nYou are investigating the current research tree.\nUse tools one call at a time to inspect promising successful leaves and instructive failures.\nSummarize the strongest opportunities for improvement, debugging, or exploration.\nDo not output proposals yet. Output only concise research notes.\nRole invariants:\n- Investigation and emission are separate phases.\n- All investigative actions must go through the declared tools.\n- The final proposal list must be valid JSON with the required schema.\n- Prefer improving strong successful leaves or debugging informative failures.\nOutput contract:\nReturn only a concise block of research notes. No JSON. No markdown fences.\nAvailable tools:\n- read_meta(node_id: str) -> str: Read the metadata JSON for a research node.\n- read_code(node_id: str) -> str: Read the generated Python source code for a research node.\n- read_stdout(node_id: str) -> str: Read the stdout log for a research node.\n- read_stderr(node_id: str) -> str: Read the stderr log for a research node.\n\nTool protocol:\nTo call a tool, emit exactly one call inside <tool></tool><stop>.\nExample: <tool>read_meta("2d28a7")</tool><stop>\nThe host will execute the tool and append the result inside <result></result>.',
        reads=['snapshot'],
        field_descriptions={'snapshot': 'The current serialized research tree snapshot.', 'research_notes': 'Findings collected from tool-assisted investigation.', 'proposals': 'A list of proposal objects with parent_id, tldr, illustration.'},
        allow_tools=True,
        allowed_tools=['read_meta', 'read_code', 'read_stdout', 'read_stderr'],
        parser_name='text',
        validator_name='nonempty_text',
        writes=['research_notes'],
        state=state,
        trace=trace,
        debug=debug,
    )
    await _execute_phase(
        client,
        tool_host,
        role_name='planner',
        phase_name='emit_proposals',
        system_prompt='The research system manages a hierarchical tree of iterative experiments.\nEach experiment is encapsulated as an Idea object with the following fields:\n- node_id (str): unique ID for the experiment node.\n- parent_id (str | None): ID of the base experiment this extends.\n- illustration (str): technical explanation of the hypothesis and design.\n- tldr (str): concise summary of the proposal.\n- metric (float | None): the primary quantitative result. Larger is always better.\n- exit_code (int | None): 0 for successful execution, non-zero for crashes.\n- status (IdeaStatus): one of pending, running, success, failed.\nRole name: planner\nRole purpose: Propose the next batch of autoresearch ideas from the current tree state.\nState schema:\n- snapshot (input): The current serialized research tree snapshot.\n- research_notes (working): Findings collected from tool-assisted investigation.\n- proposals (output): A list of proposal objects with parent_id, tldr, illustration.\nCurrent phase: emit_proposals\nPhase purpose: Transform the research notes into executable proposal objects.\nPhase instructions:\nNow produce the next batch of proposals.\nEach proposal must name a valid parent node when possible, provide a terse tldr, and give a concrete illustration.\nDo not include any commentary outside the JSON list.\nRole invariants:\n- Investigation and emission are separate phases.\n- All investigative actions must go through the declared tools.\n- The final proposal list must be valid JSON with the required schema.\n- Prefer improving strong successful leaves or debugging informative failures.\nOutput contract:\nReturn ONLY valid JSON with this exact schema:\n[\n  {"parent_id": "...", "tldr": "...", "illustration": "..."}\n]\nNo markdown fences. No conversational text.\nTool use is disabled in this phase. Emit the output directly.',
        reads=['snapshot', 'research_notes'],
        field_descriptions={'snapshot': 'The current serialized research tree snapshot.', 'research_notes': 'Findings collected from tool-assisted investigation.', 'proposals': 'A list of proposal objects with parent_id, tldr, illustration.'},
        allow_tools=False,
        allowed_tools=[],
        parser_name='json_list',
        validator_name='proposal_list',
        writes=['proposals'],
        state=state,
        trace=trace,
        debug=debug,
    )
    return state


async def run_implementer_role(
    client: KappaClient,
    tool_host: ToolHost,
    initial_state: dict[str, Any],
    trace: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    state = dict(initial_state)
    await _execute_phase(
        client,
        tool_host,
        role_name='implementer',
        phase_name='inspect_parent',
        system_prompt='The research system manages a hierarchical tree of iterative experiments.\nEach experiment is encapsulated as an Idea object with the following fields:\n- node_id (str): unique ID for the experiment node.\n- parent_id (str | None): ID of the base experiment this extends.\n- illustration (str): technical explanation of the hypothesis and design.\n- tldr (str): concise summary of the proposal.\n- metric (float | None): the primary quantitative result. Larger is always better.\n- exit_code (int | None): 0 for successful execution, non-zero for crashes.\n- status (IdeaStatus): one of pending, running, success, failed.\nRole name: implementer\nRole purpose: Implement one proposal as runnable Python code relative to a chosen parent node.\nState schema:\n- snapshot (input): The current serialized research tree snapshot.\n- node_id (input): The node being implemented.\n- parent_id (input): The parent node for consistency and inheritance.\n- tldr (input): Short summary of the target idea.\n- illustration (input): Detailed explanation of the target idea.\n- implementation_plan (working): A concrete implementation plan grounded in the parent node.\n- python_source (output): Runnable Python source code for the node.\nCurrent phase: inspect_parent\nPhase purpose: Study the parent node and prepare an explicit implementation plan.\nPhase instructions:\nInspect the parent code, metadata, and logs as needed.\nWork out how this node should differ from its parent while remaining executable.\nOutput only an implementation plan, not code.\nRole invariants:\n- Inspection and emission are separate phases.\n- Implementation must remain grounded in the selected parent node.\n- The final program must print exactly one metric line of the form Metric: <value>.\nOutput contract:\nReturn only a concise implementation plan. No code fences. No source code yet.\nAvailable tools:\n- read_meta(node_id: str) -> str: Read the metadata JSON for a research node.\n- read_code(node_id: str) -> str: Read the generated Python source code for a research node.\n- read_stdout(node_id: str) -> str: Read the stdout log for a research node.\n- read_stderr(node_id: str) -> str: Read the stderr log for a research node.\n\nTool protocol:\nTo call a tool, emit exactly one call inside <tool></tool><stop>.\nExample: <tool>read_meta("2d28a7")</tool><stop>\nThe host will execute the tool and append the result inside <result></result>.',
        reads=['snapshot', 'node_id', 'parent_id', 'tldr', 'illustration'],
        field_descriptions={'snapshot': 'The current serialized research tree snapshot.', 'node_id': 'The node being implemented.', 'parent_id': 'The parent node for consistency and inheritance.', 'tldr': 'Short summary of the target idea.', 'illustration': 'Detailed explanation of the target idea.', 'implementation_plan': 'A concrete implementation plan grounded in the parent node.', 'python_source': 'Runnable Python source code for the node.'},
        allow_tools=True,
        allowed_tools=['read_meta', 'read_code', 'read_stdout', 'read_stderr'],
        parser_name='text',
        validator_name='nonempty_text',
        writes=['implementation_plan'],
        state=state,
        trace=trace,
        debug=debug,
    )
    await _execute_phase(
        client,
        tool_host,
        role_name='implementer',
        phase_name='emit_code',
        system_prompt='The research system manages a hierarchical tree of iterative experiments.\nEach experiment is encapsulated as an Idea object with the following fields:\n- node_id (str): unique ID for the experiment node.\n- parent_id (str | None): ID of the base experiment this extends.\n- illustration (str): technical explanation of the hypothesis and design.\n- tldr (str): concise summary of the proposal.\n- metric (float | None): the primary quantitative result. Larger is always better.\n- exit_code (int | None): 0 for successful execution, non-zero for crashes.\n- status (IdeaStatus): one of pending, running, success, failed.\nRole name: implementer\nRole purpose: Implement one proposal as runnable Python code relative to a chosen parent node.\nState schema:\n- snapshot (input): The current serialized research tree snapshot.\n- node_id (input): The node being implemented.\n- parent_id (input): The parent node for consistency and inheritance.\n- tldr (input): Short summary of the target idea.\n- illustration (input): Detailed explanation of the target idea.\n- implementation_plan (working): A concrete implementation plan grounded in the parent node.\n- python_source (output): Runnable Python source code for the node.\nCurrent phase: emit_code\nPhase purpose: Emit the final runnable source code.\nPhase instructions:\nGenerate the full runnable Python source for the target node.\nThe program must execute end-to-end and print the final quantitative result on exactly one line as: Metric: <value>\nDo not include any commentary or markdown fences.\nRole invariants:\n- Inspection and emission are separate phases.\n- Implementation must remain grounded in the selected parent node.\n- The final program must print exactly one metric line of the form Metric: <value>.\nOutput contract:\nReturn ONLY runnable Python source code. No markdown fences. No explanations.\nTool use is disabled in this phase. Emit the output directly.',
        reads=['snapshot', 'node_id', 'parent_id', 'tldr', 'illustration', 'implementation_plan'],
        field_descriptions={'snapshot': 'The current serialized research tree snapshot.', 'node_id': 'The node being implemented.', 'parent_id': 'The parent node for consistency and inheritance.', 'tldr': 'Short summary of the target idea.', 'illustration': 'Detailed explanation of the target idea.', 'implementation_plan': 'A concrete implementation plan grounded in the parent node.', 'python_source': 'Runnable Python source code for the node.'},
        allow_tools=False,
        allowed_tools=[],
        parser_name='code',
        validator_name='python_source',
        writes=['python_source'],
        state=state,
        trace=trace,
        debug=debug,
    )
    return state


async def run_autoresearch_program(
    world: Any,
    client: KappaClient,
    *,
    memory: dict[str, Any] | None = None,
    trace: list[dict[str, Any]] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    world = adapt_world(world)
    state = dict(memory or {})
    current_node = state.get('current_node')
    implementer_run = state.get('implementer_run')
    new_node_ids = state.get('new_node_ids')
    pending_node_ids = state.get('pending_node_ids')
    planner_run = state.get('planner_run')
    snapshot = state.get('snapshot')
    tool_host = ToolHost(world, trace=trace, debug=debug)

    snapshot = capture_snapshot(world, state, {})
    state['snapshot'] = snapshot
    if debug: print(f"[host-step] capture_snapshot -> snapshot: {snapshot!r}")
    planner_run = await run_planner_role(client, tool_host, {'snapshot': snapshot}, trace=trace, debug=debug)
    state['planner_run'] = planner_run
    if debug: print(f"[role-step] planner -> planner_run: keys={sorted(planner_run)}")
    new_node_ids = materialize_proposals(world, state, {'proposals': planner_run['proposals']})
    state['new_node_ids'] = new_node_ids
    if debug: print(f"[host-step] materialize_proposals -> new_node_ids: {new_node_ids!r}")
    pending_node_ids = list_unimplemented_nodes(world, state, {})
    state['pending_node_ids'] = pending_node_ids
    if debug: print(f"[host-step] list_unimplemented_nodes -> pending_node_ids: {pending_node_ids!r}")
    for current_node_id in (pending_node_ids or []):
        state['current_node_id'] = current_node_id
        current_node = load_node_record(world, state, {'node_id': current_node_id})
        state['current_node'] = current_node
        if debug: print(f"[host-step] load_current_node -> current_node: {current_node!r}")
        implementer_run = await run_implementer_role(client, tool_host, {'snapshot': snapshot, 'node_id': current_node_id, 'parent_id': current_node['parent_id'], 'tldr': current_node['tldr'], 'illustration': current_node['illustration']}, trace=trace, debug=debug)
        state['implementer_run'] = implementer_run
        if debug: print(f"[role-step] implementer -> implementer_run: keys={sorted(implementer_run)}")
        write_generated_source(world, state, {'node_id': current_node_id, 'source': implementer_run['python_source']})
        if debug: print("[host-step] write_generated_source -> None")
    await run_pending_experiments(world, state, {})
    if debug: print("[host-step] run_pending_experiments -> None")

    return state

run_workflow = run_autoresearch_program


__all__ = [
    "PROGRAM_NAME",
    "PROGRAM_DESCRIPTION",
    "KappaClient",
    "run_workflow",
    "describe_program",
    "trace_to_json",
    "compare_traces",
    "adapt_world",
    "AutoresearchProgramWorldAdapter",
]
