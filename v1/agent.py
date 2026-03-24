from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import requests


JsonDict = Dict[str, Any]
StateDict = Dict[str, Any]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str


@dataclass(frozen=True)
class StateSchema:
    inputs: tuple[FieldSpec, ...]
    working: tuple[FieldSpec, ...]
    outputs: tuple[FieldSpec, ...]

    def render(self) -> str:
        sections = []
        for title, fields in (
            ("Inputs", self.inputs),
            ("Working state", self.working),
            ("Outputs", self.outputs),
        ):
            sections.append(f"{title}:")
            for field in fields:
                sections.append(f"- {field.name}: {field.description}")
        return "\n".join(sections)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    filename: str
    arg_name: str = "node_id"

    def render_signature(self) -> str:
        return f"{self.name}({self.arg_name}: str) -> str: {self.description}"


ParserFn = Callable[[str], Any]
ValidatorFn = Callable[[Any], None]


@dataclass(frozen=True)
class OutputContract:
    name: str
    instructions: str
    parser: ParserFn
    validator: Optional[ValidatorFn] = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    purpose: str
    reads: tuple[str, ...]
    writes: str
    instructions: str
    allow_tools: bool
    output_contract: OutputContract
    allowed_tools: tuple[str, ...] = ()
    max_tool_rounds: int = 8


@dataclass(frozen=True)
class RoleSpec:
    name: str
    purpose: str
    state_schema: StateSchema
    phases: tuple[PhaseSpec, ...]
    invariants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    argument: str
    raw_text: str


class ChatCompletionClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[JsonDict], stop: Optional[list[str]] = None) -> str:
        payload: JsonDict = {"model": self.model, "messages": messages}
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "choices" not in data:
            raise RuntimeError(f"API error: {data}")
        return data["choices"][0]["message"]["content"]


class NamespaceToolHost:
    def __init__(self, namespace_dir: str | Path = "namespace") -> None:
        self.namespace_dir = Path(namespace_dir)
        self.tool_specs: dict[str, ToolSpec] = {
            "read_meta": ToolSpec(
                name="read_meta",
                description="Retrieves the metadata for an experiment node.",
                filename="meta.json",
            ),
            "read_code": ToolSpec(
                name="read_code",
                description="Retrieves the implementation source code for an experiment node.",
                filename="main.py",
            ),
            "read_stdout": ToolSpec(
                name="read_stdout",
                description="Retrieves the execution logs for an experiment node.",
                filename="stdout.log",
            ),
            "read_stderr": ToolSpec(
                name="read_stderr",
                description="Retrieves the error logs for an experiment node.",
                filename="stderr.log",
            ),
        }

    def render_protocol(self, allowed_tool_names: Iterable[str]) -> str:
        tool_lines = []
        for index, name in enumerate(allowed_tool_names, start=1):
            spec = self.tool_specs[name]
            tool_lines.append(f"{index}. {spec.render_signature()}")

        protocol = [
            "Allowed tools for this phase:",
            *tool_lines,
            "",
            "Tool protocol:",
            "- To call a tool, emit exactly one call wrapped as <tool>...</tool><stop>.",
            "- Only one tool may be called at a time.",
            "- After tool execution, the system will inject the result as <result>...</result>.",
            "- When investigation is complete, stop calling tools and emit the phase output directly.",
        ]
        return "\n".join(protocol)

    def execute(self, tool_name: str, node_id: str, allowed_tool_names: Iterable[str]) -> str:
        allowed = set(allowed_tool_names)
        if tool_name not in allowed:
            return f"Error: tool '{tool_name}' is not allowed in this phase."

        spec = self.tool_specs.get(tool_name)
        if spec is None:
            return f"Error: unknown tool '{tool_name}'."

        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            return f"Error: invalid node id '{node_id}'."

        path = self.namespace_dir / node_id / spec.filename
        if not path.exists():
            return f"Error: {tool_name} failed because {path} does not exist."
        return path.read_text()


class RoleInterpreter:
    TOOL_PATTERN = re.compile(
        r"<tool>\s*([A-Za-z_][A-Za-z0-9_]*)\(\s*[\"'](.*?)[\"']\s*\)\s*</tool>\s*<stop>",
        re.DOTALL,
    )

    def __init__(self, client: ChatCompletionClient, tool_host: NamespaceToolHost, debug_mode: bool = False) -> None:
        self.client = client
        self.tool_host = tool_host
        self.debug_mode = debug_mode

    def run(self, role: RoleSpec, initial_state: StateDict) -> StateDict:
        self._validate_initial_state(role, initial_state)
        state = dict(initial_state)

        for phase in role.phases:
            raw_output = self._run_phase(role, phase, state)
            parsed_output = phase.output_contract.parser(raw_output)
            if phase.output_contract.validator is not None:
                phase.output_contract.validator(parsed_output)
            state[phase.writes] = parsed_output

        return state

    def _validate_initial_state(self, role: RoleSpec, state: StateDict) -> None:
        required_inputs = {field.name for field in role.state_schema.inputs}
        missing = sorted(name for name in required_inputs if name not in state)
        if missing:
            raise ValueError(f"Missing required state fields for role {role.name}: {missing}")

    def _run_phase(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        history: list[JsonDict] = [
            {"role": "system", "content": self._render_system_prompt(role, phase)},
            {"role": "user", "content": self._render_user_prompt(role, phase, state)},
        ]

        tool_rounds = 0
        while True:
            raw = self.client.complete(history, stop=["<stop>"] if phase.allow_tools else None)
            normalized = raw if (not phase.allow_tools or "<stop>" in raw or "</tool>" not in raw) else f"{raw}<stop>"
            invocation = self._parse_tool_invocation(normalized) if phase.allow_tools else None

            if invocation is None:
                history.append({"role": "assistant", "content": raw})
                if self.debug_mode:
                    print(f"\n[Phase {phase.name} final output]\n{raw}\n", flush=True)
                return raw

            if tool_rounds >= phase.max_tool_rounds:
                raise RuntimeError(
                    f"Phase '{phase.name}' exceeded its tool budget ({phase.max_tool_rounds})."
                )

            history.append({"role": "assistant", "content": invocation.raw_text})
            tool_result = self.tool_host.execute(
                invocation.name,
                invocation.argument,
                phase.allowed_tools,
            )
            history.append({"role": "user", "content": f"<result>\n{tool_result}\n</result>"})
            tool_rounds += 1

            if self.debug_mode:
                print(
                    f"\n[Phase {phase.name} tool call] {invocation.name}({invocation.argument!r})\n"
                    f"[Tool result]\n{tool_result}\n",
                    flush=True,
                )

    def _parse_tool_invocation(self, raw: str) -> Optional[ToolInvocation]:
        match = self.TOOL_PATTERN.search(raw)
        if not match:
            return None
        return ToolInvocation(
            name=match.group(1),
            argument=match.group(2),
            raw_text=match.group(0).strip(),
        )

    def _render_system_prompt(self, role: RoleSpec, phase: PhaseSpec) -> str:
        sections = [
            RESEARCH_SYSTEM_CONTEXT.strip(),
            f"Role name: {role.name}",
            f"Role purpose: {role.purpose}",
            "",
            "State schema:",
            role.state_schema.render(),
            "",
            f"Current phase: {phase.name}",
            f"Phase purpose: {phase.purpose}",
            "",
            "Phase instructions:",
            phase.instructions.strip(),
            "",
            "Role invariants:",
            *[f"- {item}" for item in role.invariants],
            "",
            "Output contract:",
            phase.output_contract.instructions.strip(),
        ]

        if phase.allow_tools:
            sections.extend(["", self.tool_host.render_protocol(phase.allowed_tools)])
        else:
            sections.extend(["", "Tool use is disabled in this phase. Emit the output directly."])

        return "\n".join(section for section in sections if section != "")

    def _render_user_prompt(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        lines = ["Phase state bindings:"]
        for field_name in phase.reads:
            value = state.get(field_name)
            lines.append(f"\n[{field_name}]\n{self._serialize(value)}")
        lines.append("\nProduce the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False)


def parse_research_notes(text: str) -> str:
    notes = text.strip()
    if not notes:
        raise ValueError("Investigation phase returned an empty note.")
    return notes


def parse_proposals(text: str) -> list[JsonDict]:
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find a JSON list in model output: {text!r}")
    proposals = json.loads(match.group(1))
    if not isinstance(proposals, list):
        raise ValueError("Proposal output must be a JSON list.")

    required = {"parent_id", "tldr", "illustration"}
    for index, item in enumerate(proposals):
        if not isinstance(item, dict):
            raise ValueError(f"Proposal #{index} is not an object.")
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Proposal #{index} is missing keys: {sorted(missing)}")
        extras = set(item.keys()) - required
        if extras:
            raise ValueError(f"Proposal #{index} has unexpected keys: {sorted(extras)}")
        for key in required:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Proposal #{index} field '{key}' must be a non-empty string.")
    return proposals


def validate_nonempty_list(value: list[Any]) -> None:
    if not value:
        raise ValueError("The proposal list is empty.")


def parse_python_source(text: str) -> str:
    source = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", source, re.DOTALL)
    if fenced:
        source = fenced.group(1).strip()
    if not source:
        raise ValueError("Generated Python source is empty.")
    compile(source, "<generated_main.py>", "exec")
    return source


RESEARCH_SYSTEM_CONTEXT = """
The research system manages a hierarchical tree of iterative experiments.
Each experiment is encapsulated as an Idea object with core metadata:
- node_id (str): Unique hash for the experiment.
- parent_id (str | None): ID of the base experiment this extends.
- illustration (str): Technical explanation of the hypothesis and design.
- tldr (str): Concise summary of the proposal.
- metric (float): The primary quantitative result. Larger is always better.
- exit_code (int | None): 0 for successful execution, non-zero for crashes.
- status (IdeaStatus): Lifecycle state: pending, running, success, or failed.
"""


IDEA_PROPOSER_SPEC = RoleSpec(
    name="idea_proposer",
    purpose="Inspect the experiment tree, identify promising branches or failed branches worth debugging, and propose concrete next experiments.",
    state_schema=StateSchema(
        inputs=(
            FieldSpec("snapshot", "A textual tree snapshot of all current experiment ideas and their metrics."),
        ),
        working=(
            FieldSpec("research_notes", "The explicit investigation summary produced after tool-assisted inspection."),
        ),
        outputs=(
            FieldSpec("proposals", "A JSON list of v0 experiment proposals."),
        ),
    ),
    invariants=(
        "Treat the experiment tree as the source of truth; do not invent missing node data.",
        "Tools may only be used during investigation phases, one call at a time.",
        "Prefer improving successful leaves or debugging failed leaves.",
        "Final proposals must use the exact schema [{\"parent_id\": \"...\", \"tldr\": \"...\", \"illustration\": \"...\"}].",
    ),
    phases=(
        PhaseSpec(
            name="investigate_tree",
            purpose="Use tools to inspect promising or problematic nodes and write down evidence-backed research notes.",
            reads=("snapshot",),
            writes="research_notes",
            allow_tools=True,
            instructions=(
                "Inspect the research tree and, if useful, call tools one-by-one to read metadata, code, stdout, or stderr for specific nodes. "
                "Your goal is to identify concrete improvement opportunities, not to output proposals yet. "
                "Finish this phase by writing concise research notes that mention strong parents, weak spots, and likely next moves."
            ),
            allowed_tools=("read_meta", "read_code", "read_stdout", "read_stderr"),
            output_contract=OutputContract(
                name="research_notes",
                instructions=(
                    "Output only a concise investigation note in plain text. No JSON, no markdown code fences, and no conversational preamble."
                ),
                parser=parse_research_notes,
            ),
        ),
        PhaseSpec(
            name="emit_proposals",
            purpose="Turn the explicit research notes into a structured set of candidate experiments.",
            reads=("snapshot", "research_notes"),
            writes="proposals",
            allow_tools=False,
            instructions=(
                "Use the snapshot and research notes to propose v0 experiments that maximize the metric. "
                "Each proposal must choose a concrete parent_id and describe the modification succinctly but technically."
            ),
            output_contract=OutputContract(
                name="proposal_list",
                instructions=(
                    "Output only a valid JSON list with exact objects of the form "
                    "[{\"parent_id\": \"...\", \"tldr\": \"...\", \"illustration\": \"...\"}]. "
                    "No markdown, no prose, and no extra keys."
                ),
                parser=parse_proposals,
                validator=validate_nonempty_list,
            ),
        ),
    ),
)


IMPLEMENTER_SPEC = RoleSpec(
    name="idea_implementer",
    purpose="Inspect the chosen idea and its parent artifacts, derive an implementation plan, and emit runnable Python code.",
    state_schema=StateSchema(
        inputs=(
            FieldSpec("snapshot", "A textual tree snapshot of all current experiment ideas and their metrics."),
            FieldSpec("node_id", "The node being implemented."),
            FieldSpec("parent_id", "The parent node to extend or debug."),
            FieldSpec("tldr", "The short summary of the target idea."),
            FieldSpec("illustration", "The technical description of the target idea."),
        ),
        working=(
            FieldSpec("implementation_plan", "A concrete plan distilled from parent inspection and the target idea."),
        ),
        outputs=(
            FieldSpec("python_source", "The final runnable Python program for main.py."),
        ),
    ),
    invariants=(
        "Parent code and logs are the baseline source of truth for implementation details.",
        "The generated program must be runnable Python and must print the final result on one line as: Metric: <value>.",
        "Keep tool use in the inspection phase only; code emission must be direct.",
    ),
    phases=(
        PhaseSpec(
            name="inspect_parent",
            purpose="Inspect the relevant parent artifacts and derive a concrete implementation plan.",
            reads=("snapshot", "node_id", "parent_id", "tldr", "illustration"),
            writes="implementation_plan",
            allow_tools=True,
            instructions=(
                "Use tools to inspect the chosen parent node's metadata, code, stdout, or stderr as needed. "
                "Determine whether this idea is an improvement over a successful parent or a repair for a failed parent. "
                "End by writing a crisp implementation plan with the exact changes to make."
            ),
            allowed_tools=("read_meta", "read_code", "read_stdout", "read_stderr"),
            output_contract=OutputContract(
                name="implementation_plan",
                instructions=(
                    "Output only a concise implementation plan in plain text. No markdown code fences and no conversational filler."
                ),
                parser=parse_research_notes,
            ),
        ),
        PhaseSpec(
            name="emit_code",
            purpose="Generate the final main.py based on the explicit plan.",
            reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "implementation_plan"),
            writes="python_source",
            allow_tools=False,
            instructions=(
                "Write the full runnable Python program implementing the target idea. "
                "Honor the implementation plan, keep the code self-contained, and ensure the final metric line is printed exactly as 'Metric: <value>'."
            ),
            output_contract=OutputContract(
                name="python_source",
                instructions=(
                    "Output only the complete Python source for main.py. No markdown fences and no prose."
                ),
                parser=parse_python_source,
            ),
        ),
    ),
)


class Agent:
    """Drop-in replacement for the original specialized agent.

    Public methods remain the same as the original version:
    - generate_ideas(snapshot) -> list[dict[str, str]]
    - implement(snapshot, node_id) -> None

    The difference is architectural: role semantics are now carried by RoleSpec
    objects, while this class delegates execution to a thin RoleInterpreter.
    """

    def __init__(self, base_url: str, api_key: str, model: str, debug_mode: bool = False) -> None:
        client = ChatCompletionClient(base_url=base_url, api_key=api_key, model=model)
        tool_host = NamespaceToolHost(namespace_dir="namespace")
        self.interpreter = RoleInterpreter(client=client, tool_host=tool_host, debug_mode=debug_mode)

    def generate_ideas(self, snapshot: str) -> list[JsonDict]:
        final_state = self.interpreter.run(
            IDEA_PROPOSER_SPEC,
            {
                "snapshot": snapshot,
            },
        )
        return final_state["proposals"]

    def implement(self, snapshot: str, node_id: str) -> None:
        meta_path = Path("namespace") / node_id / "meta.json"
        idea_data = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        final_state = self.interpreter.run(
            IMPLEMENTER_SPEC,
            {
                "snapshot": snapshot,
                "node_id": node_id,
                "parent_id": idea_data.get("parent_id") or "",
                "tldr": idea_data.get("tldr", ""),
                "illustration": idea_data.get("illustration", ""),
            },
        )

        dest_dir = Path("namespace") / node_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "main.py").write_text(final_state["python_source"])
