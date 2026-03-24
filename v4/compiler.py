from __future__ import annotations

import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, Sequence

from dsl import (
    ForeachStepSpec,
    HostOpSpec,
    HostStepSpec,
    OutputContractSpec,
    PhaseSpec,
    ProgramSpec,
    Ref,
    RoleSpec,
    RoleStepSpec,
    ToolSpec,
    WorkflowStep,
)


class CompileError(ValueError):
    pass


ValueGetter = Callable[[dict[str, Any]], Any]
ValidatorFn = Callable[[Any], Any]
ParserFn = Callable[[str], Any]


@dataclass(frozen=True)
class CompiledContract:
    name: str
    instructions: str
    parser_name: str
    validator_name: str | None
    parser: ParserFn
    validator: ValidatorFn


@dataclass(frozen=True)
class CompiledPhase:
    name: str
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    allow_tools: bool
    allowed_tools: tuple[str, ...]
    system_prompt: str
    contract: CompiledContract
    field_descriptions: dict[str, str]


@dataclass(frozen=True)
class CompiledRole:
    name: str
    purpose: str
    phases: tuple[CompiledPhase, ...]
    required_inputs: tuple[str, ...]
    state_fields: tuple[str, ...]
    output_fields: tuple[str, ...]


@dataclass(frozen=True)
class CompiledHostStep:
    name: str
    op_name: str
    fn: Callable[[Any, dict[str, Any], dict[str, Any]], Any]
    arg_getter: ValueGetter
    arg_expr: Any
    save_as: str | None


@dataclass(frozen=True)
class CompiledRoleStep:
    name: str
    role_name: str
    role: CompiledRole
    binding_getter: ValueGetter
    binding_expr: Any
    save_as: str


@dataclass(frozen=True)
class CompiledForeachStep:
    name: str
    items_getter: ValueGetter
    items_expr: Any
    item_name: str
    body: tuple[CompiledStep, ...]


CompiledStep = CompiledHostStep | CompiledRoleStep | CompiledForeachStep


@dataclass(frozen=True)
class CompiledWorkflow:
    name: str
    steps: tuple[CompiledStep, ...]
    state_slots: tuple[str, ...]


@dataclass(frozen=True)
class CompiledProgram:
    name: str
    roles: dict[str, CompiledRole]
    host_ops: dict[str, HostOpSpec]
    tools: dict[str, ToolSpec]
    workflow: CompiledWorkflow

    def describe(self) -> str:
        lines = [f"CompiledProgram<{self.name}>", "", "Roles:"]
        for role in self.roles.values():
            lines.append(f"- {role.name}: inputs={list(role.required_inputs)} outputs={list(role.output_fields)}")
            for phase in role.phases:
                lines.append(
                    f"    phase {phase.name}: reads={list(phase.reads)} writes={list(phase.writes)} tools={list(phase.allowed_tools)}"
                )
        lines.extend(["", "Workflow slots:"])
        for slot in self.workflow.state_slots:
            lines.append(f"- {slot}")
        return "\n".join(lines)


# ---------------------------
# Parsers and validators
# ---------------------------


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


PARSERS: dict[str, ParserFn] = {
    "text": parse_text,
    "code": parse_code,
    "json_list": parse_json_list,
    "json_dict": parse_json_dict,
}


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


VALIDATORS: dict[str | None, ValidatorFn] = {
    None: identity,
    "nonempty_text": validate_nonempty_text,
    "proposal_list": validate_proposal_list,
    "python_source": validate_python_source,
}


# ---------------------------
# Compiler proper
# ---------------------------


def compile_program(spec: ProgramSpec) -> CompiledProgram:
    roles = {name: compile_role(role, spec.tools) for name, role in spec.roles.items()}
    workflow = compile_workflow(spec, roles)
    return CompiledProgram(
        name=spec.name,
        roles=roles,
        host_ops=spec.host_ops,
        tools=spec.tools,
        workflow=workflow,
    )


def compile_role(role: RoleSpec, tools: dict[str, ToolSpec]) -> CompiledRole:
    schema = role.state_schema.field_map()
    if not role.phases:
        raise CompileError(f"Role {role.name} has no phases.")

    compiled_phases: list[CompiledPhase] = []
    written_so_far = set(role.state_schema.input_names())
    output_fields: list[str] = []

    for phase in role.phases:
        unknown_reads = set(phase.reads) - set(schema)
        unknown_writes = set(phase.writes) - set(schema)
        if unknown_reads:
            raise CompileError(f"Role {role.name} phase {phase.name} reads unknown fields: {sorted(unknown_reads)}")
        if unknown_writes:
            raise CompileError(f"Role {role.name} phase {phase.name} writes unknown fields: {sorted(unknown_writes)}")
        unavailable_reads = set(phase.reads) - written_so_far
        if unavailable_reads:
            raise CompileError(
                f"Role {role.name} phase {phase.name} reads fields before they are available: {sorted(unavailable_reads)}"
            )
        if phase.allow_tools:
            for tool_name in phase.allowed_tools:
                if tool_name not in tools:
                    raise CompileError(f"Role {role.name} phase {phase.name} references unknown tool {tool_name!r}")
        contract = compile_contract(phase.contract)
        system_prompt = build_phase_system_prompt(role, phase, tools)
        compiled_phases.append(
            CompiledPhase(
                name=phase.name,
                purpose=phase.purpose,
                reads=tuple(phase.reads),
                writes=tuple(phase.writes),
                allow_tools=phase.allow_tools,
                allowed_tools=tuple(phase.allowed_tools),
                system_prompt=system_prompt,
                contract=contract,
                field_descriptions={name: schema[name].description for name in schema},
            )
        )
        written_so_far.update(phase.writes)
        output_fields.extend(name for name in phase.writes if schema[name].kind == "output")

    return CompiledRole(
        name=role.name,
        purpose=role.purpose,
        phases=tuple(compiled_phases),
        required_inputs=tuple(sorted(role.state_schema.input_names())),
        state_fields=tuple(schema.keys()),
        output_fields=tuple(dict.fromkeys(output_fields)),
    )


def compile_contract(contract: OutputContractSpec) -> CompiledContract:
    if contract.parser not in PARSERS:
        raise CompileError(f"Unknown parser kind: {contract.parser}")
    if contract.validator not in VALIDATORS:
        raise CompileError(f"Unknown validator: {contract.validator!r}")
    return CompiledContract(
        name=contract.name,
        instructions=contract.instructions.strip(),
        parser_name=contract.parser,
        validator_name=contract.validator,
        parser=PARSERS[contract.parser],
        validator=VALIDATORS[contract.validator],
    )


def build_phase_system_prompt(role: RoleSpec, phase: PhaseSpec, tools: dict[str, ToolSpec]) -> str:
    sections = [
        role.system_context.strip(),
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
        phase.contract.instructions.strip(),
    ]
    if phase.allow_tools:
        sections.extend(["", render_tool_protocol(phase.allowed_tools, tools)])
    else:
        sections.extend(["", "Tool use is disabled in this phase. Emit the output directly."])
    return "\n".join(section for section in sections if section != "")


def render_tool_protocol(allowed_tools: Sequence[str], tools: dict[str, ToolSpec]) -> str:
    lines = ["Available tools:"]
    for tool_name in allowed_tools:
        tool = tools[tool_name]
        lines.append(f"- {tool.signature}: {tool.description}")
    lines.extend(
        [
            "",
            "Tool protocol:",
            "To call a tool, emit exactly one call inside <tool></tool><stop>.",
            'Example: <tool>read_meta("2d28a7")</tool><stop>',
            "The host will execute the tool and append the result inside <result></result>.",
        ]
    )
    return "\n".join(lines)


def compile_workflow(spec: ProgramSpec, roles: dict[str, CompiledRole]) -> CompiledWorkflow:
    scope = set()
    steps = tuple(compile_step(step, spec, roles, scope) for step in spec.workflow.steps)
    return CompiledWorkflow(name=spec.workflow.name, steps=steps, state_slots=tuple(sorted(scope)))


def compile_step(
    step: WorkflowStep,
    spec: ProgramSpec,
    roles: dict[str, CompiledRole],
    scope: set[str],
) -> CompiledStep:
    if isinstance(step, HostStepSpec):
        if step.op not in spec.host_ops:
            raise CompileError(f"Unknown host op: {step.op}")
        validate_refs(step.args, scope, step.name)
        if step.save_as is not None:
            scope.add(step.save_as)
        return CompiledHostStep(
            name=step.name,
            op_name=step.op,
            fn=spec.host_ops[step.op].fn,
            arg_getter=compile_expr(step.args),
            arg_expr=freeze_expr(step.args),
            save_as=step.save_as,
        )

    if isinstance(step, RoleStepSpec):
        if step.role not in roles:
            raise CompileError(f"Unknown role: {step.role}")
        role = roles[step.role]
        validate_refs(step.bindings, scope, step.name)
        binding_names = set(step.bindings)
        missing_inputs = set(role.required_inputs) - binding_names
        unknown_bindings = binding_names - set(role.state_fields)
        if missing_inputs:
            raise CompileError(
                f"Role step {step.name} is missing required bindings for role {step.role}: {sorted(missing_inputs)}"
            )
        if unknown_bindings:
            raise CompileError(
                f"Role step {step.name} passes unknown fields to role {step.role}: {sorted(unknown_bindings)}"
            )
        scope.add(step.save_as)
        return CompiledRoleStep(
            name=step.name,
            role_name=step.role,
            role=role,
            binding_getter=compile_expr(step.bindings),
            binding_expr=freeze_expr(step.bindings),
            save_as=step.save_as,
        )

    if isinstance(step, ForeachStepSpec):
        validate_refs(step.items, scope, step.name)
        nested_scope = set(scope)
        nested_scope.add(step.item_name)
        body = tuple(compile_step(child, spec, roles, nested_scope) for child in step.body)
        scope.update(slot for slot in nested_scope if slot != step.item_name)
        return CompiledForeachStep(
            name=step.name,
            items_getter=compile_expr(step.items),
            items_expr=freeze_expr(step.items),
            item_name=step.item_name,
            body=body,
        )

    raise CompileError(f"Unknown workflow step: {step!r}")


def validate_refs(value: Any, scope: set[str], step_name: str) -> None:
    for ref_path in collect_refs(value):
        root = ref_path.split(".")[0]
        if root not in scope:
            raise CompileError(f"Step {step_name} references unknown workflow slot {root!r} via {ref_path!r}")


def collect_refs(value: Any) -> list[str]:
    if isinstance(value, Ref):
        return [value.path]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(collect_refs(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(collect_refs(item))
        return out
    return []


def compile_expr(expr: Any) -> ValueGetter:
    if isinstance(expr, Ref):
        path = expr.path

        def _get_ref(state: dict[str, Any], *, _path: str = path) -> Any:
            return resolve_path(state, _path)

        return _get_ref

    if isinstance(expr, dict):
        compiled = {key: compile_expr(value) for key, value in expr.items()}

        def _get_dict(state: dict[str, Any], *, _compiled: dict[str, ValueGetter] = compiled) -> dict[str, Any]:
            return {key: getter(state) for key, getter in _compiled.items()}

        return _get_dict

    if isinstance(expr, list):
        compiled_list = [compile_expr(item) for item in expr]

        def _get_list(state: dict[str, Any], *, _compiled_list: list[ValueGetter] = compiled_list) -> list[Any]:
            return [getter(state) for getter in _compiled_list]

        return _get_list

    return lambda state, _value=expr: _value


def freeze_expr(expr: Any) -> Any:
    if isinstance(expr, Ref):
        return {"$ref": expr.path}
    if isinstance(expr, dict):
        return {key: freeze_expr(value) for key, value in expr.items()}
    if isinstance(expr, (list, tuple)):
        return [freeze_expr(item) for item in expr]
    return expr


def resolve_path(state: dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


# ---------------------------
# Kernel emission
# ---------------------------

# ---------------------------
# Kernel emission
# ---------------------------


def emit_python_kernel(program: CompiledProgram, module_name: str = "compiled_kernel") -> str:
    del module_name  # reserved for future naming hooks

    sections = [
        "from __future__ import annotations\n\n",
        f'"""Task-family residual kernel emitted for {program.name}.\n\n'
        "This kernel no longer carries the generic ResidualRuntimeSession or a generic workflow\n"
        "dispatcher. The workflow is emitted as a dedicated program function, and each role is\n"
        "emitted as a dedicated runner with explicit phase order.\n"
        '"""\n\n',
        "import asyncio\nimport json\nimport re\nfrom typing import Any, Callable\n\n",
        "import requests\n\n",
        f"PROGRAM_NAME = {program.name!r}\n",
        f"PROGRAM_DESCRIPTION = {program.describe()!r}\n\n",
        render_host_functions(program),
        "\n\n\n",
        STATIC_KERNEL_HELPERS,
        "\n\n\n",
        render_role_functions(program),
        "\n\n\n",
        render_workflow_function(program),
        "\n\n\n",
        KERNEL_FOOTER,
    ]
    return "".join(sections)


STATIC_KERNEL_HELPERS = """def parse_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError("Expected non-empty text output.")
    return text


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\\s*(.*?)\\s*```$", re.DOTALL)


def parse_code(raw: str) -> str:
    text = raw.strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    if not text:
        raise ValueError("Expected non-empty source code.")
    return text


_JSON_LIST_RE = re.compile(r"(\\[.*\\])", re.DOTALL)
_JSON_DICT_RE = re.compile(r"(\\{.*\\})", re.DOTALL)


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


_METRIC_HINT_RE = re.compile(r"Metric\\s*:")


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
    TOOL_CALL_RE = re.compile(r"<tool>\\s*(\\w+)\\([\\"\\'](.*?)[\\"\\']\\)\\s*</tool>")

    def __init__(self, world: Any, debug: bool = False):
        self.world = world
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

    def execute(self, tool_name: str, argument: str) -> str:
        if tool_name not in self.handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.handlers[tool_name](argument)


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
            lines.append(f"\\n[{field_name}] {description}\\n{_serialize(value)}")
        else:
            lines.append(f"\\n[{field_name}]\\n{_serialize(value)}")
    lines.append("\\nProduce the required phase output now.")
    return "\\n".join(lines)


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
    debug: bool = False,
) -> None:
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _render_user_prompt(reads, field_descriptions, state)},
    ]
    tool_block_re = re.compile(r"(.*?<tool>\\s*\\w+\\([\\"\\'].*?[\\"\\']\\)\\s*</tool><stop>)", re.DOTALL)

    while True:
        raw = client.complete(history, stop=["<stop>"])
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
            result = tool_host.execute(invocation.name, invocation.argument)
            history.append({"role": "user", "content": f"<result>\\n{result}\\n</result>"})
            if debug:
                print(f"\\n[tool] {invocation.name}({invocation.argument!r})\\n{result}\\n", flush=True)
            continue

        parsed = PARSERS[parser_name](raw)
        validated = VALIDATORS[validator_name](parsed)
        _write_phase_output(state, writes, validated)
        if debug:
            print(f"\\n[phase {phase_name} output]\\n{validated!r}\\n", flush=True)
        return


async def _execute_phase(
    client: KappaClient,
    tool_host: ToolHost,
    *,
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
    debug: bool = False,
) -> None:
    await asyncio.to_thread(
        _execute_phase_sync,
        client,
        tool_host,
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
        debug=debug,
    )


def describe_program() -> str:
    return PROGRAM_DESCRIPTION
"""


KERNEL_FOOTER = """__all__ = [
    "PROGRAM_NAME",
    "PROGRAM_DESCRIPTION",
    "KappaClient",
    "run_workflow",
    "describe_program",
]
"""


def render_role_functions(program: CompiledProgram) -> str:
    blocks: list[str] = []
    for role in program.roles.values():
        blocks.append(render_role_function(role))
    return "\n\n\n".join(blocks)


def render_role_function(role: CompiledRole) -> str:
    lines = [
        f"async def run_{role.name}_role(client: KappaClient, tool_host: ToolHost, initial_state: dict[str, Any], debug: bool = False) -> dict[str, Any]:",
        "    state = dict(initial_state)",
    ]
    for phase in role.phases:
        lines.extend(
            [
                "    await _execute_phase(",
                "        client,",
                "        tool_host,",
                f"        phase_name={phase.name!r},",
                f"        system_prompt={phase.system_prompt!r},",
                f"        reads={list(phase.reads)!r},",
                f"        field_descriptions={dict(phase.field_descriptions)!r},",
                f"        allow_tools={phase.allow_tools!r},",
                f"        allowed_tools={list(phase.allowed_tools)!r},",
                f"        parser_name={phase.contract.parser_name!r},",
                f"        validator_name={phase.contract.validator_name!r},",
                f"        writes={list(phase.writes)!r},",
                "        state=state,",
                "        debug=debug,",
                "    )",
            ]
        )
    lines.append("    return state")
    return "\n".join(lines)


def render_workflow_function(program: CompiledProgram) -> str:
    workflow_fn_name = f"run_{program.name}"
    lines = [
        f"async def {workflow_fn_name}(world: Any, client: KappaClient, *, memory: dict[str, Any] | None = None, debug: bool = False) -> dict[str, Any]:",
        "    state = dict(memory or {})",
    ]
    for slot in program.workflow.state_slots:
        lines.append(f"    {slot} = state.get({slot!r})")
    lines.extend(
        [
            "    tool_host = ToolHost(world, debug=debug)",
            "",
        ]
    )
    lines.extend(render_step_lines(program.workflow.steps, indent="    "))
    lines.extend(
        [
            "",
            "    return state",
            "",
            f"run_workflow = {workflow_fn_name}",
        ]
    )
    return "\n".join(lines)


def render_step_lines(steps: Sequence[CompiledStep], indent: str) -> list[str]:
    lines: list[str] = []
    for step in steps:
        lines.extend(render_step(step, indent))
    return lines


def render_step(step: CompiledStep, indent: str) -> list[str]:
    if isinstance(step, CompiledHostStep):
        args_expr = expr_to_python(step.arg_expr)
        fn_name = step.fn.__name__
        lines: list[str] = []
        if step.save_as is not None:
            if inspect.iscoroutinefunction(step.fn):
                lines.append(f"{indent}{step.save_as} = await {fn_name}(world, state, {args_expr})")
            else:
                lines.append(f"{indent}{step.save_as} = {fn_name}(world, state, {args_expr})")
            lines.append(f"{indent}state[{step.save_as!r}] = {step.save_as}")
            lines.append(f"{indent}if debug: print(f\"[host-step] {step.name} -> {step.save_as}: {{{step.save_as}!r}}\")")
        else:
            if inspect.iscoroutinefunction(step.fn):
                lines.append(f"{indent}await {fn_name}(world, state, {args_expr})")
            else:
                lines.append(f"{indent}{fn_name}(world, state, {args_expr})")
            lines.append(f"{indent}if debug: print(\"[host-step] {step.name} -> None\")")
        return lines

    if isinstance(step, CompiledRoleStep):
        binding_expr = expr_to_python(step.binding_expr)
        lines = [
            f"{indent}{step.save_as} = await run_{step.role_name}_role(client, tool_host, {binding_expr}, debug=debug)",
            f"{indent}state[{step.save_as!r}] = {step.save_as}",
            f"{indent}if debug: print(f\"[role-step] {step.name} -> {step.save_as}: keys={{sorted({step.save_as})}}\")",
        ]
        return lines

    if isinstance(step, CompiledForeachStep):
        items_expr = expr_to_python(step.items_expr)
        lines = [
            f"{indent}for {step.item_name} in ({items_expr} or []):",
            f"{indent}    state[{step.item_name!r}] = {step.item_name}",
        ]
        lines.extend(render_step_lines(step.body, indent + "    "))
        return lines

    raise TypeError(f"Unknown compiled step type: {type(step)!r}")


def expr_to_python(expr: Any) -> str:
    if isinstance(expr, dict) and set(expr) == {"$ref"}:
        return path_to_python(expr["$ref"])
    if isinstance(expr, dict):
        parts = [f"{key!r}: {expr_to_python(value)}" for key, value in expr.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(expr, list):
        return "[" + ", ".join(expr_to_python(item) for item in expr) + "]"
    return repr(expr)


def path_to_python(path: str) -> str:
    parts = path.split(".")
    expr = parts[0]
    for part in parts[1:]:
        expr += f"[{part!r}]"
    return expr


def render_host_functions(program: CompiledProgram) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for step in iter_steps(program.workflow.steps):
        if isinstance(step, CompiledHostStep) and step.op_name not in seen:
            source = textwrap.dedent(inspect.getsource(step.fn)).rstrip()
            lines.append(source)
            seen.add(step.op_name)
    return "\n\n\n".join(lines)


def iter_steps(steps: Sequence[CompiledStep]) -> list[CompiledStep]:
    out: list[CompiledStep] = []
    for step in steps:
        out.append(step)
        if isinstance(step, CompiledForeachStep):
            out.extend(iter_steps(step.body))
    return out


def write_python_kernel(program: CompiledProgram, path: str) -> str:
    source = emit_python_kernel(program)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return source
