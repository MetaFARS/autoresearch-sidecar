from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from dsl import (
    FieldSpec,
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
    save_as: str | None


@dataclass(frozen=True)
class CompiledRoleStep:
    name: str
    role_name: str
    role: CompiledRole
    binding_getter: ValueGetter
    save_as: str


@dataclass(frozen=True)
class CompiledForeachStep:
    name: str
    items_getter: ValueGetter
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
        out = []
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


def resolve_path(state: dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current
