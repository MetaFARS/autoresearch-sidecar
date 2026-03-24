from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence


ParserKind = Literal["text", "json_list", "json_dict", "code"]
FieldKind = Literal["input", "working", "output"]


@dataclass(frozen=True)
class Ref:
    """Reference to a workflow memory path such as 'snapshot' or 'planner_run.proposals'."""

    path: str


def ref(path: str) -> Ref:
    return Ref(path)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    kind: FieldKind = "working"


@dataclass(frozen=True)
class StateSchema:
    fields: Sequence[FieldSpec]

    def field_map(self) -> dict[str, FieldSpec]:
        return {field.name: field for field in self.fields}

    def render(self) -> str:
        lines = []
        for field in self.fields:
            lines.append(f"- {field.name} ({field.kind}): {field.description}")
        return "\n".join(lines)

    def names(self) -> set[str]:
        return set(self.field_map())

    def input_names(self) -> set[str]:
        return {field.name for field in self.fields if field.kind == "input"}


@dataclass(frozen=True)
class OutputContractSpec:
    name: str
    parser: ParserKind
    instructions: str
    validator: str | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    purpose: str
    reads: Sequence[str]
    writes: Sequence[str]
    instructions: str
    contract: OutputContractSpec
    allow_tools: bool = False
    allowed_tools: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class RoleSpec:
    name: str
    purpose: str
    system_context: str
    state_schema: StateSchema
    invariants: Sequence[str]
    phases: Sequence[PhaseSpec]


HostFn = Callable[[Any, dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True)
class HostOpSpec:
    name: str
    doc: str
    fn: HostFn


@dataclass(frozen=True)
class ToolSpec:
    name: str
    signature: str
    description: str


@dataclass(frozen=True)
class HostStepSpec:
    name: str
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    save_as: str | None = None


@dataclass(frozen=True)
class RoleStepSpec:
    name: str
    role: str
    bindings: dict[str, Any]
    save_as: str


@dataclass(frozen=True)
class ForeachStepSpec:
    name: str
    items: Ref
    item_name: str
    body: Sequence["WorkflowStep"]


WorkflowStep = HostStepSpec | RoleStepSpec | ForeachStepSpec


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    steps: Sequence[WorkflowStep]


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    roles: dict[str, RoleSpec]
    host_ops: dict[str, HostOpSpec]
    tools: dict[str, ToolSpec]
    workflow: WorkflowSpec
