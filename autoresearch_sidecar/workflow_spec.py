from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


StateDict = dict[str, Any]
Parser = Callable[[str], Any]
Validator = Callable[[Any], Any]


@dataclass(frozen=True)
class OutputSpec:
    instructions: str
    parser: Parser
    validator: Validator | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    purpose: str
    reads: tuple[str, ...]
    writes: str
    instructions: str
    output: OutputSpec
    allow_tools: bool = False
    allowed_tools: tuple[str, ...] = ()
    max_tool_rounds: int = 8


@dataclass(frozen=True)
class RoleSpec:
    name: str
    purpose: str
    system_context: str
    required_inputs: tuple[str, ...]
    field_descriptions: Mapping[str, str]
    phases: tuple[PhaseSpec, ...]
    invariants: tuple[str, ...] = ()
