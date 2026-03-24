from __future__ import annotations

from dataclasses import dataclass

from .work_context import WorkContext
from .workflow_spec import OutputSpec, PhaseSpec, RoleSpec
from .orchestrator_validators import (
    parse_json_list,
    parse_text,
    validate_nonempty_text,
    validate_proposals,
    validate_python_source,
)


@dataclass(frozen=True)
class OrchestratorConfig:
    required_parent_anchors: tuple[str, ...]
    forbidden_new_patterns: tuple[str, ...]


def build_planner_role(work_context: WorkContext, allowed_tools: tuple[str, ...]) -> RoleSpec:
    return RoleSpec(
        name="planner",
        purpose="Propose the next batch of train.py experiments.",
        system_context=f"{work_context.experiment_context}\n\n{work_context.target_contract}",
        required_inputs=("snapshot",),
        field_descriptions={
            "snapshot": "Serialized experiment tree snapshot.",
            "experiment_notes": "Concise findings from tool-assisted inspection.",
            "proposals": "A JSON list of proposal objects with parent_id, tldr, and illustration.",
        },
        invariants=(
            "Investigation and proposal emission are separate phases.",
            "Use tools to ground proposals in existing code or results.",
            "Prefer improving strong successful nodes or debugging informative failures.",
            "Do not emit markdown fences or conversational filler in final outputs.",
        ),
        phases=(
            PhaseSpec(
                name="investigate",
                purpose="Inspect promising nodes and collect explicit experiment notes.",
                reads=("snapshot",),
                writes="experiment_notes",
                instructions="""
Inspect successful nodes and informative failures before proposing new work.
Use tools one call at a time.
Focus on ideas that could improve val_bpb without violating the train.py-only contract.
Output concise experiment notes only. Do not emit proposals yet.
""".strip(),
                output=OutputSpec(
                    instructions="Return only concise experiment notes. No JSON. No markdown fences.",
                    parser=parse_text,
                    validator=validate_nonempty_text,
                ),
                allow_tools=True,
                allowed_tools=allowed_tools,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_proposals",
                purpose="Turn the experiment notes into concrete proposals.",
                reads=("snapshot", "experiment_notes"),
                writes="proposals",
                instructions="""
Return 1 to 3 concrete proposals.
Each proposal must include parent_id, tldr, and illustration.
Choose parent_id values from the snapshot when possible.
Output only the JSON list.
""".strip(),
                output=OutputSpec(
                    instructions='Return only valid JSON of the form [{"parent_id": "...", "tldr": "...", "illustration": "..."}].',
                    parser=parse_json_list,
                    validator=validate_proposals,
                ),
            ),
        ),
    )


def build_implementer_role(work_context: WorkContext, allowed_tools: tuple[str, ...]) -> RoleSpec:
    return RoleSpec(
        name="implementer",
        purpose="Implement one proposal as a runnable train.py variant.",
        system_context=f"{work_context.experiment_context}\n\n{work_context.target_contract}",
        required_inputs=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source"),
        field_descriptions={
            "snapshot": "Serialized experiment tree snapshot.",
            "node_id": "Node being implemented.",
            "parent_id": "Parent node whose train.py should be used as the baseline.",
            "tldr": "Short summary of the proposal.",
            "illustration": "Detailed rationale for the proposal.",
            "parent_source": "Exact train.py source from the parent node. Treat this as the file you are editing.",
            "implementation_plan": "A concise plan describing the concrete train.py edits.",
            "train_py": "Full runnable Python source for train.py.",
        },
        invariants=(
            "Inspect the parent node before writing code.",
            "Keep the implementation grounded in the chosen parent train.py.",
            "Edit the parent file instead of replacing the task with a different training script.",
            "Only emit the final train.py source in the final phase.",
            "Preserve the fixed summary output contract used by the backend.",
        ),
        phases=(
            PhaseSpec(
                name="inspect_parent",
                purpose="Study the parent implementation and prepare an edit plan.",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source"),
                writes="implementation_plan",
                instructions="""
Inspect the parent node's code and logs as needed.
Work out the smallest coherent implementation that matches the proposal.
Prefer small, targeted edits over unrelated rewrites.
The parent_source shown in state is the file you must edit.
Preserve the parent's prepare.py integration, data loading path, tokenizer usage, and evaluation harness unless the proposal explicitly changes them.
Do not introduce alternate dataset conventions or config-file conventions that are not already present in parent_source.
Output only the implementation plan, not code.
""".strip(),
                output=OutputSpec(
                    instructions="Return only a concise implementation plan. No code fences.",
                    parser=parse_text,
                    validator=validate_nonempty_text,
                ),
                allow_tools=True,
                allowed_tools=allowed_tools,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_train_py",
                purpose="Emit the full runnable train.py source.",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source", "implementation_plan"),
                writes="train_py",
                instructions="""
Generate the full runnable train.py source for this node by editing parent_source.
Most of the file should remain identical to parent_source except for the targeted experiment changes.
Do not emit commentary.
Do not emit markdown fences.
The code must remain compatible with the experiment training contract.
Keep integration with prepare.py intact.
Do not switch to a different project layout, dataset format, config file scheme, or evaluation harness.
""".strip(),
                output=OutputSpec(
                    instructions="Return only runnable Python source for train.py.",
                    parser=parse_text,
                    validator=validate_python_source,
                ),
            ),
        ),
    )
