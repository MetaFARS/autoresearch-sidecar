from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .agent_runtime import RoleRunner
from .backend_protocol import ExperimentBackendPort
from .orchestrator_validators import (
    parse_json_list,
    parse_text,
    validate_nonempty_text,
    validate_proposals,
    validate_python_source,
    validate_train_py_against_parent,
)
from .work_context import WorkContext
from .workflow_spec import CommitSpec, OutputSpec, PhaseSpec, RoleSpec

if TYPE_CHECKING:
    from .experiment_contract import ExperimentContract


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
                purpose="Observe promising nodes and collect grounded experiment notes.",
                dominant_mode="observe",
                reads=("snapshot",),
                instructions="""
Observe successful nodes and informative failures before planning new work.
Use tools one call at a time when direct observation is needed.
Focus on grounded findings that could improve val_bpb without violating the train.py-only contract.
Commit concise experiment notes only. Do not emit proposals yet.
""".strip(),
                commit=CommitSpec(
                    writes="experiment_notes",
                    output=OutputSpec(
                        instructions="Return only concise experiment notes. No JSON. No markdown fences.",
                        parser=parse_text,
                        validator=validate_nonempty_text,
                    ),
                ),
                allow_tools=True,
                allowed_tools=allowed_tools,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_proposals",
                purpose="Plan the next concrete proposals from the observed notes.",
                dominant_mode="plan",
                reads=("snapshot", "experiment_notes"),
                instructions="""
Plan 1 to 3 concrete proposals from the current world view.
Each proposal must include parent_id, tldr, and illustration.
Choose parent_id values from the snapshot when possible.
Commit only the JSON list.
""".strip(),
                commit=CommitSpec(
                    writes="proposals",
                    output=OutputSpec(
                        instructions='Return only valid JSON of the form [{"parent_id": "...", "tldr": "...", "illustration": "..."}].',
                        parser=parse_json_list,
                        validator=validate_proposals,
                    ),
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
                purpose="Observe the parent implementation and derive a grounded edit plan.",
                dominant_mode="observe",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source"),
                instructions="""
Observe the parent node's code and logs as needed.
Work out the smallest coherent implementation that matches the proposal.
Prefer small, targeted edits over unrelated rewrites.
The parent_source shown in state is the file you must edit.
Preserve the parent's prepare.py integration, data loading path, tokenizer usage, and evaluation harness unless the proposal explicitly changes them.
Do not introduce alternate dataset conventions or config-file conventions that are not already present in parent_source.
Commit only the implementation plan, not code.
""".strip(),
                commit=CommitSpec(
                    writes="implementation_plan",
                    output=OutputSpec(
                        instructions="Return only a concise implementation plan. No code fences.",
                        parser=parse_text,
                        validator=validate_nonempty_text,
                    ),
                ),
                allow_tools=True,
                allowed_tools=allowed_tools,
                max_tool_rounds=6,
            ),
            PhaseSpec(
                name="emit_train_py",
                purpose="Act by producing the full runnable train.py source.",
                dominant_mode="action",
                reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "parent_source", "implementation_plan"),
                instructions="""
Act by generating the full runnable train.py source for this node from parent_source and implementation_plan.
Most of the file should remain identical to parent_source except for the targeted experiment changes.
Do not emit commentary.
Do not emit markdown fences.
The code must remain compatible with the experiment training contract.
Keep integration with prepare.py intact.
Do not switch to a different project layout, dataset format, config file scheme, or evaluation harness.
Commit only the runnable train.py source.
""".strip(),
                commit=CommitSpec(
                    writes="train_py",
                    output=OutputSpec(
                        instructions="Return only runnable Python source for train.py.",
                        parser=parse_text,
                        validator=validate_python_source,
                    ),
                ),
            ),
        ),
    )


class ExperimentOrchestrator:
    def __init__(self, contract: ExperimentContract, runner: RoleRunner) -> None:
        self.contract = contract
        self.runner = runner
        allowed_tools = tuple(contract.tools.tool_specs)
        self.planner_role = build_planner_role(contract.work_context, allowed_tools)
        self.implementer_role = build_implementer_role(contract.work_context, allowed_tools)

    def propose(self, backend: ExperimentBackendPort) -> list[dict[str, str | None]]:
        state = self.runner.run(self.planner_role, {"snapshot": backend.snapshot()})
        return state["proposals"]

    def materialize_proposals(
        self,
        backend: ExperimentBackendPort,
        proposals: list[dict[str, str | None]],
    ) -> list[str]:
        fallback_parent = None
        best = backend.best_success()
        if best is not None:
            fallback_parent = best.node_id
        else:
            fallback_parent = backend.get_root_id()

        new_node_ids: list[str] = []
        for proposal in proposals:
            parent_id = proposal["parent_id"]
            if parent_id is None or not backend.has_node(parent_id):
                parent_id = fallback_parent
            if parent_id is None:
                raise ValueError("Unable to resolve a parent node for the proposal.")
            node = backend.add_experiment(
                parent_id=parent_id,
                tldr=str(proposal["tldr"]),
                illustration=str(proposal["illustration"]),
            )
            new_node_ids.append(node.node_id)
        return new_node_ids

    def implement_pending_nodes(self, backend: ExperimentBackendPort) -> list[str]:
        pending_ids = [node.node_id for node in backend.pending_nodes() if not backend.has_code(node.node_id)]
        if not pending_ids:
            return []

        snapshot = backend.snapshot()
        for node_id in pending_ids:
            node = backend.get_node_record(node_id)
            parent_id = node.get("parent_id")
            if parent_id is None:
                raise ValueError(f"Node {node_id} has no resolved parent.")
            if not isinstance(parent_id, str):
                raise ValueError(f"Node {node_id} has invalid parent id {parent_id!r}.")
            parent_source = backend.read_code(parent_id)
            try:
                state = self.runner.run(
                    self.implementer_role,
                    {
                        "snapshot": snapshot,
                        "node_id": node_id,
                        "parent_id": parent_id,
                        "tldr": str(node["tldr"]),
                        "illustration": str(node["illustration"]),
                        "parent_source": parent_source,
                    },
                )
                candidate_source = validate_train_py_against_parent(
                    self.contract.orchestrator,
                    parent_source,
                    state["train_py"],
                )
                backend.write_code(node_id, candidate_source)
            except Exception as exc:
                backend.mark_failed(node_id, f"Implementer rejected before execution: {exc}", exit_code=1)
                continue
        return pending_ids

    async def run_iteration(self, backend: ExperimentBackendPort) -> list[str]:
        proposals = self.propose(backend)
        new_node_ids = self.materialize_proposals(backend, proposals)
        self.implement_pending_nodes(backend)
        await backend.run_pending_experiments()
        return new_node_ids
