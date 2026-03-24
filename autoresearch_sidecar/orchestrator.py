from __future__ import annotations

from .agent_runtime import RoleRunner
from .backend_protocol import ExperimentBackendPort
from .experiment_contract import ExperimentContract
from .orchestrator_factory import build_implementer_role, build_planner_role
from .orchestrator_validators import validate_train_py_against_parent


class ExperimentOrchestrator:
    def __init__(self, contract: ExperimentContract, allowed_tools: tuple[str, ...], runner: RoleRunner) -> None:
        self.contract = contract
        self.runner = runner
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
            if parent_id is not None and not backend.has_node(parent_id):
                parent_id = fallback_parent
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
            parent_id = node.get("parent_id") or backend.get_root_id()
            if parent_id is None:
                raise ValueError(f"Node {node_id} has no parent and backend has no root.")
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
