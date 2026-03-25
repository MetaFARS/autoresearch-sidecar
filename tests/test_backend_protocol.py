from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from autoresearch_sidecar.agent_runtime import RoleRunner
from autoresearch_sidecar.backend_protocol import (
    ExperimentBackendPort,
    assert_experiment_backend_port,
    assert_inspection_toolset,
    build_backend_tool_handlers,
)
from autoresearch_sidecar.experiment_contract import make_karpathy_experiment_contract
from autoresearch_sidecar.orchestrator import ExperimentOrchestrator


PARENT_SOURCE = """
from prepare import Tokenizer, evaluate_bpb, make_dataloader

MAX_SEQ_LEN = 1
TIME_BUDGET = 1
Tokenizer.from_directory()
make_dataloader(
evaluate_bpb(
print("val_bpb: 1.0")
print("peak_vram_mb: 2.0")
""".strip()


class FakeRunner:
    def run(self, role: object, initial_state: dict[str, object]) -> dict[str, str]:
        return {"train_py": str(initial_state["parent_source"])}


class FakeExperimentBackend:
    def __init__(self, parent_source: str) -> None:
        self.parent_source = parent_source
        self.root_id = "root"
        self.records: dict[str, dict[str, object]] = {
            "root": {
                "node_id": "root",
                "parent_id": None,
                "tldr": "baseline",
                "illustration": "baseline",
                "status": "success",
            }
        }
        self.code = {"root": parent_source}
        self.failed: dict[str, str] = {}

    def add_experiment(self, *, parent_id: str | None, tldr: str, illustration: str) -> SimpleNamespace:
        node_id = "child"
        self.records[node_id] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "tldr": tldr,
            "illustration": illustration,
            "status": "pending",
        }
        return SimpleNamespace(node_id=node_id)

    def best_success(self) -> SimpleNamespace:
        return SimpleNamespace(node_id=self.root_id)

    def get_node_record(self, node_id: str) -> dict[str, object]:
        return self.records[node_id]

    def get_root_id(self) -> str:
        return self.root_id

    def has_code(self, node_id: str) -> bool:
        return node_id in self.code

    def has_node(self, node_id: str) -> bool:
        return node_id in self.records

    def mark_failed(self, node_id: str, message: str, exit_code: int = 1) -> None:
        self.failed[node_id] = f"{exit_code}:{message}"

    def pending_nodes(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(node_id=node_id) for node_id, record in self.records.items() if record["status"] == "pending"]

    def read_code(self, node_id: str) -> str:
        return self.code[node_id]

    def read_meta(self, node_id: str) -> str:
        return ""

    def read_stderr(self, node_id: str) -> str:
        return ""

    def read_stdout(self, node_id: str) -> str:
        return ""

    async def run_pending_experiments(self) -> None:
        return None

    def snapshot_data(self) -> dict[str, object]:
        return {
            "root_ids": ["root"],
            "nodes": {
                node_id: {
                    "node_id": record["node_id"],
                    "parent_id": record["parent_id"],
                    "tldr": record["tldr"],
                    "status": record["status"],
                    "metric": None,
                    "memory_gb": None,
                    "has_code": node_id in self.code,
                }
                for node_id, record in self.records.items()
            },
        }

    def snapshot(self) -> str:
        return "snapshot"

    def write_code(self, node_id: str, code: str) -> None:
        self.code[node_id] = code


def test_experiment_backend_assert_rejects_missing_methods() -> None:
    class IncompleteBackend:
        def snapshot_data(self) -> dict[str, object]:
            return {}

        def snapshot(self) -> str:
            return "snapshot"

    try:
        assert_experiment_backend_port(IncompleteBackend())
    except TypeError as exc:
        assert "missing required methods" in str(exc)
    else:
        raise AssertionError("Expected experiment backend assertion to fail for incomplete backend.")


def test_orchestrator_runs_against_asserted_backend(tmp_path) -> None:
    (tmp_path / "train.py").write_text(PARENT_SOURCE)
    (tmp_path / "prepare.py").write_text("pass\n")
    contract = make_karpathy_experiment_contract(tmp_path)
    backend_impl = FakeExperimentBackend(PARENT_SOURCE)
    backend: ExperimentBackendPort = assert_experiment_backend_port(backend_impl)
    toolset = assert_inspection_toolset(backend_impl)
    handlers = build_backend_tool_handlers(toolset, tuple(contract.tools.tool_specs))
    orchestrator = ExperimentOrchestrator(contract, cast(RoleRunner, FakeRunner()))

    new_node_ids = orchestrator.materialize_proposals(
        backend,
        [{"parent_id": "missing-parent", "tldr": "idea", "illustration": "detail"}],
    )
    pending_ids = orchestrator.implement_pending_nodes(backend)

    assert set(handlers) == set(contract.tools.tool_specs)
    assert new_node_ids == ["child"]
    assert pending_ids == ["child"]
    assert backend.get_node_record("child")["parent_id"] == "root"
    assert backend.read_code("child") == PARENT_SOURCE
    assert backend_impl.failed == {}
