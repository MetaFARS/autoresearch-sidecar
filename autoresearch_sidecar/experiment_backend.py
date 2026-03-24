from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

from .experiment_executor import ExperimentExecutor


@dataclass(frozen=True)
class BackendConfig:
    repo_root: Path
    namespace_dir: Path
    init_code_path: Path
    runner_command: tuple[str, ...]
    code_filename: str
    readable_files: Mapping[str, str]
    metric_pattern: str
    peak_vram_pattern: str


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class ExperimentNode:
    node_id: str
    parent_id: Optional[str]
    illustration: str
    tldr: str
    metric: float | None = None
    peak_vram_mb: float | None = None
    memory_gb: float | None = None
    exit_code: int | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "illustration": self.illustration,
            "tldr": self.tldr,
            "metric": self.metric,
            "peak_vram_mb": self.peak_vram_mb,
            "memory_gb": self.memory_gb,
            "exit_code": self.exit_code,
            "status": self.status.value,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


class ExperimentBackend:
    def __init__(self, config: BackendConfig, gpu_ids: list[int | None] | None = None) -> None:
        self.config = config
        self.repo_root = config.repo_root
        self.namespace = config.namespace_dir
        self.init_code_path = config.init_code_path
        self.runner_command = config.runner_command
        self.code_filename = config.code_filename
        self.readable_files = dict(config.readable_files)
        self.metric_re = re.compile(config.metric_pattern, re.MULTILINE)
        self.peak_vram_re = re.compile(config.peak_vram_pattern, re.MULTILINE)
        self.nodes: dict[str, ExperimentNode] = {}
        self.executor = ExperimentExecutor(gpu_ids)
        self.root_id: str | None = None

    def initialize(self, clear: bool = True) -> ExperimentNode:
        if clear and self.namespace.exists():
            shutil.rmtree(self.namespace)
        self.namespace.mkdir(parents=True, exist_ok=True)
        self.nodes = {}
        self.root_id = None

        root = ExperimentNode(
            node_id=self.new_node_id(),
            parent_id=None,
            illustration="baseline",
            tldr="baseline",
            status=ExperimentStatus.PENDING,
        )
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        self.persist_experiment(root)
        shutil.copy(self.init_code_path, self.node_dir(root.node_id) / self.code_filename)
        return root

    def new_node_id(self) -> str:
        return str(uuid.uuid4())[:6]

    def node_dir(self, node_id: str) -> Path:
        path = self.namespace / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist_experiment(self, experiment: ExperimentNode) -> None:
        (self.node_dir(experiment.node_id) / "meta.json").write_text(experiment.as_json())

    def update_experiment(self, node_id: str, **updates) -> ExperimentNode:
        experiment = self.nodes[node_id]
        for key, value in updates.items():
            setattr(experiment, key, value)
        self.persist_experiment(experiment)
        return experiment

    def add_experiment(self, *, parent_id: str | None, tldr: str, illustration: str) -> ExperimentNode:
        experiment = ExperimentNode(
            node_id=self.new_node_id(),
            parent_id=parent_id,
            tldr=tldr,
            illustration=illustration,
            status=ExperimentStatus.PENDING,
        )
        self.nodes[experiment.node_id] = experiment
        self.persist_experiment(experiment)
        return experiment

    def has_code(self, node_id: str) -> bool:
        return (self.node_dir(node_id) / self.code_filename).exists()

    def pending_nodes(self) -> list[ExperimentNode]:
        return [node for node in self.nodes.values() if node.status == ExperimentStatus.PENDING]

    async def run_pending_experiments(self) -> None:
        pending = self.pending_nodes()
        if not pending:
            return
        await asyncio.gather(*(self.executor.execute(self, node) for node in pending))

    def best_success(self) -> ExperimentNode | None:
        candidates = [
            node for node in self.nodes.values() if node.status == ExperimentStatus.SUCCESS and node.metric is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda node: node.metric if node.metric is not None else float("inf"))

    def get_root_id(self) -> str | None:
        return self.root_id

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def get_node_record(self, node_id: str) -> dict[str, object]:
        return self.nodes[node_id].as_dict()

    def snapshot(self) -> str:
        roots = [nid for nid, node in self.nodes.items() if not node.parent_id or node.parent_id not in self.nodes]
        lines: list[str] = []

        def traverse(nid: str, indent: str = "", visited: set[str] | None = None) -> None:
            seen = visited or set()
            if nid in seen:
                return
            seen.add(nid)

            node = self.nodes[nid]
            node_info = (
                f"ID: {node.node_id} | ParentID: {node.parent_id} | TLDR: {node.tldr} | "
                f"Status: {node.status.value} | val_bpb: {node.metric} | mem_gb: {node.memory_gb}"
            )
            lines.append(f"{indent}* {node_info}")

            children = sorted((cid for cid, child in self.nodes.items() if child.parent_id == nid))
            for index, child_id in enumerate(children):
                is_last = index == len(children) - 1
                child_indent = indent + ("  " if is_last else "| ")
                traverse(child_id, child_indent, seen)

        for root_id in sorted(roots):
            traverse(root_id)
        return "\n".join(lines)

    def extract_summary(self, node_id: str) -> dict[str, float | None]:
        stdout = self.read_stdout(node_id)
        metric_match = self.metric_re.search(stdout)
        peak_match = self.peak_vram_re.search(stdout)
        peak_vram_mb = float(peak_match.group(1)) if peak_match else None
        return {
            "metric": float(metric_match.group(1)) if metric_match else None,
            "peak_vram_mb": peak_vram_mb,
            "memory_gb": round(peak_vram_mb / 1024.0, 1) if peak_vram_mb is not None else None,
        }

    def write_code(self, node_id: str, code: str) -> None:
        self._write_node_file(node_id, self.code_filename, code)

    def mark_failed(self, node_id: str, message: str, exit_code: int = 1) -> None:
        self._write_node_file(node_id, "stderr.log", message.rstrip() + "\n")
        stdout_path = self.node_dir(node_id) / "stdout.log"
        if not stdout_path.exists():
            stdout_path.write_text("")
        self.update_experiment(node_id, status=ExperimentStatus.FAILED, exit_code=exit_code)

    def read_meta(self, node_id: str) -> str:
        return self._read_node_file(node_id, "meta.json")

    def read_code(self, node_id: str) -> str:
        return self._read_node_file(node_id, self.code_filename)

    def read_stdout(self, node_id: str) -> str:
        return self._read_node_file(node_id, "stdout.log", missing_ok=True)

    def read_stderr(self, node_id: str) -> str:
        return self._read_node_file(node_id, "stderr.log", missing_ok=True)

    def _read_node_file(self, node_id: str, relative_path: str, missing_ok: bool = False) -> str:
        path = self.node_dir(node_id) / relative_path
        if missing_ok and not path.exists():
            return ""
        return path.read_text()

    def _write_node_file(self, node_id: str, relative_path: str, contents: str) -> None:
        (self.node_dir(node_id) / relative_path).write_text(contents)
