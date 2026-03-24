from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class IdeaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Idea:
    node_id: str
    parent_id: Optional[str]
    illustration: str
    tldr: str
    metric: Optional[float] = None
    exit_code: Optional[int] = None
    status: IdeaStatus = IdeaStatus.PENDING

    def as_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "illustration": self.illustration,
            "tldr": self.tldr,
            "metric": self.metric,
            "exit_code": self.exit_code,
            "status": self.status.value,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)


class Executor:
    def __init__(self, gpu_ids: list[int]):
        self.available_gpus: asyncio.Queue[int] = asyncio.Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put_nowait(gpu_id)

    async def execute(self, world: "ResearchWorld", idea: Idea) -> None:
        gpu_id = await self.available_gpus.get()
        idea_dir = world.node_dir(idea.node_id)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

        try:
            world.update_idea(idea.node_id, status=IdeaStatus.RUNNING)
            stdout_path = idea_dir / "stdout.log"
            stderr_path = idea_dir / "stderr.log"
            with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    str(idea_dir / "main.py"),
                    cwd=os.getcwd(),
                    env=env,
                    stdout=f_out,
                    stderr=f_err,
                )
                await proc.wait()
            world.update_idea(idea.node_id, exit_code=proc.returncode)
            if proc.returncode == 0:
                metric = world.extract_metric(idea.node_id)
                world.update_idea(idea.node_id, status=IdeaStatus.SUCCESS, metric=metric)
            else:
                world.update_idea(idea.node_id, status=IdeaStatus.FAILED)
        except Exception:
            world.update_idea(idea.node_id, status=IdeaStatus.FAILED)
        finally:
            self.available_gpus.put_nowait(gpu_id)


class ResearchWorld:
    def __init__(self, namespace_dir: str, init_code_path: str, gpu_ids: list[int]):
        self.namespace = Path(namespace_dir)
        self.init_code_path = Path(init_code_path)
        self.nodes: Dict[str, Idea] = {}
        self.executor = Executor(gpu_ids)

    def initialize(self, clear: bool = True) -> Idea:
        if clear and self.namespace.exists():
            shutil.rmtree(self.namespace)
        self.namespace.mkdir(parents=True, exist_ok=True)

        root = Idea(
            node_id=self.new_node_id(),
            parent_id=None,
            illustration="",
            tldr="",
            status=IdeaStatus.PENDING,
        )
        self.nodes[root.node_id] = root
        self.persist_idea(root)
        shutil.copy(self.init_code_path, self.node_dir(root.node_id) / "main.py")
        return root

    def new_node_id(self) -> str:
        return str(uuid.uuid4())[:6]

    def node_dir(self, node_id: str) -> Path:
        path = self.namespace / node_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def persist_idea(self, idea: Idea) -> None:
        (self.node_dir(idea.node_id) / "meta.json").write_text(idea.as_json())

    def update_idea(self, node_id: str, **updates) -> Idea:
        idea = self.nodes[node_id]
        for key, value in updates.items():
            setattr(idea, key, value)
        self.persist_idea(idea)
        return idea

    def add_idea(self, *, parent_id: Optional[str], tldr: str, illustration: str) -> Idea:
        idea = Idea(
            node_id=self.new_node_id(),
            parent_id=parent_id,
            tldr=tldr,
            illustration=illustration,
            status=IdeaStatus.PENDING,
        )
        self.nodes[idea.node_id] = idea
        self.persist_idea(idea)
        return idea

    def has_code(self, node_id: str) -> bool:
        return (self.node_dir(node_id) / "main.py").exists()

    def pending_nodes(self) -> list[Idea]:
        return [node for node in self.nodes.values() if node.status == IdeaStatus.PENDING]

    async def run_pending_experiments(self) -> None:
        pending = self.pending_nodes()
        await asyncio.gather(*(self.executor.execute(self, node) for node in pending))

    def snapshot(self) -> str:
        roots = [nid for nid, node in self.nodes.items() if not node.parent_id or node.parent_id not in self.nodes]
        lines: list[str] = []

        def traverse(nid: str, indent: str = "", visited=None) -> None:
            visited = visited or set()
            if nid in visited:
                return
            visited.add(nid)

            node = self.nodes[nid]
            node_info = (
                f"ID: {node.node_id} | ParentID: {node.parent_id} | TLDR: {node.tldr} | "
                f"Status: {node.status.value} | Metric: {node.metric}"
            )
            lines.append(f"{indent}* {node_info}")

            children = [cid for cid, cnode in self.nodes.items() if cnode.parent_id == nid]
            for index, child_id in enumerate(children):
                is_last = index == len(children) - 1
                new_indent = indent + ("  " if is_last else "| ")
                traverse(child_id, new_indent, visited)

        for root_id in roots:
            traverse(root_id)
        return "\n".join(lines)

    def extract_metric(self, node_id: str) -> Optional[float]:
        stdout = self.read_stdout(node_id)
        match = re.search(r"Metric:\s*(-?\d+(\.\d+)?)", stdout)
        return float(match.group(1)) if match else None

    def write_code(self, node_id: str, code: str) -> None:
        (self.node_dir(node_id) / "main.py").write_text(code)

    def read_meta(self, node_id: str) -> str:
        return (self.node_dir(node_id) / "meta.json").read_text()

    def read_code(self, node_id: str) -> str:
        return (self.node_dir(node_id) / "main.py").read_text()

    def read_stdout(self, node_id: str) -> str:
        path = self.node_dir(node_id) / "stdout.log"
        return path.read_text() if path.exists() else ""

    def read_stderr(self, node_id: str) -> str:
        path = self.node_dir(node_id) / "stderr.log"
        return path.read_text() if path.exists() else ""

    def get_node_record(self, node_id: str) -> dict[str, object]:
        node = self.nodes[node_id]
        return node.as_dict()
