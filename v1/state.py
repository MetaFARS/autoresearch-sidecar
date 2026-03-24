import asyncio
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


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

    def save(self) -> None:
        dest = Path("namespace") / self.node_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "meta.json").write_text(
            json.dumps(
                {
                    "node_id": self.node_id,
                    "parent_id": self.parent_id,
                    "illustration": self.illustration,
                    "tldr": self.tldr,
                    "metric": self.metric,
                    "exit_code": self.exit_code,
                    "status": self.status.value,
                },
                indent=4,
            )
        )

    def mark(self, *, status: Optional[IdeaStatus] = None, metric: Optional[float] = None, exit_code: Optional[int] = None) -> None:
        if status is not None:
            self.status = status
        if metric is not None:
            self.metric = metric
        if exit_code is not None:
            self.exit_code = exit_code
        self.save()

    def create_runner(self, gpu_id: int):
        dest = Path("namespace") / self.node_id
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

        async def _run() -> int:
            stdout_handle = open(dest / "stdout.log", "w")
            stderr_handle = open(dest / "stderr.log", "w")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python",
                    str(dest / "main.py"),
                    cwd=os.getcwd(),
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                await proc.wait()
                return proc.returncode
            finally:
                stdout_handle.close()
                stderr_handle.close()

        return _run


class Executor:
    def __init__(self, gpu_ids):
        self.available_gpus = asyncio.Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put_nowait(gpu_id)

    async def execute(self, idea: Idea) -> None:
        gpu_id = await self.available_gpus.get()
        print(f"Executing {idea.node_id} on GPU {gpu_id}")
        try:
            idea.mark(status=IdeaStatus.RUNNING)
            return_code = await idea.create_runner(gpu_id)()
            idea.exit_code = return_code

            if return_code == 0:
                log_path = Path("namespace") / idea.node_id / "stdout.log"
                log_text = log_path.read_text() if log_path.exists() else ""
                match = re.search(r"Metric:\s*(-?\d+(\.\d+)?)", log_text)
                if match:
                    idea.metric = float(match.group(1))
                idea.mark(status=IdeaStatus.SUCCESS, metric=idea.metric, exit_code=return_code)
            else:
                idea.mark(status=IdeaStatus.FAILED, exit_code=return_code)
        except Exception:
            idea.mark(status=IdeaStatus.FAILED)
        finally:
            self.available_gpus.put_nowait(gpu_id)


class IdeaManager:
    def __init__(self, agent, init_code_path: str, gpu_ids):
        self.nodes: Dict[str, Idea] = {}
        self.agent = agent

        root = Idea(
            node_id=str(uuid.uuid4())[:6],
            parent_id=None,
            illustration="",
            tldr="",
            status=IdeaStatus.PENDING,
        )
        root.save()
        self.nodes[root.node_id] = root

        shutil.copy(init_code_path, Path("namespace") / root.node_id / "main.py")
        self.executor = Executor(gpu_ids)

    def gen_ideas(self) -> None:
        ideas = self.agent.generate_ideas(self.snapshot())

        for idea_data in ideas:
            node = Idea(node_id=str(uuid.uuid4())[:6], **idea_data, status=IdeaStatus.PENDING)
            node.save()
            self.nodes[node.node_id] = node

        for node in self.nodes.values():
            if node.status == IdeaStatus.PENDING and not (Path("namespace") / node.node_id / "main.py").exists():
                self.agent.implement(self.snapshot(), node.node_id)

    async def do_experiments(self) -> None:
        pending_nodes = [node for node in self.nodes.values() if node.status == IdeaStatus.PENDING]
        tasks = [self.executor.execute(node) for node in pending_nodes]
        await asyncio.gather(*tasks)

    def snapshot(self) -> str:
        roots = [nid for nid, node in self.nodes.items() if not node.parent_id or node.parent_id not in self.nodes]
        lines = []

        def _traverse(nid: str, indent: str = "", visited: Optional[set[str]] = None) -> None:
            if visited is None:
                visited = set()
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
                _traverse(child_id, new_indent, visited)

        for root_id in roots:
            _traverse(root_id)

        return "\n".join(lines)
