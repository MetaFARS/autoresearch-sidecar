import json
import os
import uuid
import asyncio
import subprocess
import re
import shutil
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


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
    metric: float = None
    exit_code: Optional[int] = None
    status: IdeaStatus = IdeaStatus.PENDING

    def save(self):
        (Path("namespace") / self.node_id).mkdir(parents=True, exist_ok=True)
        (Path("namespace") / self.node_id / "meta.json").write_text(
            json.dumps(vars(self), indent=4, default=lambda o: o.value if isinstance(o, Enum) else str(o)))

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        self.save()

    def closure(self, gpu_id):
        dest = Path("namespace") / self.node_id
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        
        async def _run():
            f_out = open(dest / "stdout.log", "w")
            f_err = open(dest / "stderr.log", "w")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python", str(dest / "main.py"),
                    cwd=os.getcwd(),
                    env=env,
                    stdout=f_out,
                    stderr=f_err
                )
                await proc.wait()
                self.exit_code = proc.returncode

            finally:
                f_out.close()
                f_err.close()
                
        return _run()


class Executor:
    def __init__(self, gpu_ids):
        self.available_gpus = asyncio.Queue()
        for gpu_id in gpu_ids:
            self.available_gpus.put_nowait(gpu_id)

    async def execute(self, idea: Idea):
        gpu_id = await self.available_gpus.get()
        print(f"Executing {idea.node_id} on GPU {gpu_id}")
        try:
            idea.status = IdeaStatus.RUNNING
            await idea.closure(gpu_id)
            if idea.exit_code == 0:
                idea.status = IdeaStatus.SUCCESS
                log_path = Path("namespace") / idea.node_id / "stdout.log"
                log_text = log_path.read_text()
                match = re.search(r"Metric:\s*(-?\d+(\.\d+)?)", log_text)
                if match:
                    idea.metric = float(match.group(1))
            else:
                idea.status = IdeaStatus.FAILED
        except Exception as e:
            idea.status = IdeaStatus.FAILED
        finally:
            self.available_gpus.put_nowait(gpu_id)


class IdeaManager:
    def __init__(self, agent, init_code_path, gpu_ids):
        self.nodes: Dict[str, Idea] = {}
        self.agent = agent
        
        root = Idea(
            node_id=str(uuid.uuid4())[:6], 
            parent_id=None, 
            illustration="", 
            tldr="", 
            status=IdeaStatus.PENDING)
        self.nodes[root.node_id] = root

        shutil.copy(init_code_path, Path("namespace") / root.node_id / "main.py")
        self.executer = Executor(gpu_ids)


    def gen_ideas(self):
        ideas = self.agent.generate_ideas(self.snapshot())

        for idea_data in ideas:
            node = Idea(node_id=str(uuid.uuid4())[:6], **idea_data, status=IdeaStatus.PENDING)
            self.nodes[node.node_id] = node

        for node in self.nodes.values():
            if node.status == IdeaStatus.PENDING:
                self.agent.implement(self.snapshot(), node.node_id)

    async def do_experiments(self):
        pending_nodes = [node for node in self.nodes.values() if node.status == IdeaStatus.PENDING]
        tasks = [self.executer.execute(node) for node in pending_nodes]
        await asyncio.gather(*tasks)

    def snapshot(self):
        roots = [nid for nid, node in self.nodes.items() if not node.parent_id or node.parent_id not in self.nodes]
        
        lines = []
        def _traverse(nid, indent="", visited=None):
            if visited is None: visited = set()
            if nid in visited: return
            visited.add(nid)
            
            node = self.nodes[nid]
            node_info = f"ID: {node.node_id} | ParentID: {node.parent_id}| TLDR: {node.tldr} | Status: {node.status.value} | Metric: {node.metric}"
            lines.append(f"{indent}* {node_info}")
            
            children = [cid for cid, cnode in self.nodes.items() if cnode.parent_id == nid]
            for i, cid in enumerate(children):
                is_last = (i == len(children) - 1)
                new_indent = indent + ("  " if is_last else "| ")
                _traverse(cid, new_indent, visited)

        for rid in roots:
            _traverse(rid)
        
        return "\n".join(lines)

