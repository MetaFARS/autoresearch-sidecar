from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch_bundle import REQUIRED_WORLD_METHODS, run_workflow as run_bundle_workflow
from compiler import compile_program
from runtime import RuntimeSession, compare_traces, trace_to_json
from spec import AUTORESEARCH_PROGRAM
from world import IdeaStatus, ResearchWorld


class DeterministicResearchWorld(ResearchWorld):
    def __init__(self, namespace_dir: str, init_code_path: str, gpu_ids: list[int]):
        super().__init__(namespace_dir=namespace_dir, init_code_path=init_code_path, gpu_ids=gpu_ids)
        self._counter = 0

    def new_node_id(self) -> str:
        self._counter += 1
        return f"n{self._counter:04d}"


class FakeClient:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def complete(self, messages, stop=None):
        if not self.outputs:
            raise RuntimeError("FakeClient has no more outputs.")
        return self.outputs.pop(0)


def _make_outputs(root_id: str) -> list[str]:
    proposals = json.dumps([{"parent_id": root_id, "tldr": "bump metric", "illustration": "Try a trivial code tweak."}])
    return [
        f'<tool>read_meta("{root_id}")</tool><stop>',
        "Investigate root performance and propose a simple variant.",
        proposals,
        f'<tool>read_code("{root_id}")</tool><stop>',
        "Plan: keep structure simple and emit a higher metric for smoke testing.",
        'print("Metric: 1.23")',
    ]


async def _prepare_world(td_path: Path, name: str) -> tuple[DeterministicResearchWorld, str]:
    init_code = td_path / "x0.py"
    init_code.write_text('print("Metric: 0.5")\n', encoding="utf-8")
    world = DeterministicResearchWorld(namespace_dir=str(td_path / name), init_code_path=str(init_code), gpu_ids=[0])
    root = world.initialize(clear=True)
    world.update_idea(root.node_id, status=IdeaStatus.SUCCESS, metric=0.5, exit_code=0)
    return world, root.node_id


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        generic_world, root_id = await _prepare_world(td_path, "generic_ns")
        bundle_world, bundle_root_id = await _prepare_world(td_path, "bundle_ns")
        assert bundle_root_id == root_id
        program = compile_program(AUTORESEARCH_PROGRAM)
        generic_trace: list[dict[str, object]] = []
        bundle_trace: list[dict[str, object]] = []
        generic_state = await RuntimeSession(program, generic_world, FakeClient(_make_outputs(root_id)), trace=generic_trace, debug=False).run_workflow()
        bundle_state = await run_bundle_workflow(bundle_world, FakeClient(_make_outputs(root_id)), trace=bundle_trace, debug=False)
        ok, diff = compare_traces(generic_trace, bundle_trace)
        if not ok:
            print(diff)
            raise AssertionError("generic and bundle traces diverged")
        assert generic_world.snapshot() == bundle_world.snapshot()
        assert generic_state["planner_run"]["proposals"] == bundle_state["planner_run"]["proposals"]
        assert all(hasattr(bundle_world, name) for name in REQUIRED_WORLD_METHODS)
        print("generic and bundle oracle traces match")
        print(f"specialized world boundary: {list(REQUIRED_WORLD_METHODS)}")
        print(f"trace events: {len(generic_trace)}")
        print(trace_to_json(generic_trace))


if __name__ == "__main__":
    asyncio.run(main())
