from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from compiled_kernel import create_session
from world import IdeaStatus, ResearchWorld


class FakeClient:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def complete(self, messages, stop=None):
        if not self.outputs:
            raise RuntimeError("FakeClient has no more outputs.")
        return self.outputs.pop(0)


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        init_code = td_path / "x0.py"
        init_code.write_text('print("Metric: 0.5")\n', encoding="utf-8")

        world = ResearchWorld(namespace_dir=str(td_path / "namespace"), init_code_path=str(init_code), gpu_ids=[0])
        root = world.initialize(clear=True)
        world.update_idea(root.node_id, status=IdeaStatus.SUCCESS, metric=0.5, exit_code=0)

        proposals = json.dumps(
            [
                {
                    "parent_id": root.node_id,
                    "tldr": "bump metric",
                    "illustration": "Try a trivial code tweak.",
                }
            ]
        )

        client = FakeClient(
            [
                "Investigate root performance and propose a simple variant.",
                proposals,
                "Plan: keep structure simple and emit a higher metric for smoke testing.",
                'print("Metric: 1.23")',
            ]
        )

        session = create_session(world, client, debug=False)
        state = await session.run_workflow()

        new_nodes = [node for node in world.nodes.values() if node.node_id != root.node_id]
        assert len(new_nodes) == 1
        node = new_nodes[0]
        assert node.status == IdeaStatus.SUCCESS
        assert abs((node.metric or 0.0) - 1.23) < 1e-9

        print("residual runtime smoke test passed")
        print("generated node:", node.node_id)
        print("planner proposals:", state["planner_run"]["proposals"])


if __name__ == "__main__":
    asyncio.run(main())
