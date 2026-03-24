from __future__ import annotations

import asyncio
import os

from compiler import compile_program
from runtime import KappaClient, RuntimeSession
from spec import AUTORESEARCH_PROGRAM
from world import IdeaStatus, ResearchWorld


async def run_research_loop(max_iterations: int, init_code_path: str, gpu_ids: list[int], debug: bool = False) -> None:
    compiled = compile_program(AUTORESEARCH_PROGRAM)
    print(compiled.describe(), flush=True)

    world = ResearchWorld(namespace_dir="namespace", init_code_path=init_code_path, gpu_ids=gpu_ids)
    root = world.initialize(clear=True)

    client = KappaClient(
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model="openai/gpt-5.2",
        debug=debug,
    )
    session = RuntimeSession(compiled, world, client, debug=debug)

    print("Running initial baseline (root node)...", flush=True)
    await world.run_pending_experiments()
    if world.nodes[root.node_id].status == IdeaStatus.FAILED:
        print("Root node failed, exiting.")
        return

    for index in range(max_iterations):
        print(f"\n{'=' * 20} Iteration {index + 1} / {max_iterations} {'=' * 20}")
        await session.run_workflow()
        print("\nIteration complete. Current research tree:")
        print(world.snapshot())


if __name__ == "__main__":
    asyncio.run(run_research_loop(10, "x0.py", [0, 1, 2, 3]))
