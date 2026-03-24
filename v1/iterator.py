import asyncio
import os
import shutil

from agent import Agent
from state import IdeaManager, IdeaStatus


async def run_research_loop(max_iterations: int, init_code_path: str, gpu_ids) -> None:
    if os.path.exists("namespace"):
        shutil.rmtree("namespace")
    os.makedirs("namespace", exist_ok=True)

    agent = Agent(
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model="openai/gpt-5.2",
    )

    manager = IdeaManager(agent, init_code_path, gpu_ids)
    print("Running initial baseline (root node)...", flush=True)
    await manager.do_experiments()

    root = list(manager.nodes.values())[0]
    if root.status == IdeaStatus.FAILED:
        print("Root node failed, exiting.")
        return

    for i in range(max_iterations):
        print(f"\n{'=' * 20} Iteration {i + 1} / {max_iterations} {'=' * 20}")

        print("Generating and implementing v0 ideas...")
        manager.gen_ideas()

        print("Running experiments...")
        await manager.do_experiments()

        print("\nIteration complete. Current research tree:")
        print(manager.snapshot())


if __name__ == "__main__":
    asyncio.run(run_research_loop(10, "x0.py", [0, 1, 2, 3]))
