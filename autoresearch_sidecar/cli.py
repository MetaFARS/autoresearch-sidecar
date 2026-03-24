from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .tool_environment import ToolHost
from .agent_runtime import ChatCompletionClient, RoleRunner
from .agent_trace import JsonDict, trace_to_json
from .backend_protocol import assert_experiment_backend_port, assert_inspection_toolset, build_backend_tool_handlers
from .experiment_backend import ExperimentBackend, ExperimentStatus
from .experiment_contract import make_karpathy_experiment_contract
from .orchestrator import ExperimentOrchestrator


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_default_envs(repo_root: Path) -> None:
    load_env_file(repo_root / ".env")
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env != (repo_root / ".env").resolve():
        load_env_file(cwd_env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoresearch-sidecar against a target repo.")
    parser.add_argument("--repo-root", default=".", help="Path to the target autoresearch repo.")
    parser.add_argument("--namespace-dir", default=None, help="Override namespace directory inside the target repo.")
    parser.add_argument("--max-iterations", type=int, default=10, help="Number of planner/implementer iterations to run after baseline.")
    parser.add_argument("--debug", action="store_true", help="Print intermediate role/tool outputs.")
    parser.add_argument("--model", default=None, help="Override the LLM model name.")
    parser.add_argument("--base-url", default=None, help="Override the chat completions base URL.")
    parser.add_argument("--trace-file", default=None, help="Write the v5-style runtime trace as JSON.")
    parser.add_argument("--gpu-id", dest="gpu_ids", action="append", type=int, default=None, help="Repeat to provide one or more GPU ids.")
    return parser.parse_args()


async def run_orchestration_loop(
    repo_root: str | Path = ".",
    *,
    namespace_dir: str | Path | None = None,
    max_iterations: int = 10,
    gpu_ids: list[int | None] | None = None,
    debug: bool = False,
    model: str | None = None,
    base_url: str | None = None,
    trace_file: str | Path | None = None,
) -> None:
    repo_root = Path(repo_root).resolve()
    load_default_envs(repo_root)

    contract = make_karpathy_experiment_contract(repo_root=repo_root, namespace_dir=namespace_dir)
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    backend_impl = ExperimentBackend(contract.backend, gpu_ids=gpu_ids)
    root = backend_impl.initialize(clear=True)
    backend = assert_experiment_backend_port(backend_impl)
    toolset = assert_inspection_toolset(backend_impl)
    trace: list[JsonDict] = []
    resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or contract.default_base_url
    resolved_model = model or os.getenv("AUTORESEARCH_MODEL") or contract.default_model
    orchestrator: ExperimentOrchestrator | None = None
    if max_iterations > 0:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        client = ChatCompletionClient(
            base_url=resolved_base_url,
            api_key=api_key,
            model=resolved_model,
        )
        tool_host = ToolHost(
            contract.tools.tool_specs,
            build_backend_tool_handlers(toolset, tuple(contract.tools.tool_specs)),
            trace=trace,
        )
        runner = RoleRunner(client, tool_host, trace=trace, debug_mode=debug)
        orchestrator = ExperimentOrchestrator(contract, runner)

    try:
        print(f"Target repo: {repo_root}", flush=True)
        print(f"Namespace: {contract.backend.namespace_dir}", flush=True)
        print("Running initial baseline (root node)...", flush=True)
        await backend.run_pending_experiments()
        root_record = backend.get_node_record(root.node_id)
        if root_record["status"] == ExperimentStatus.FAILED.value:
            print("Root node failed, exiting.")
            return

        for index in range(max_iterations):
            if orchestrator is None:
                raise RuntimeError("Orchestrator is not available for iterative execution.")
            print(f"\n{'=' * 20} Iteration {index + 1} / {max_iterations} {'=' * 20}", flush=True)
            new_node_ids = await orchestrator.run_iteration(backend)
            print(f"New nodes: {new_node_ids}", flush=True)
            print("\nCurrent experiment tree:", flush=True)
            print(backend.snapshot(), flush=True)
    finally:
        if trace_file is not None:
            Path(trace_file).resolve().write_text(trace_to_json(trace))


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_orchestration_loop(
            repo_root=args.repo_root,
            namespace_dir=args.namespace_dir,
            max_iterations=args.max_iterations,
            gpu_ids=args.gpu_ids,
            debug=args.debug,
            model=args.model,
            base_url=args.base_url,
            trace_file=args.trace_file,
        )
    )
