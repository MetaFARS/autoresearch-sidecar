from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tool_environment import ToolCatalog, ToolSpec
from .experiment_backend import BackendConfig
from .orchestrator_factory import OrchestratorConfig
from .work_context import WorkContext


@dataclass(frozen=True)
class ExperimentContract:
    name: str
    backend: BackendConfig
    work_context: WorkContext
    tools: ToolCatalog
    orchestrator: OrchestratorConfig
    default_model: str = "openai/gpt-5.2"
    default_base_url: str = "https://openrouter.ai/api/v1/chat/completions"


def make_karpathy_experiment_contract(
    repo_root: str | Path,
    namespace_dir: str | Path | None = None,
) -> ExperimentContract:
    repo_root = Path(repo_root).resolve()
    namespace_dir = Path(namespace_dir).resolve() if namespace_dir else (repo_root / "namespace").resolve()

    tool_specs = {
        "read_meta": ToolSpec(
            signature="read_meta(node_id: str) -> str",
            description="Read the metadata JSON for an experiment node.",
        ),
        "read_code": ToolSpec(
            signature="read_code(node_id: str) -> str",
            description="Read the train.py source for an experiment node.",
        ),
        "read_stdout": ToolSpec(
            signature="read_stdout(node_id: str) -> str",
            description="Read the stdout log for an experiment node.",
        ),
        "read_stderr": ToolSpec(
            signature="read_stderr(node_id: str) -> str",
            description="Read the stderr log for an experiment node.",
        ),
    }

    experiment_context = """
The experiment backend manages isolated train.py variants under namespace/<node_id>/train.py.
Each node is an experiment with:
- node_id: unique node identifier.
- parent_id: the base node this experiment extends.
- illustration: longer technical rationale for the change.
- tldr: short summary of the change.
- metric: val_bpb from the final summary. Smaller is always better.
- peak_vram_mb / memory_gb: memory usage from the final summary.
- exit_code: process return code.
- status: pending, running, success, or failed.
""".strip()

    target_contract = """
You are operating inside Karpathy's experiment pipeline.

Ground rules derived from the repository contract:
- The goal is to lower val_bpb. Smaller is always better.
- The training budget is fixed by prepare.py, so experiments must remain comparable under that wall-clock budget.
- Only train.py may change. prepare.py and its evaluation harness are read-only.
- Do not add dependencies or require extra files.
- Preserve the runnable command shape: the generated file must still work as train.py under `uv run`.
- Preserve the final summary format, especially the exact `val_bpb:` and `peak_vram_mb:` lines.
- Simplicity matters. Prefer changes that are easy to justify and easy to keep if they work.
- Experiments run from the repository root so they reuse the real prepare.py, tokenizer, data, and environment.
""".strip()

    return ExperimentContract(
        name="karpathy_experiment",
        backend=BackendConfig(
            repo_root=repo_root,
            namespace_dir=namespace_dir,
            init_code_path=repo_root / "train.py",
            runner_command=("uv", "run"),
            code_filename="train.py",
            readable_files={
                "read_meta": "meta.json",
                "read_code": "train.py",
                "read_stdout": "stdout.log",
                "read_stderr": "stderr.log",
            },
            metric_pattern=r"^val_bpb:\s*(-?\d+(?:\.\d+)?)\s*$",
            peak_vram_pattern=r"^peak_vram_mb:\s*(-?\d+(?:\.\d+)?)\s*$",
        ),
        work_context=WorkContext(
            experiment_context=experiment_context,
            target_contract=target_contract,
        ),
        tools=ToolCatalog(tool_specs=tool_specs),
        orchestrator=OrchestratorConfig(
            required_parent_anchors=(
                "from prepare import",
                "MAX_SEQ_LEN",
                "TIME_BUDGET",
                "Tokenizer.from_directory()",
                "make_dataloader(",
                "evaluate_bpb(",
                "val_bpb:",
                "peak_vram_mb:",
            ),
            forbidden_new_patterns=(
                r"tinyshakespeare",
                r"train\.bin",
                r"val\.bin",
                r"train\.pt",
                r"val\.pt",
                r"meta\.pkl",
                r"train_config\.json",
                r"config\.json",
                r"TRAIN_CONFIG",
            ),
        ),
    )
