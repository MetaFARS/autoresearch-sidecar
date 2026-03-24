from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment_backend import ExperimentBackend, ExperimentNode


class ExperimentExecutor:
    def __init__(self, gpu_ids: list[int | None] | None) -> None:
        slots = gpu_ids or [None]
        self.available_gpus: asyncio.Queue[int | None] = asyncio.Queue()
        for gpu_id in slots:
            self.available_gpus.put_nowait(gpu_id)

    async def execute(self, backend: ExperimentBackend, experiment: ExperimentNode) -> None:
        from .experiment_backend import ExperimentStatus

        gpu_id = await self.available_gpus.get()
        experiment_dir = backend.node_dir(experiment.node_id)
        stdout_path = experiment_dir / "stdout.log"
        stderr_path = experiment_dir / "stderr.log"
        script_path = experiment_dir / backend.code_filename

        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        pythonpath_parts = [str(backend.repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        command = [*backend.runner_command, os.path.relpath(script_path, backend.repo_root)]
        gpu_label = "default" if gpu_id is None else str(gpu_id)
        print(f"Executing {experiment.node_id} on slot {gpu_label}", flush=True)

        try:
            backend.update_experiment(experiment.node_id, status=ExperimentStatus.RUNNING)
            with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(backend.repo_root),
                    env=env,
                    stdout=f_out,
                    stderr=f_err,
                )
                await proc.wait()

            summary = backend.extract_summary(experiment.node_id)
            backend.update_experiment(
                experiment.node_id,
                exit_code=proc.returncode,
                metric=summary["metric"],
                peak_vram_mb=summary["peak_vram_mb"],
                memory_gb=summary["memory_gb"],
                status=(
                    ExperimentStatus.SUCCESS
                    if proc.returncode == 0 and summary["metric"] is not None
                    else ExperimentStatus.FAILED
                ),
            )
        except Exception:
            backend.update_experiment(experiment.node_id, status=ExperimentStatus.FAILED)
        finally:
            self.available_gpus.put_nowait(gpu_id)
