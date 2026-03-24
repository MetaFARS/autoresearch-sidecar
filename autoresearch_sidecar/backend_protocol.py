from __future__ import annotations

from typing import Protocol, Sequence, TypeGuard, cast

from .tool_environment import ToolHandler


PROJECT_NAME = "autoresearch_sidecar"
HOST_BACKEND_METHODS = (
    "add_experiment",
    "best_success",
    "get_node_record",
    "get_root_id",
    "has_code",
    "has_node",
    "mark_failed",
    "pending_nodes",
    "read_code",
    "run_pending_experiments",
    "snapshot",
    "write_code",
)
INSPECTION_TOOL_METHODS = (
    "read_code",
    "read_meta",
    "read_stderr",
    "read_stdout",
)
ExperimentRecord = dict[str, object]


class BackendNode(Protocol):
    node_id: str


class ExperimentBackendPort(Protocol):
    def add_experiment(self, *, parent_id: str | None, tldr: str, illustration: str) -> BackendNode: ...
    def best_success(self) -> BackendNode | None: ...
    def get_node_record(self, node_id: str) -> ExperimentRecord: ...
    def get_root_id(self) -> str | None: ...
    def has_code(self, node_id: str) -> bool: ...
    def has_node(self, node_id: str) -> bool: ...
    def mark_failed(self, node_id: str, message: str, exit_code: int = 1) -> None: ...
    def pending_nodes(self) -> Sequence[BackendNode]: ...
    def read_code(self, node_id: str) -> str: ...
    async def run_pending_experiments(self) -> None: ...
    def snapshot(self) -> str: ...
    def write_code(self, node_id: str, code: str) -> None: ...


class ExperimentInspectionToolset(Protocol):
    def read_code(self, node_id: str) -> str: ...
    def read_meta(self, node_id: str) -> str: ...
    def read_stderr(self, node_id: str) -> str: ...
    def read_stdout(self, node_id: str) -> str: ...


def is_experiment_backend_port(backend: object) -> TypeGuard[ExperimentBackendPort]:
    return all(callable(getattr(backend, name, None)) for name in HOST_BACKEND_METHODS)


def is_inspection_toolset(toolset: object) -> TypeGuard[ExperimentInspectionToolset]:
    return all(callable(getattr(toolset, name, None)) for name in INSPECTION_TOOL_METHODS)


def assert_experiment_backend_port(backend: object) -> ExperimentBackendPort:
    if not is_experiment_backend_port(backend):
        missing = [name for name in HOST_BACKEND_METHODS if not callable(getattr(backend, name, None))]
        missing_text = ", ".join(missing)
        raise TypeError(f"Backend object is missing required methods for {PROJECT_NAME}: {missing_text}")
    return cast(ExperimentBackendPort, backend)


def assert_inspection_toolset(toolset: object) -> ExperimentInspectionToolset:
    if not is_inspection_toolset(toolset):
        missing = [name for name in INSPECTION_TOOL_METHODS if not callable(getattr(toolset, name, None))]
        missing_text = ", ".join(missing)
        raise TypeError(f"Toolset object is missing required methods for {PROJECT_NAME}: {missing_text}")
    return cast(ExperimentInspectionToolset, toolset)


def build_backend_tool_handlers(
    toolset: ExperimentInspectionToolset,
    tool_names: tuple[str, ...],
) -> dict[str, ToolHandler]:
    handlers: dict[str, ToolHandler] = {}
    missing: list[str] = []
    for name in tool_names:
        handler = getattr(toolset, name, None)
        if not callable(handler):
            missing.append(name)
            continue
        handlers[name] = cast(ToolHandler, handler)
    if missing:
        missing_text = ", ".join(missing)
        raise TypeError(f"Backend object is missing declared tool handlers for {PROJECT_NAME}: {missing_text}")
    return handlers
