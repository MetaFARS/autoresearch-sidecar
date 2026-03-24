from __future__ import annotations

"""Specialized world adapter emitted for autoresearch_program.

This adapter narrows the residual runtime boundary to the exact world
methods required by the compiled host ops and declared tools.
"""

from typing import Any

PROGRAM_NAME = 'autoresearch_program'
HOST_WORLD_METHODS = ('add_idea', 'get_node_record', 'has_code', 'pending_nodes', 'run_pending_experiments', 'snapshot', 'write_code')
TOOL_WORLD_METHODS = ('read_code', 'read_meta', 'read_stderr', 'read_stdout')
REQUIRED_WORLD_METHODS = ('add_idea', 'get_node_record', 'has_code', 'pending_nodes', 'read_code', 'read_meta', 'read_stderr', 'read_stdout', 'run_pending_experiments', 'snapshot', 'write_code')

class AutoresearchProgramWorldAdapter:
    __slots__ = ("_backend",)

    REQUIRED_WORLD_METHODS = REQUIRED_WORLD_METHODS
    HOST_WORLD_METHODS = HOST_WORLD_METHODS
    TOOL_WORLD_METHODS = TOOL_WORLD_METHODS

    def __init__(self, world: Any):
        missing = [name for name in REQUIRED_WORLD_METHODS if not hasattr(world, name)]
        if missing:
            missing_text = ", ".join(missing)
            raise TypeError(f"World object is missing required methods for {PROGRAM_NAME}: {missing_text}")
        self._backend = world

    @property
    def backend(self) -> Any:
        return self._backend

    def __repr__(self) -> str:
        return f"AutoresearchProgramWorldAdapter<{type(self._backend).__name__}>"

    def add_idea(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.add_idea(*args, **kwargs)

    def get_node_record(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.get_node_record(*args, **kwargs)

    def has_code(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.has_code(*args, **kwargs)

    def pending_nodes(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.pending_nodes(*args, **kwargs)

    def read_code(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.read_code(*args, **kwargs)

    def read_meta(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.read_meta(*args, **kwargs)

    def read_stderr(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.read_stderr(*args, **kwargs)

    def read_stdout(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.read_stdout(*args, **kwargs)

    def run_pending_experiments(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.run_pending_experiments(*args, **kwargs)

    def snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.snapshot(*args, **kwargs)

    def write_code(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.write_code(*args, **kwargs)

def adapt_world(world: Any) -> AutoresearchProgramWorldAdapter:
    if isinstance(world, AutoresearchProgramWorldAdapter):
        return world
    return AutoresearchProgramWorldAdapter(world)

__all__ = [
    "AutoresearchProgramWorldAdapter",
    "PROGRAM_NAME",
    "HOST_WORLD_METHODS",
    "TOOL_WORLD_METHODS",
    "REQUIRED_WORLD_METHODS",
    "adapt_world",
]
