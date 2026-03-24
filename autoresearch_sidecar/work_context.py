from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkContext:
    experiment_context: str
    target_contract: str
