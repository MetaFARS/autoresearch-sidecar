from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator_factory import OrchestratorConfig


def parse_text(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"Expected text output, got {type(raw).__name__}.")
    return raw.strip()


def validate_nonempty_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty text output.")
    return value.strip()


def parse_json_list(raw: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError(f"Expected JSON text output, got {type(raw).__name__}.")
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def validate_proposals(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Expected a non-empty list of proposals.")

    cleaned: list[dict[str, str | None]] = []
    for item in value[:3]:
        if not isinstance(item, dict):
            raise ValueError(f"Each proposal must be an object, got {type(item)!r}.")
        parent_id = item.get("parent_id")
        tldr = item.get("tldr")
        illustration = item.get("illustration")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("parent_id must be a string or null.")
        if not isinstance(tldr, str) or not tldr.strip():
            raise ValueError("Proposal tldr must be non-empty.")
        if not isinstance(illustration, str) or not illustration.strip():
            raise ValueError("Proposal illustration must be non-empty.")
        cleaned.append(
            {
                "parent_id": parent_id.strip() if isinstance(parent_id, str) else None,
                "tldr": tldr.strip(),
                "illustration": illustration.strip(),
            }
        )
    return cleaned


def validate_python_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty Python source.")
    source = value.strip()
    if "val_bpb:" not in source or "peak_vram_mb:" not in source:
        raise ValueError("Generated train.py does not preserve the required summary lines.")
    return source


def validate_train_py_against_parent(
    config: OrchestratorConfig,
    parent_source: str,
    candidate_source: str,
) -> str:
    missing = [
        anchor
        for anchor in config.required_parent_anchors
        if anchor in parent_source and anchor not in candidate_source
    ]
    if missing:
        raise ValueError(
            "Generated train.py dropped required parent-contract anchors: "
            + ", ".join(repr(item) for item in missing)
        )

    introduced = [
        pattern
        for pattern in config.forbidden_new_patterns
        if re.search(pattern, candidate_source) and not re.search(pattern, parent_source)
    ]
    if introduced:
        raise ValueError(
            "Generated train.py introduced incompatible data/config assumptions: "
            + ", ".join(repr(item) for item in introduced)
        )
    return candidate_source
