from __future__ import annotations

from typing import Any

from dsl import HostOpSpec
from world import ResearchWorld


def capture_snapshot(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> str:
    return world.snapshot()


def materialize_proposals(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> list[str]:
    proposals = args["proposals"]
    node_ids: list[str] = []
    for item in proposals:
        node = world.add_idea(
            parent_id=item.get("parent_id"),
            tldr=item.get("tldr", ""),
            illustration=item.get("illustration", ""),
        )
        node_ids.append(node.node_id)
    return node_ids


def list_unimplemented_nodes(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> list[str]:
    return [node.node_id for node in world.pending_nodes() if not world.has_code(node.node_id)]


def load_node_record(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> dict[str, object]:
    return world.get_node_record(args["node_id"])


def write_generated_source(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> None:
    world.write_code(args["node_id"], args["source"])


async def run_pending_experiments(world: ResearchWorld, memory: dict[str, Any], args: dict[str, Any]) -> None:
    await world.run_pending_experiments()


HOST_OPS = {
    "capture_snapshot": HostOpSpec(
        name="capture_snapshot",
        doc="Serialize the current research tree snapshot.",
        fn=capture_snapshot,
    ),
    "materialize_proposals": HostOpSpec(
        name="materialize_proposals",
        doc="Create Idea nodes from planner proposals.",
        fn=materialize_proposals,
    ),
    "list_unimplemented_nodes": HostOpSpec(
        name="list_unimplemented_nodes",
        doc="List PENDING nodes that still need code generation.",
        fn=list_unimplemented_nodes,
    ),
    "load_node_record": HostOpSpec(
        name="load_node_record",
        doc="Load the current node metadata as a structured record.",
        fn=load_node_record,
    ),
    "write_generated_source": HostOpSpec(
        name="write_generated_source",
        doc="Write generated Python source into the node workspace.",
        fn=write_generated_source,
    ),
    "run_pending_experiments": HostOpSpec(
        name="run_pending_experiments",
        doc="Execute all pending experiments on the available GPUs.",
        fn=run_pending_experiments,
    ),
}
