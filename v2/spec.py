from __future__ import annotations

from dsl import (
    FieldSpec,
    ForeachStepSpec,
    HostStepSpec,
    OutputContractSpec,
    PhaseSpec,
    ProgramSpec,
    RoleSpec,
    RoleStepSpec,
    StateSchema,
    ToolSpec,
    WorkflowSpec,
    ref,
)
from host_ops import HOST_OPS


RESEARCH_SYSTEM_CONTEXT = """
The research system manages a hierarchical tree of iterative experiments.
Each experiment is encapsulated as an Idea object with the following fields:
- node_id (str): unique ID for the experiment node.
- parent_id (str | None): ID of the base experiment this extends.
- illustration (str): technical explanation of the hypothesis and design.
- tldr (str): concise summary of the proposal.
- metric (float | None): the primary quantitative result. Larger is always better.
- exit_code (int | None): 0 for successful execution, non-zero for crashes.
- status (IdeaStatus): one of pending, running, success, failed.
""".strip()


TOOLS = {
    "read_meta": ToolSpec(
        name="read_meta",
        signature="read_meta(node_id: str) -> str",
        description="Read the metadata JSON for a research node.",
    ),
    "read_code": ToolSpec(
        name="read_code",
        signature="read_code(node_id: str) -> str",
        description="Read the generated Python source code for a research node.",
    ),
    "read_stdout": ToolSpec(
        name="read_stdout",
        signature="read_stdout(node_id: str) -> str",
        description="Read the stdout log for a research node.",
    ),
    "read_stderr": ToolSpec(
        name="read_stderr",
        signature="read_stderr(node_id: str) -> str",
        description="Read the stderr log for a research node.",
    ),
}


PLANNER_ROLE = RoleSpec(
    name="planner",
    purpose="Propose the next batch of autoresearch ideas from the current tree state.",
    system_context=RESEARCH_SYSTEM_CONTEXT,
    state_schema=StateSchema(
        fields=[
            FieldSpec("snapshot", "The current serialized research tree snapshot.", "input"),
            FieldSpec("research_notes", "Findings collected from tool-assisted investigation.", "working"),
            FieldSpec("proposals", "A list of proposal objects with parent_id, tldr, illustration.", "output"),
        ]
    ),
    invariants=(
        "Investigation and emission are separate phases.",
        "All investigative actions must go through the declared tools.",
        "The final proposal list must be valid JSON with the required schema.",
        "Prefer improving strong successful leaves or debugging informative failures.",
    ),
    phases=(
        PhaseSpec(
            name="investigate",
            purpose="Inspect prior experiments and collect explicit research notes.",
            reads=("snapshot",),
            writes=("research_notes",),
            allow_tools=True,
            allowed_tools=("read_meta", "read_code", "read_stdout", "read_stderr"),
            instructions="""
You are investigating the current research tree.
Use tools one call at a time to inspect promising successful leaves and instructive failures.
Summarize the strongest opportunities for improvement, debugging, or exploration.
Do not output proposals yet. Output only concise research notes.
""".strip(),
            contract=OutputContractSpec(
                name="research_notes",
                parser="text",
                validator="nonempty_text",
                instructions="Return only a concise block of research notes. No JSON. No markdown fences.",
            ),
        ),
        PhaseSpec(
            name="emit_proposals",
            purpose="Transform the research notes into executable proposal objects.",
            reads=("snapshot", "research_notes"),
            writes=("proposals",),
            allow_tools=False,
            instructions="""
Now produce the next batch of proposals.
Each proposal must name a valid parent node when possible, provide a terse tldr, and give a concrete illustration.
Do not include any commentary outside the JSON list.
""".strip(),
            contract=OutputContractSpec(
                name="proposal_list",
                parser="json_list",
                validator="proposal_list",
                instructions='''Return ONLY valid JSON with this exact schema:\n[\n  {"parent_id": "...", "tldr": "...", "illustration": "..."}\n]\nNo markdown fences. No conversational text.''',
            ),
        ),
    ),
)


IMPLEMENTER_ROLE = RoleSpec(
    name="implementer",
    purpose="Implement one proposal as runnable Python code relative to a chosen parent node.",
    system_context=RESEARCH_SYSTEM_CONTEXT,
    state_schema=StateSchema(
        fields=[
            FieldSpec("snapshot", "The current serialized research tree snapshot.", "input"),
            FieldSpec("node_id", "The node being implemented.", "input"),
            FieldSpec("parent_id", "The parent node for consistency and inheritance.", "input"),
            FieldSpec("tldr", "Short summary of the target idea.", "input"),
            FieldSpec("illustration", "Detailed explanation of the target idea.", "input"),
            FieldSpec("implementation_plan", "A concrete implementation plan grounded in the parent node.", "working"),
            FieldSpec("python_source", "Runnable Python source code for the node.", "output"),
        ]
    ),
    invariants=(
        "Inspection and emission are separate phases.",
        "Implementation must remain grounded in the selected parent node.",
        "The final program must print exactly one metric line of the form Metric: <value>.",
    ),
    phases=(
        PhaseSpec(
            name="inspect_parent",
            purpose="Study the parent node and prepare an explicit implementation plan.",
            reads=("snapshot", "node_id", "parent_id", "tldr", "illustration"),
            writes=("implementation_plan",),
            allow_tools=True,
            allowed_tools=("read_meta", "read_code", "read_stdout", "read_stderr"),
            instructions="""
Inspect the parent code, metadata, and logs as needed.
Work out how this node should differ from its parent while remaining executable.
Output only an implementation plan, not code.
""".strip(),
            contract=OutputContractSpec(
                name="implementation_plan",
                parser="text",
                validator="nonempty_text",
                instructions="Return only a concise implementation plan. No code fences. No source code yet.",
            ),
        ),
        PhaseSpec(
            name="emit_code",
            purpose="Emit the final runnable source code.",
            reads=("snapshot", "node_id", "parent_id", "tldr", "illustration", "implementation_plan"),
            writes=("python_source",),
            allow_tools=False,
            instructions="""
Generate the full runnable Python source for the target node.
The program must execute end-to-end and print the final quantitative result on exactly one line as: Metric: <value>
Do not include any commentary or markdown fences.
""".strip(),
            contract=OutputContractSpec(
                name="python_source",
                parser="code",
                validator="python_source",
                instructions="Return ONLY runnable Python source code. No markdown fences. No explanations.",
            ),
        ),
    ),
)


AUTORESEARCH_WORKFLOW = WorkflowSpec(
    name="autoresearch_iteration",
    steps=(
        HostStepSpec(
            name="capture_snapshot",
            op="capture_snapshot",
            save_as="snapshot",
        ),
        RoleStepSpec(
            name="planner",
            role="planner",
            bindings={"snapshot": ref("snapshot")},
            save_as="planner_run",
        ),
        HostStepSpec(
            name="materialize_proposals",
            op="materialize_proposals",
            args={"proposals": ref("planner_run.proposals")},
            save_as="new_node_ids",
        ),
        HostStepSpec(
            name="list_unimplemented_nodes",
            op="list_unimplemented_nodes",
            save_as="pending_node_ids",
        ),
        ForeachStepSpec(
            name="implement_each_pending_node",
            items=ref("pending_node_ids"),
            item_name="current_node_id",
            body=(
                HostStepSpec(
                    name="load_current_node",
                    op="load_node_record",
                    args={"node_id": ref("current_node_id")},
                    save_as="current_node",
                ),
                RoleStepSpec(
                    name="implementer",
                    role="implementer",
                    bindings={
                        "snapshot": ref("snapshot"),
                        "node_id": ref("current_node_id"),
                        "parent_id": ref("current_node.parent_id"),
                        "tldr": ref("current_node.tldr"),
                        "illustration": ref("current_node.illustration"),
                    },
                    save_as="implementer_run",
                ),
                HostStepSpec(
                    name="write_generated_source",
                    op="write_generated_source",
                    args={
                        "node_id": ref("current_node_id"),
                        "source": ref("implementer_run.python_source"),
                    },
                ),
            ),
        ),
        HostStepSpec(
            name="run_pending_experiments",
            op="run_pending_experiments",
        ),
    ),
)


AUTORESEARCH_PROGRAM = ProgramSpec(
    name="autoresearch_program",
    roles={
        "planner": PLANNER_ROLE,
        "implementer": IMPLEMENTER_ROLE,
    },
    host_ops=HOST_OPS,
    tools=TOOLS,
    workflow=AUTORESEARCH_WORKFLOW,
)
