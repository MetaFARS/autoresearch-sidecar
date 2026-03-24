from __future__ import annotations

"""Residual runtime emitted for autoresearch_program.

This module is standalone with respect to the DSL/compiler layers. It embeds:
- the fully bound role/phase prompts and contracts,
- the bound workflow with ref expressions already lowered,
- the host-op implementations actually used by the program.
"""

import asyncio
import json
import re
from typing import Any, Callable

import requests

PROGRAM_NAME = 'autoresearch_program'
PROGRAM = {'name': 'autoresearch_program',
 'roles': {'planner': {'name': 'planner',
                       'purpose': 'Propose the next batch of autoresearch ideas from the current '
                                  'tree state.',
                       'phases': [{'name': 'investigate',
                                   'purpose': 'Inspect prior experiments and collect explicit '
                                              'research notes.',
                                   'reads': ['snapshot'],
                                   'writes': ['research_notes'],
                                   'allow_tools': True,
                                   'allowed_tools': ['read_meta',
                                                     'read_code',
                                                     'read_stdout',
                                                     'read_stderr'],
                                   'system_prompt': 'The research system manages a hierarchical '
                                                    'tree of iterative experiments.\n'
                                                    'Each experiment is encapsulated as an Idea '
                                                    'object with the following fields:\n'
                                                    '- node_id (str): unique ID for the experiment '
                                                    'node.\n'
                                                    '- parent_id (str | None): ID of the base '
                                                    'experiment this extends.\n'
                                                    '- illustration (str): technical explanation '
                                                    'of the hypothesis and design.\n'
                                                    '- tldr (str): concise summary of the '
                                                    'proposal.\n'
                                                    '- metric (float | None): the primary '
                                                    'quantitative result. Larger is always '
                                                    'better.\n'
                                                    '- exit_code (int | None): 0 for successful '
                                                    'execution, non-zero for crashes.\n'
                                                    '- status (IdeaStatus): one of pending, '
                                                    'running, success, failed.\n'
                                                    'Role name: planner\n'
                                                    'Role purpose: Propose the next batch of '
                                                    'autoresearch ideas from the current tree '
                                                    'state.\n'
                                                    'State schema:\n'
                                                    '- snapshot (input): The current serialized '
                                                    'research tree snapshot.\n'
                                                    '- research_notes (working): Findings '
                                                    'collected from tool-assisted investigation.\n'
                                                    '- proposals (output): A list of proposal '
                                                    'objects with parent_id, tldr, illustration.\n'
                                                    'Current phase: investigate\n'
                                                    'Phase purpose: Inspect prior experiments and '
                                                    'collect explicit research notes.\n'
                                                    'Phase instructions:\n'
                                                    'You are investigating the current research '
                                                    'tree.\n'
                                                    'Use tools one call at a time to inspect '
                                                    'promising successful leaves and instructive '
                                                    'failures.\n'
                                                    'Summarize the strongest opportunities for '
                                                    'improvement, debugging, or exploration.\n'
                                                    'Do not output proposals yet. Output only '
                                                    'concise research notes.\n'
                                                    'Role invariants:\n'
                                                    '- Investigation and emission are separate '
                                                    'phases.\n'
                                                    '- All investigative actions must go through '
                                                    'the declared tools.\n'
                                                    '- The final proposal list must be valid JSON '
                                                    'with the required schema.\n'
                                                    '- Prefer improving strong successful leaves '
                                                    'or debugging informative failures.\n'
                                                    'Output contract:\n'
                                                    'Return only a concise block of research '
                                                    'notes. No JSON. No markdown fences.\n'
                                                    'Available tools:\n'
                                                    '- read_meta(node_id: str) -> str: Read the '
                                                    'metadata JSON for a research node.\n'
                                                    '- read_code(node_id: str) -> str: Read the '
                                                    'generated Python source code for a research '
                                                    'node.\n'
                                                    '- read_stdout(node_id: str) -> str: Read the '
                                                    'stdout log for a research node.\n'
                                                    '- read_stderr(node_id: str) -> str: Read the '
                                                    'stderr log for a research node.\n'
                                                    '\n'
                                                    'Tool protocol:\n'
                                                    'To call a tool, emit exactly one call inside '
                                                    '<tool></tool><stop>.\n'
                                                    'Example: '
                                                    '<tool>read_meta("2d28a7")</tool><stop>\n'
                                                    'The host will execute the tool and append the '
                                                    'result inside <result></result>.',
                                   'contract': {'name': 'research_notes',
                                                'instructions': 'Return only a concise block of '
                                                                'research notes. No JSON. No '
                                                                'markdown fences.',
                                                'parser_name': 'text',
                                                'validator_name': 'nonempty_text'},
                                   'field_descriptions': {'snapshot': 'The current serialized '
                                                                      'research tree snapshot.',
                                                          'research_notes': 'Findings collected '
                                                                            'from tool-assisted '
                                                                            'investigation.',
                                                          'proposals': 'A list of proposal objects '
                                                                       'with parent_id, tldr, '
                                                                       'illustration.'}},
                                  {'name': 'emit_proposals',
                                   'purpose': 'Transform the research notes into executable '
                                              'proposal objects.',
                                   'reads': ['snapshot', 'research_notes'],
                                   'writes': ['proposals'],
                                   'allow_tools': False,
                                   'allowed_tools': [],
                                   'system_prompt': 'The research system manages a hierarchical '
                                                    'tree of iterative experiments.\n'
                                                    'Each experiment is encapsulated as an Idea '
                                                    'object with the following fields:\n'
                                                    '- node_id (str): unique ID for the experiment '
                                                    'node.\n'
                                                    '- parent_id (str | None): ID of the base '
                                                    'experiment this extends.\n'
                                                    '- illustration (str): technical explanation '
                                                    'of the hypothesis and design.\n'
                                                    '- tldr (str): concise summary of the '
                                                    'proposal.\n'
                                                    '- metric (float | None): the primary '
                                                    'quantitative result. Larger is always '
                                                    'better.\n'
                                                    '- exit_code (int | None): 0 for successful '
                                                    'execution, non-zero for crashes.\n'
                                                    '- status (IdeaStatus): one of pending, '
                                                    'running, success, failed.\n'
                                                    'Role name: planner\n'
                                                    'Role purpose: Propose the next batch of '
                                                    'autoresearch ideas from the current tree '
                                                    'state.\n'
                                                    'State schema:\n'
                                                    '- snapshot (input): The current serialized '
                                                    'research tree snapshot.\n'
                                                    '- research_notes (working): Findings '
                                                    'collected from tool-assisted investigation.\n'
                                                    '- proposals (output): A list of proposal '
                                                    'objects with parent_id, tldr, illustration.\n'
                                                    'Current phase: emit_proposals\n'
                                                    'Phase purpose: Transform the research notes '
                                                    'into executable proposal objects.\n'
                                                    'Phase instructions:\n'
                                                    'Now produce the next batch of proposals.\n'
                                                    'Each proposal must name a valid parent node '
                                                    'when possible, provide a terse tldr, and give '
                                                    'a concrete illustration.\n'
                                                    'Do not include any commentary outside the '
                                                    'JSON list.\n'
                                                    'Role invariants:\n'
                                                    '- Investigation and emission are separate '
                                                    'phases.\n'
                                                    '- All investigative actions must go through '
                                                    'the declared tools.\n'
                                                    '- The final proposal list must be valid JSON '
                                                    'with the required schema.\n'
                                                    '- Prefer improving strong successful leaves '
                                                    'or debugging informative failures.\n'
                                                    'Output contract:\n'
                                                    'Return ONLY valid JSON with this exact '
                                                    'schema:\n'
                                                    '[\n'
                                                    '  {"parent_id": "...", "tldr": "...", '
                                                    '"illustration": "..."}\n'
                                                    ']\n'
                                                    'No markdown fences. No conversational text.\n'
                                                    'Tool use is disabled in this phase. Emit the '
                                                    'output directly.',
                                   'contract': {'name': 'proposal_list',
                                                'instructions': 'Return ONLY valid JSON with this '
                                                                'exact schema:\n'
                                                                '[\n'
                                                                '  {"parent_id": "...", "tldr": '
                                                                '"...", "illustration": "..."}\n'
                                                                ']\n'
                                                                'No markdown fences. No '
                                                                'conversational text.',
                                                'parser_name': 'json_list',
                                                'validator_name': 'proposal_list'},
                                   'field_descriptions': {'snapshot': 'The current serialized '
                                                                      'research tree snapshot.',
                                                          'research_notes': 'Findings collected '
                                                                            'from tool-assisted '
                                                                            'investigation.',
                                                          'proposals': 'A list of proposal objects '
                                                                       'with parent_id, tldr, '
                                                                       'illustration.'}}],
                       'required_inputs': ['snapshot'],
                       'state_fields': ['snapshot', 'research_notes', 'proposals'],
                       'output_fields': ['proposals']},
           'implementer': {'name': 'implementer',
                           'purpose': 'Implement one proposal as runnable Python code relative to '
                                      'a chosen parent node.',
                           'phases': [{'name': 'inspect_parent',
                                       'purpose': 'Study the parent node and prepare an explicit '
                                                  'implementation plan.',
                                       'reads': ['snapshot',
                                                 'node_id',
                                                 'parent_id',
                                                 'tldr',
                                                 'illustration'],
                                       'writes': ['implementation_plan'],
                                       'allow_tools': True,
                                       'allowed_tools': ['read_meta',
                                                         'read_code',
                                                         'read_stdout',
                                                         'read_stderr'],
                                       'system_prompt': 'The research system manages a '
                                                        'hierarchical tree of iterative '
                                                        'experiments.\n'
                                                        'Each experiment is encapsulated as an '
                                                        'Idea object with the following fields:\n'
                                                        '- node_id (str): unique ID for the '
                                                        'experiment node.\n'
                                                        '- parent_id (str | None): ID of the base '
                                                        'experiment this extends.\n'
                                                        '- illustration (str): technical '
                                                        'explanation of the hypothesis and '
                                                        'design.\n'
                                                        '- tldr (str): concise summary of the '
                                                        'proposal.\n'
                                                        '- metric (float | None): the primary '
                                                        'quantitative result. Larger is always '
                                                        'better.\n'
                                                        '- exit_code (int | None): 0 for '
                                                        'successful execution, non-zero for '
                                                        'crashes.\n'
                                                        '- status (IdeaStatus): one of pending, '
                                                        'running, success, failed.\n'
                                                        'Role name: implementer\n'
                                                        'Role purpose: Implement one proposal as '
                                                        'runnable Python code relative to a chosen '
                                                        'parent node.\n'
                                                        'State schema:\n'
                                                        '- snapshot (input): The current '
                                                        'serialized research tree snapshot.\n'
                                                        '- node_id (input): The node being '
                                                        'implemented.\n'
                                                        '- parent_id (input): The parent node for '
                                                        'consistency and inheritance.\n'
                                                        '- tldr (input): Short summary of the '
                                                        'target idea.\n'
                                                        '- illustration (input): Detailed '
                                                        'explanation of the target idea.\n'
                                                        '- implementation_plan (working): A '
                                                        'concrete implementation plan grounded in '
                                                        'the parent node.\n'
                                                        '- python_source (output): Runnable Python '
                                                        'source code for the node.\n'
                                                        'Current phase: inspect_parent\n'
                                                        'Phase purpose: Study the parent node and '
                                                        'prepare an explicit implementation plan.\n'
                                                        'Phase instructions:\n'
                                                        'Inspect the parent code, metadata, and '
                                                        'logs as needed.\n'
                                                        'Work out how this node should differ from '
                                                        'its parent while remaining executable.\n'
                                                        'Output only an implementation plan, not '
                                                        'code.\n'
                                                        'Role invariants:\n'
                                                        '- Inspection and emission are separate '
                                                        'phases.\n'
                                                        '- Implementation must remain grounded in '
                                                        'the selected parent node.\n'
                                                        '- The final program must print exactly '
                                                        'one metric line of the form Metric: '
                                                        '<value>.\n'
                                                        'Output contract:\n'
                                                        'Return only a concise implementation '
                                                        'plan. No code fences. No source code '
                                                        'yet.\n'
                                                        'Available tools:\n'
                                                        '- read_meta(node_id: str) -> str: Read '
                                                        'the metadata JSON for a research node.\n'
                                                        '- read_code(node_id: str) -> str: Read '
                                                        'the generated Python source code for a '
                                                        'research node.\n'
                                                        '- read_stdout(node_id: str) -> str: Read '
                                                        'the stdout log for a research node.\n'
                                                        '- read_stderr(node_id: str) -> str: Read '
                                                        'the stderr log for a research node.\n'
                                                        '\n'
                                                        'Tool protocol:\n'
                                                        'To call a tool, emit exactly one call '
                                                        'inside <tool></tool><stop>.\n'
                                                        'Example: '
                                                        '<tool>read_meta("2d28a7")</tool><stop>\n'
                                                        'The host will execute the tool and append '
                                                        'the result inside <result></result>.',
                                       'contract': {'name': 'implementation_plan',
                                                    'instructions': 'Return only a concise '
                                                                    'implementation plan. No code '
                                                                    'fences. No source code yet.',
                                                    'parser_name': 'text',
                                                    'validator_name': 'nonempty_text'},
                                       'field_descriptions': {'snapshot': 'The current serialized '
                                                                          'research tree snapshot.',
                                                              'node_id': 'The node being '
                                                                         'implemented.',
                                                              'parent_id': 'The parent node for '
                                                                           'consistency and '
                                                                           'inheritance.',
                                                              'tldr': 'Short summary of the target '
                                                                      'idea.',
                                                              'illustration': 'Detailed '
                                                                              'explanation of the '
                                                                              'target idea.',
                                                              'implementation_plan': 'A concrete '
                                                                                     'implementation '
                                                                                     'plan '
                                                                                     'grounded in '
                                                                                     'the parent '
                                                                                     'node.',
                                                              'python_source': 'Runnable Python '
                                                                               'source code for '
                                                                               'the node.'}},
                                      {'name': 'emit_code',
                                       'purpose': 'Emit the final runnable source code.',
                                       'reads': ['snapshot',
                                                 'node_id',
                                                 'parent_id',
                                                 'tldr',
                                                 'illustration',
                                                 'implementation_plan'],
                                       'writes': ['python_source'],
                                       'allow_tools': False,
                                       'allowed_tools': [],
                                       'system_prompt': 'The research system manages a '
                                                        'hierarchical tree of iterative '
                                                        'experiments.\n'
                                                        'Each experiment is encapsulated as an '
                                                        'Idea object with the following fields:\n'
                                                        '- node_id (str): unique ID for the '
                                                        'experiment node.\n'
                                                        '- parent_id (str | None): ID of the base '
                                                        'experiment this extends.\n'
                                                        '- illustration (str): technical '
                                                        'explanation of the hypothesis and '
                                                        'design.\n'
                                                        '- tldr (str): concise summary of the '
                                                        'proposal.\n'
                                                        '- metric (float | None): the primary '
                                                        'quantitative result. Larger is always '
                                                        'better.\n'
                                                        '- exit_code (int | None): 0 for '
                                                        'successful execution, non-zero for '
                                                        'crashes.\n'
                                                        '- status (IdeaStatus): one of pending, '
                                                        'running, success, failed.\n'
                                                        'Role name: implementer\n'
                                                        'Role purpose: Implement one proposal as '
                                                        'runnable Python code relative to a chosen '
                                                        'parent node.\n'
                                                        'State schema:\n'
                                                        '- snapshot (input): The current '
                                                        'serialized research tree snapshot.\n'
                                                        '- node_id (input): The node being '
                                                        'implemented.\n'
                                                        '- parent_id (input): The parent node for '
                                                        'consistency and inheritance.\n'
                                                        '- tldr (input): Short summary of the '
                                                        'target idea.\n'
                                                        '- illustration (input): Detailed '
                                                        'explanation of the target idea.\n'
                                                        '- implementation_plan (working): A '
                                                        'concrete implementation plan grounded in '
                                                        'the parent node.\n'
                                                        '- python_source (output): Runnable Python '
                                                        'source code for the node.\n'
                                                        'Current phase: emit_code\n'
                                                        'Phase purpose: Emit the final runnable '
                                                        'source code.\n'
                                                        'Phase instructions:\n'
                                                        'Generate the full runnable Python source '
                                                        'for the target node.\n'
                                                        'The program must execute end-to-end and '
                                                        'print the final quantitative result on '
                                                        'exactly one line as: Metric: <value>\n'
                                                        'Do not include any commentary or markdown '
                                                        'fences.\n'
                                                        'Role invariants:\n'
                                                        '- Inspection and emission are separate '
                                                        'phases.\n'
                                                        '- Implementation must remain grounded in '
                                                        'the selected parent node.\n'
                                                        '- The final program must print exactly '
                                                        'one metric line of the form Metric: '
                                                        '<value>.\n'
                                                        'Output contract:\n'
                                                        'Return ONLY runnable Python source code. '
                                                        'No markdown fences. No explanations.\n'
                                                        'Tool use is disabled in this phase. Emit '
                                                        'the output directly.',
                                       'contract': {'name': 'python_source',
                                                    'instructions': 'Return ONLY runnable Python '
                                                                    'source code. No markdown '
                                                                    'fences. No explanations.',
                                                    'parser_name': 'code',
                                                    'validator_name': 'python_source'},
                                       'field_descriptions': {'snapshot': 'The current serialized '
                                                                          'research tree snapshot.',
                                                              'node_id': 'The node being '
                                                                         'implemented.',
                                                              'parent_id': 'The parent node for '
                                                                           'consistency and '
                                                                           'inheritance.',
                                                              'tldr': 'Short summary of the target '
                                                                      'idea.',
                                                              'illustration': 'Detailed '
                                                                              'explanation of the '
                                                                              'target idea.',
                                                              'implementation_plan': 'A concrete '
                                                                                     'implementation '
                                                                                     'plan '
                                                                                     'grounded in '
                                                                                     'the parent '
                                                                                     'node.',
                                                              'python_source': 'Runnable Python '
                                                                               'source code for '
                                                                               'the node.'}}],
                           'required_inputs': ['illustration',
                                               'node_id',
                                               'parent_id',
                                               'snapshot',
                                               'tldr'],
                           'state_fields': ['snapshot',
                                            'node_id',
                                            'parent_id',
                                            'tldr',
                                            'illustration',
                                            'implementation_plan',
                                            'python_source'],
                           'output_fields': ['python_source']}},
 'workflow': {'name': 'autoresearch_iteration',
              'state_slots': ['current_node',
                              'implementer_run',
                              'new_node_ids',
                              'pending_node_ids',
                              'planner_run',
                              'snapshot'],
              'steps': [{'kind': 'host',
                         'name': 'capture_snapshot',
                         'op_name': 'capture_snapshot',
                         'arg_expr': {},
                         'save_as': 'snapshot'},
                        {'kind': 'role',
                         'name': 'planner',
                         'role_name': 'planner',
                         'binding_expr': {'snapshot': {'$ref': 'snapshot'}},
                         'save_as': 'planner_run'},
                        {'kind': 'host',
                         'name': 'materialize_proposals',
                         'op_name': 'materialize_proposals',
                         'arg_expr': {'proposals': {'$ref': 'planner_run.proposals'}},
                         'save_as': 'new_node_ids'},
                        {'kind': 'host',
                         'name': 'list_unimplemented_nodes',
                         'op_name': 'list_unimplemented_nodes',
                         'arg_expr': {},
                         'save_as': 'pending_node_ids'},
                        {'kind': 'foreach',
                         'name': 'implement_each_pending_node',
                         'items_expr': {'$ref': 'pending_node_ids'},
                         'item_name': 'current_node_id',
                         'body': [{'kind': 'host',
                                   'name': 'load_current_node',
                                   'op_name': 'load_node_record',
                                   'arg_expr': {'node_id': {'$ref': 'current_node_id'}},
                                   'save_as': 'current_node'},
                                  {'kind': 'role',
                                   'name': 'implementer',
                                   'role_name': 'implementer',
                                   'binding_expr': {'snapshot': {'$ref': 'snapshot'},
                                                    'node_id': {'$ref': 'current_node_id'},
                                                    'parent_id': {'$ref': 'current_node.parent_id'},
                                                    'tldr': {'$ref': 'current_node.tldr'},
                                                    'illustration': {'$ref': 'current_node.illustration'}},
                                   'save_as': 'implementer_run'},
                                  {'kind': 'host',
                                   'name': 'write_generated_source',
                                   'op_name': 'write_generated_source',
                                   'arg_expr': {'node_id': {'$ref': 'current_node_id'},
                                                'source': {'$ref': 'implementer_run.python_source'}},
                                   'save_as': None}]},
                        {'kind': 'host',
                         'name': 'run_pending_experiments',
                         'op_name': 'run_pending_experiments',
                         'arg_expr': {},
                         'save_as': None}]}}


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


def parse_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError("Expected non-empty text output.")
    return text


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*(.*?)\s*```$", re.DOTALL)


def parse_code(raw: str) -> str:
    text = raw.strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    if not text:
        raise ValueError("Expected non-empty source code.")
    return text


_JSON_LIST_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_DICT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_json_list(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_LIST_RE.search(text)
        if match is None:
            raise
        return json.loads(match.group(1))


def parse_json_dict(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_DICT_RE.search(text)
        if match is None:
            raise
        return json.loads(match.group(1))


def identity(value: Any) -> Any:
    return value


def validate_nonempty_text(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty string.")
    return value


_REQUIRED_PROPOSAL_KEYS = {"parent_id", "tldr", "illustration"}


def validate_proposal_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Planner output must be a list.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Proposal {index} must be an object.")
        missing = _REQUIRED_PROPOSAL_KEYS - set(item)
        if missing:
            raise ValueError(f"Proposal {index} is missing keys: {sorted(missing)}")
        parent_id = item.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"Proposal {index} has invalid parent_id: {parent_id!r}")
        tldr = item.get("tldr")
        illustration = item.get("illustration")
        if not isinstance(tldr, str) or not tldr.strip():
            raise ValueError(f"Proposal {index} has empty tldr.")
        if not isinstance(illustration, str) or not illustration.strip():
            raise ValueError(f"Proposal {index} has empty illustration.")
        normalized.append(
            {
                "parent_id": parent_id,
                "tldr": tldr.strip(),
                "illustration": illustration.strip(),
            }
        )
    return normalized


_METRIC_HINT_RE = re.compile(r"Metric\s*:")


def validate_python_source(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Generated source must be a non-empty string.")
    source = value.strip()
    compile(source, "<generated.py>", "exec")
    if not _METRIC_HINT_RE.search(source):
        raise ValueError("Generated code must contain a 'Metric:' print path.")
    return source


PARSERS = {
    "text": parse_text,
    "code": parse_code,
    "json_list": parse_json_list,
    "json_dict": parse_json_dict,
}

VALIDATORS = {
    None: identity,
    "nonempty_text": validate_nonempty_text,
    "proposal_list": validate_proposal_list,
    "python_source": validate_python_source,
}

HOST_OPS = {
    'capture_snapshot': capture_snapshot,
    'materialize_proposals': materialize_proposals,
    'list_unimplemented_nodes': list_unimplemented_nodes,
    'load_node_record': load_node_record,
    'write_generated_source': write_generated_source,
    'run_pending_experiments': run_pending_experiments,
}


class KappaClient:
    def __init__(self, base_url: str, api_key: str, model: str, debug: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.debug = debug

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload), timeout=300).json()
        if "choices" not in response:
            raise RuntimeError(f"API error: {response}")
        return response["choices"][0]["message"]["content"]


class ToolCall:
    def __init__(self, name: str, argument: str, raw_text: str):
        self.name = name
        self.argument = argument
        self.raw_text = raw_text


class ToolHost:
    TOOL_CALL_RE = re.compile(r"<tool>\s*(\w+)\([\"\'](.*?)[\"\']\)\s*</tool>")

    def __init__(self, world: Any, debug: bool = False):
        self.world = world
        self.debug = debug
        self.handlers: dict[str, Callable[[str], str]] = {
            "read_meta": self.world.read_meta,
            "read_code": self.world.read_code,
            "read_stdout": self.world.read_stdout,
            "read_stderr": self.world.read_stderr,
        }

    def parse_tool_call(self, raw: str) -> ToolCall | None:
        match = self.TOOL_CALL_RE.search(raw)
        if not match:
            return None
        return ToolCall(name=match.group(1), argument=match.group(2), raw_text=match.group(0).strip())

    def execute(self, tool_name: str, argument: str) -> str:
        if tool_name not in self.handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.handlers[tool_name](argument)


class RoleRunner:
    TOOL_BLOCK_RE = re.compile(r"(.*?<tool>\s*\w+\([\"\'].*?[\"\']\)\s*</tool><stop>)", re.DOTALL)

    def __init__(self, client: KappaClient, tool_host: ToolHost, debug: bool = False):
        self.client = client
        self.tool_host = tool_host
        self.debug = debug

    async def run(self, role: dict[str, Any], initial_state: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._run_sync, role, initial_state)

    def _run_sync(self, role: dict[str, Any], initial_state: dict[str, Any]) -> dict[str, Any]:
        local_state = dict(initial_state)
        for phase in role["phases"]:
            history = [
                {"role": "system", "content": phase["system_prompt"]},
                {"role": "user", "content": self._render_user_prompt(phase["reads"], phase["field_descriptions"], local_state)},
            ]
            while True:
                raw = self.client.complete(history, stop=["<stop>"])
                if "</tool>" in raw and "<stop>" not in raw:
                    raw += "<stop>"
                match = self.TOOL_BLOCK_RE.search(raw)
                if match:
                    if not phase["allow_tools"]:
                        raise PermissionError(f"Phase {phase['name']} does not allow tools.")
                    assistant_message = match.group(1).strip()
                    history.append({"role": "assistant", "content": assistant_message})
                    invocation = self.tool_host.parse_tool_call(assistant_message)
                    if invocation is None:
                        raise ValueError(f"Malformed tool call: {assistant_message}")
                    if invocation.name not in phase["allowed_tools"]:
                        raise PermissionError(f"Phase {phase['name']} cannot call {invocation.name}")
                    result = self.tool_host.execute(invocation.name, invocation.argument)
                    history.append({"role": "user", "content": f"<result>\n{result}\n</result>"})
                    if self.debug:
                        print(f"\n[tool] {invocation.name}({invocation.argument!r})\n{result}\n", flush=True)
                    continue
                contract = phase["contract"]
                parsed = PARSERS[contract["parser_name"]](raw)
                validated = VALIDATORS[contract["validator_name"]](parsed)
                self._write_phase_output(local_state, phase["writes"], validated)
                if self.debug:
                    print(f"\n[phase {phase['name']} output]\n{validated!r}\n", flush=True)
                break
        return local_state

    def _render_user_prompt(self, reads: list[str], field_descriptions: dict[str, str], state: dict[str, Any]) -> str:
        lines = ["Phase state bindings:"]
        for field_name in reads:
            description = field_descriptions.get(field_name, "")
            value = state.get(field_name)
            if description:
                lines.append(f"\n[{field_name}] {description}\n{self._serialize(value)}")
            else:
                lines.append(f"\n[{field_name}]\n{self._serialize(value)}")
        lines.append("\nProduce the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)

    def _write_phase_output(self, state: dict[str, Any], writes: list[str], value: Any) -> None:
        if len(writes) == 1:
            state[writes[0]] = value
            return
        if not isinstance(value, dict):
            raise ValueError(f"Phase writes {writes!r} require a dict output, got {type(value)!r}")
        for key in writes:
            if key not in value:
                raise ValueError(f"Phase output is missing expected field {key!r}")
            state[key] = value[key]


class ResidualRuntimeSession:
    def __init__(self, world: Any, client: KappaClient, debug: bool = False):
        self.program = PROGRAM
        self.world = world
        self.client = client
        self.debug = debug
        self.role_runner = RoleRunner(client, ToolHost(world, debug=debug), debug=debug)

    async def run_workflow(self, memory: dict[str, Any] | None = None) -> dict[str, Any]:
        state = dict(memory or {})
        for step in self.program["workflow"]["steps"]:
            await self._run_step(step, state)
        return state

    async def _run_step(self, step: dict[str, Any], state: dict[str, Any]) -> None:
        kind = step["kind"]
        if kind == "host":
            args = resolve_expr(state, step["arg_expr"])
            result = HOST_OPS[step["op_name"]](self.world, state, args)
            if asyncio.iscoroutine(result):
                result = await result
            if step["save_as"] is not None:
                state[step["save_as"]] = result
            if self.debug:
                print(f"[host-step] {step['name']} -> {step['save_as']}: {result!r}")
            return
        if kind == "role":
            bindings = resolve_expr(state, step["binding_expr"])
            role = self.program["roles"][step["role_name"]]
            result = await self.role_runner.run(role, bindings)
            state[step["save_as"]] = result
            if self.debug:
                print(f"[role-step] {step['name']} -> {step['save_as']}: keys={sorted(result)}")
            return
        if kind == "foreach":
            items = resolve_expr(state, step["items_expr"])
            if not isinstance(items, list):
                raise TypeError(f"Foreach step {step['name']} expected a list, got {type(items)!r}")
            for item in items:
                state[step["item_name"]] = item
                for child in step["body"]:
                    await self._run_step(child, state)
            return
        raise TypeError(f"Unknown compiled step kind: {kind!r}")


def resolve_path(state: dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in path.split('.'):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def resolve_expr(state: dict[str, Any], expr: Any) -> Any:
    if isinstance(expr, dict) and set(expr) == {"$ref"}:
        return resolve_path(state, expr["$ref"])
    if isinstance(expr, dict):
        return {key: resolve_expr(state, value) for key, value in expr.items()}
    if isinstance(expr, list):
        return [resolve_expr(state, item) for item in expr]
    return expr


def describe_program() -> str:
    lines = [f"ResidualProgram<{PROGRAM_NAME}>", "", "Roles:"]
    for role in PROGRAM["roles"].values():
        lines.append(f"- {role['name']}: inputs={role['required_inputs']} outputs={role['output_fields']}")
        for phase in role["phases"]:
            lines.append(
                f"    phase {phase['name']}: reads={phase['reads']} writes={phase['writes']} tools={phase['allowed_tools']}"
            )
    lines.extend(["", "Workflow slots:"])
    for slot in PROGRAM["workflow"]["state_slots"]:
        lines.append(f"- {slot}")
    return "\n".join(lines)


def create_session(world: Any, client: KappaClient, debug: bool = False) -> ResidualRuntimeSession:
    return ResidualRuntimeSession(world=world, client=client, debug=debug)


__all__ = [
    "PROGRAM_NAME",
    "PROGRAM",
    "KappaClient",
    "ResidualRuntimeSession",
    "create_session",
    "describe_program",
]
