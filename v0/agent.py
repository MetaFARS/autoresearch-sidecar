import json
import re
import os
import requests
from typing import Any, List, Dict, Optional
from pathlib import Path

class Agent:

    RESEARCH_SYSTEM_CONTEXT = """
The research system manages a hierarchical tree of iterative experiments.
Each experiment is encapsulates an 'Idea' object with core metadata:
- node_id (str): Unique hash for the experiment.
- parent_id (str | None): ID of the base experiment this extends.
- illustration (str): Technical explanation of the hypothesis and design.
- tldr (str): Concise summary of the proposal.
- metric (float): The primary quantitative result. **Larger is always better.**
- exit_code (int | None): 0 for successful execution, non-zero for crashes.
- status (IdeaStatus): Lifecycle state: 'pending', 'running', 'success', or 'failed'.
"""

    TOOL_DESCRIPTIONS = """
You are equipped with the following investigative tools:
1. read_meta(node_id: str) -> str: Retrieves the metadata.
2. read_code(node_id: str) -> str: Retrieves the implementation source code.
3. read_stdout(node_id: str) -> str: Retrieves the execution logs.
4. read_stderr(node_id: str) -> str: Retrieves error logs.

Protocol for Tool Use:
To call a tool, wrap the function call in <tool></tool><stop> tags. The system will automatically execute the tool after seeing the "<stop>" tag, capture the output, and inject it back into your context wrapped in <result></result> tags.

For example:
Assistant: I need to understand the baseline configuration before proposing v0 modifications.
<tool>read_meta("2d28a7")</tool><stop>
User: <result>{ "node_id": "2d28a7", ... }</result>
Assistant: I noticed that experiment "6438da" failed. I need to check the error logs to diagnose the issue.
<tool>read_stderr("6438da")</tool><stop>
User: <result>Traceback (most recent call last): ... ModuleNotFoundError: No module named 'torch'</result>
"""

    def __init__(self, base_url, api_key, model, debug_mode=False):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.debug_mode = debug_mode

    def chat_with_history(self, history):
        while True:
            payload = {
                "model": self.model, 
                "messages": history,
                "stop": ["<stop>"] 
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post(self.base_url, headers=headers, data=json.dumps(payload)).json()
            
            if "choices" not in resp:
                raise RuntimeError(f"API Error: {resp}")
                
            raw = resp["choices"][0]["message"]["content"]
            
            if "</tool>" in raw and "<stop>" not in raw:
                raw += "<stop>"
            
            match = re.search(r"(.*?<tool>\s*\w+\([\"\'].*?[\"\']\)\s*</tool><stop>)", raw, re.DOTALL)
            
            if not match:
                if self.debug_mode:
                    print(f"\n\033[94m[Assistant Final Output]:\n{raw}\033[0m\n", flush=True)
                history.append({"role": "assistant", "content": raw})
                return raw
                
            clean_raw = match.group(1).strip()
            
            if self.debug_mode:
                print(f"\n\033[94m[Assistant]:\n{clean_raw}\033[0m\n", flush=True)
                
            history.append({"role": "assistant", "content": clean_raw})
            
            action_match = re.search(r"<tool>\s*(\w+)\([\"\'](.*?)[\"\']\)\s*</tool>", clean_raw)
            tool, node = action_match.groups()
            
            mapping = {
                "read_meta": "meta.json", 
                "read_code": "main.py", 
                "read_stdout": "stdout.log", 
                "read_stderr": "stderr.log"
            }
            path = Path("namespace") / node / mapping.get(tool, "")
            result = path.read_text() if path.exists() else f"Error: {tool} failed, file not found."
            
            if self.debug_mode:
                print(f"\n\033[92m[Tool Result]:\n{result}\033[0m\n", flush=True)
                
            history.append({"role": "user", "content": f"<result>\n{result}\n</result>"})

    def generate_ideas(self, snapshot):
        system_prompt = (
            f"{self.RESEARCH_SYSTEM_CONTEXT}\n\n"
            f"{self.TOOL_DESCRIPTIONS}\n\n"
            "Act as a World-class AI Researcher to propose experimental ideas that maximize the target metric. You must strictly follow these constraints:\n"
            "1. Investigation: Use tools to analyze previous experiments and identify areas for improvement. You MUST call tools one by one.\n"
            "2. Parent Selection: You can choose ANY successful leaf node as your `parent_id` to iterate and further improve the metric, or choose ANY failed leaf node to debug and fix its implementation.\n"
            "3. Output Format: Once tool calls are complete, output ONLY a valid JSON list of proposals.\n"
            "4. Strict Schema: The JSON must exactly match this structure: [{\"parent_id\": \"...\", \"tldr\": \"...\", \"illustration\": \"...\"}]. Strictly NO markdown formatting (e.g., ```) and NO conversational text."
        )
        
        history = [{"role": "system", "content": system_prompt}, 
                   {"role": "user", "content": f"Snapshot:\n{snapshot}\n\nPropose v0 ideas."}]
        
        final_output = self.chat_with_history(history)
        match = re.search(r"(\[.*\])", final_output, re.DOTALL)
        return json.loads(match.group(1)) if match else []

    def implement(self, snapshot, node_id):
        meta_path = Path("namespace") / node_id / "meta.json"
        idea_data = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        tldr = idea_data.get("tldr", "")
        illustration = idea_data.get("illustration", "")

        system_prompt = (
            f"{self.RESEARCH_SYSTEM_CONTEXT}\n\n"
            f"{self.TOOL_DESCRIPTIONS}\n\n"
            "Act as an expert ML Engineer to implement the target research idea. You must strictly follow these constraints:\n"
            "1. Consistency: Use tools to inspect the parent node's code and logs to ensure your implementation aligns with the baseline.\n"
            "2. Evaluation: Your `main.py` MUST print the final quantitative result on a single line exactly as: 'Metric: <value>'.\n"
            "3. Final Output: Once tool calls are complete, output ONLY the full, runnable Python code. Strictly NO markdown formatting (e.g., ```) and NO conversational text."
        )
        
        user_msg = (
            f"Snapshot:\n{snapshot}\n\n"
            f"Implementing Node ID: {node_id}\n"
            f"TLDR: {tldr}\n"
            f"Illustration: {illustration}\n"
        )
        
        history = [{"role": "system", "content": system_prompt}, 
                   {"role": "user", "content": user_msg}]
        
        code = self.chat_with_history(history)
        dest_dir = Path("namespace") / node_id
        (dest_dir / "main.py").write_text(code)
