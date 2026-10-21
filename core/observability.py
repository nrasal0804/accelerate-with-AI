"""Agent observability logger.

Captures plan, reasoning, tool calls, inputs, outputs, and timing for every
agent invocation. Writes a structured JSON trace to data/traces/ alongside
existing audit logs. Zero changes to any agent's input/output contract.

Usage in any agent::

    trace = AgentTrace("bronze_agent", run_id)
    trace.set_input(input_files=input_files, sttm_path=sttm_path)
    result = agent.invoke(...)
    trace.extract_from_messages(result.get("messages", []))
    trace.set_output(output_paths=output_paths).complete()
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    """Collects full observability data for a single agent invocation.

    Captures:
    - agent name, run_id, start timestamp
    - inputs passed to the agent
    - plan extracted from the agent's first substantive reasoning message
    - every tool call made (tool name + args)
    - step-by-step reasoning (task input → ai reasoning → tool results)
    - final outputs produced
    - wall-clock duration and terminal status (success / failed)
    """

    def __init__(self, agent_name: str, run_id: str):
        self.agent_name = agent_name
        self.run_id = run_id
        self.start_time = time.time()
        self.trace: dict[str, Any] = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": "",
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": 0.0,
            "status": "started",
        }

    def set_input(self, **kwargs) -> "AgentTrace":
        """Record the inputs this agent received from its caller."""
        self.trace["input"] = {k: v for k, v in kwargs.items()}
        return self

    def set_plan(self, plan: str) -> "AgentTrace":
        """Manually set the agent's stated plan (overrides auto-extraction)."""
        self.trace["plan"] = plan
        return self

    def set_output(self, **kwargs) -> "AgentTrace":
        """Record the final outputs this agent produced."""
        self.trace["output"] = {k: v for k, v in kwargs.items()}
        return self

    def extract_from_messages(self, messages: list) -> "AgentTrace":
        """Parse LangGraph message history to extract plan, tool calls, and reasoning.

        Handles HumanMessage, AIMessage (with optional tool_calls list), and
        ToolMessage. Unknown message types are silently skipped — safe to call
        with any message list including empty ones.
        """
        tool_calls: list[dict] = []
        reasoning_steps: list[dict] = []
        plan_text: str = ""

        for msg in messages:
            msg_type = type(msg).__name__
            content = getattr(msg, "content", "")

            if msg_type == "HumanMessage":
                if isinstance(content, str) and content.strip():
                    reasoning_steps.append({
                        "role": "task_input",
                        "content": content.strip()[:400],
                    })

            elif msg_type == "AIMessage":
                # Capture every tool call the agent decided to make
                raw_tool_calls = getattr(msg, "tool_calls", []) or []
                for tc in raw_tool_calls:
                    tool_calls.append({
                        "tool": (
                            tc.get("name", "unknown")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown")
                        ),
                        "args": (
                            tc.get("args", {})
                            if isinstance(tc, dict)
                            else getattr(tc, "args", {})
                        ),
                    })
                if isinstance(content, str) and content.strip():
                    reasoning_steps.append({
                        "role": "ai_reasoning",
                        "content": content.strip()[:600],
                    })
                    # First substantive AI message is treated as the plan
                    if not plan_text and len(content.strip()) > 20:
                        plan_text = content.strip()[:600]

            elif msg_type == "ToolMessage":
                tool_name = getattr(msg, "name", "unknown_tool")
                raw = content if isinstance(content, str) else str(content)
                preview = raw[:400] + "..." if len(raw) > 400 else raw
                reasoning_steps.append({
                    "role": "tool_result",
                    "tool": tool_name,
                    "content": preview,
                })

        self.trace["tool_calls"] = tool_calls
        self.trace["reasoning_steps"] = reasoning_steps
        if plan_text and not self.trace["plan"]:
            self.trace["plan"] = plan_text

        return self

    def complete(self, status: str = "success") -> dict:
        """Finalise the trace, append to disk, print a one-line summary."""
        self.trace["duration_seconds"] = round(time.time() - self.start_time, 3)
        self.trace["status"] = status
        self._save()
        self._print_summary()
        return self.trace

    def fail(self, error: str) -> dict:
        """Record failure reason, finalise, and save."""
        self.trace["error"] = error
        return self.complete(status="failed")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self):
        """Append this trace as a JSON entry to per-agent trace file for this run."""
        path = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        existing: list = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                existing = data if isinstance(data, list) else [data]
            except Exception:
                existing = []
        existing.append(self.trace)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

    def _print_summary(self):
        d = self.trace["duration_seconds"]
        tools_called = [t["tool"] for t in self.trace["tool_calls"]]
        steps = len(self.trace["reasoning_steps"])
        print(
            f"[OBSERVE][{self.agent_name}] status={self.trace['status']} "
            f"duration={d}s tools_called={tools_called} reasoning_steps={steps}"
        )
        if self.trace.get("plan"):
            print(f"[OBSERVE][{self.agent_name}] plan_preview={self.trace['plan'][:150]}")
