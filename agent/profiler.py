"""Data profiling AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, uses inspect_files_tool to
preview structure first, forms an explicit plan, then calls profiler_tool for
full statistics, and returns enriched semantic analysis.

I/O contract (UNCHANGED — UI and orchestrator safe):
    profile_dataset(file_path, run_id, task_description) -> str
    profile_multiple_datasets(file_paths, run_id, task_description) -> str
"""

import json
import os
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import PROFILES_DIR, LLM_PROVIDER
from core.audit import AuditLogger
from core.llm import create_chat_model
from core.observability import AgentTrace


PROFILER_AGENT_PROMPT = """You are an autonomous Data Analyst specialising in data profiling and schema
analysis for Medallion data pipelines. You operate independently: you receive a goal
from the orchestrator, inspect available data, reason about the approach, execute
profiling, and return verified semantic analysis.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the datasets, the business intent, and what quality
   observations will matter downstream for STTM generation.

2. INSPECT: Call inspect_files_tool FIRST. This gives you a lightweight preview of
   file shapes, column names, dtypes, and sample values without running full stats.
   After seeing the results, state your observations.

3. PLAN: Based on the inspection output, write your explicit plan:
   - Which columns likely carry semantic meaning (IDs, dates, amounts, categories)?
   - Which column pairs across datasets are candidate join keys?
   - What quality issues do you anticipate (high nulls, low cardinality, mixed types)?

4. ACT: Call profiler_tool to compute full column-level statistics for all datasets.

5. VERIFY: Using the statistics returned, produce the final semantic analysis JSON
   and confirm output was saved.

## Available tools

- **inspect_files_tool**: Quickly previews each uploaded CSV — shape, column names,
  dtypes, and 3 sample values per column. Call this FIRST to inform your plan.
  Returns a JSON summary keyed by dataset name.

- **profiler_tool**: Computes full column-level statistics for all datasets (dtype,
  null count, null %, unique count, sample values, min/max/mean for numerics).
  Call this AFTER forming your plan. Returns a JSON statistics object.

## Final output format
Return ONLY a valid JSON object — no markdown fences, no prose:
{
  "semantic_meanings": {
    "dataset_name": { "column_name": "short semantic description" }
  },
  "join_keys": [
    {
      "left_dataset": "name", "left_column": "col",
      "right_dataset": "name", "right_column": "col",
      "confidence": "high|medium|low"
    }
  ],
  "quality_notes": ["observation 1", "observation 2"]
}"""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM, called via tool closures
# ---------------------------------------------------------------------------

def _inspect_files(file_paths: list[str]) -> dict:
    """Lightweight CSV preview: shape, columns, dtypes, 3 sample values. No LLM."""
    summary = {}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            summary[os.path.basename(fp)] = {"error": str(e)}
            continue
        name = os.path.basename(fp).replace(".csv", "")
        col_previews = {}
        for col in df.columns:
            col_previews[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist(),
                "null_count": int(df[col].isnull().sum()),
            }
        summary[name] = {
            "file": fp,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_preview": col_previews,
        }
    return summary


def _compute_stats(file_paths: list[str]) -> dict:
    """Full column-level statistics across all CSV files. No LLM."""
    combined_profile: dict = {"files": [], "datasets": {}}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[PROFILER] Could not read {fp}: {e}")
            continue
        dataset_name = os.path.basename(fp).replace(".csv", "")
        combined_profile["files"].append(fp)
        ds_profile: dict = {
            "file": fp,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": {},
        }
        for col in df.columns:
            col_info: dict = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            if df[col].dtype in ["int64", "float64"]:
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            else:
                col_info["sample_values"] = df[col].dropna().head(5).tolist()
            ds_profile["columns"][col] = col_info
        combined_profile["datasets"][dataset_name] = ds_profile
    return combined_profile


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_profiler_tools(file_paths: list[str], run_id: str):
    """Returns inspect + profiler tools bound to this run's file_paths via closure."""

    @tool
    def inspect_files_tool(confirmation: str = "execute") -> str:
        """Preview uploaded CSV files to understand structure before full profiling.

        Returns a JSON summary of each file's shape, column names, dtypes, and 3 sample
        values per column. Call this FIRST to form your profiling and analysis plan.
        """
        return json.dumps(_inspect_files(file_paths), default=str)

    @tool
    def profiler_tool(confirmation: str = "execute") -> str:
        """Compute full column-level statistics for all uploaded CSV datasets.

        Reads each CSV and computes dtype, null count, null %, unique count, sample
        values, and numeric distributions (min/max/mean). Returns a JSON statistics
        object covering all datasets. Call this AFTER inspect_files_tool.
        """
        return json.dumps(_compute_stats(file_paths), default=str)

    return inspect_files_tool, profiler_tool


# ---------------------------------------------------------------------------
# LLM factory — single point for provider selection
# ---------------------------------------------------------------------------

def _make_llm():
    return create_chat_model()


# ---------------------------------------------------------------------------
# Public entry points — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def profile_dataset(file_path: str, run_id: str, task_description: str) -> str:
    """Profile a single CSV file. Delegates to profile_multiple_datasets."""
    return profile_multiple_datasets([file_path], run_id, task_description)


def profile_multiple_datasets(file_paths: list[str], run_id: str, task_description: str) -> str:
    """Profiler AI agent entry point — autonomous ReAct version.

    Args:
        file_paths: CSV file paths to profile.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved combined profile JSON.
    """
    trace = AgentTrace("profiler", run_id)
    trace.set_input(file_paths=file_paths)

    audit = AuditLogger(run_id)
    print(f"[PROFILER] Started — files: {file_paths}")
    audit.log("profiler", "started_multi", input_files=file_paths)

    inspect_tool, stats_tool = _make_profiler_tools(file_paths, run_id)
    llm = _make_llm()

    print(f"[PROFILER] Running autonomous ReAct agent ({LLM_PROVIDER})")
    agent = create_agent(llm, [inspect_tool, stats_tool], system_prompt=PROFILER_AGENT_PROMPT)

    try:
        result = agent.invoke({"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract raw stats (from profiler_tool message) and semantic analysis (from final AI message)
    raw_stats: dict = {}
    analysis: dict = {}

    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif text.strip().startswith("```"):
            text = text.strip().lstrip("`").strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "datasets" in parsed and "files" in parsed:
                    raw_stats = parsed
                elif any(k in parsed for k in ("semantic_meanings", "join_keys", "quality_notes")):
                    analysis = parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Fallback: recompute stats locally if tool message was not parseable
    if not raw_stats:
        print("[PROFILER] Recomputing stats locally (tool message not parseable)")
        raw_stats = _compute_stats(file_paths)

    combined_profile = raw_stats
    combined_profile["analysis"] = analysis if analysis else {}

    profile_filename = f"profile_combined_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    profile_path = str(PROFILES_DIR / profile_filename)
    print(f"[PROFILER] Saving profile → {profile_path}")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(combined_profile, f, indent=2)

    audit.log("profiler", "completed_multi", output_file=profile_path)
    trace.set_output(profile_path=profile_path).complete()
    print(f"[PROFILER] Done — {profile_path}")
    return profile_path
