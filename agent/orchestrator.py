"""Workflow orchestrator for the Intent-Driven Medallion pipeline.

The Supervisor is itself a fully autonomous ReAct agent. For each pipeline phase
it receives the current pipeline state, THINKS about what needs to be done, PLANS
which specialist agents to call and in what order, ACTS by dispatching them as tools
with rich goal descriptions, VERIFIES each output before proceeding, and updates
the pipeline state.

Full observability is captured for the Supervisor itself (via AgentTrace) in addition
to the per-agent traces written by each specialist agent.

## Architecture

Four HITL-gated phases. Each phase runs a fresh Supervisor agent with the tools
relevant to that phase. The Supervisor is NOT given a rigid script — it reasons
about the pipeline state and decides how to proceed.

Phase 1 — Profile & Bronze STTM 
    Tools available: profiler_agent_tool, sttm_agent_tool
    Goal: understand raw data structure and produce Bronze ingestion rules for review.

Phase 2 — Bronze Execution & Silver STTM 
    Tools available: bronze_agent_tool, sttm_agent_tool
    Goal: ingest approved Bronze rules, then produce Silver cleansing rules for review.

Phase 3 — Silver Execution & Gold STTM (intent-driven Gold STTM)
    Tools available: silver_agent_tool, sttm_agent_tool
    Goal: cleanse Bronze outputs, then produce Gold materialisation rules for review.

Phase 4 — Gold Execution & Report (intent-driven Report)
    Tools available: gold_agent_tool, reporter_agent_tool
    Goal: materialise Gold tables, then produce the executive report.

UI contract (UNCHANGED — streamlit_app.py reads these):
    run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState
    run_bronze_to_silver_sttm(state) -> PipelineState
    run_silver_to_gold_sttm(state) -> PipelineState
    run_gold_and_report(state) -> PipelineState

PipelineState keys read by UI (UNCHANGED):
    run_id, status, error, sttm_bronze_path, sttm_silver_path, sttm_gold_path, report_path
"""

import json
import uuid
import traceback
from typing import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.audit import AuditLogger
from core.config import LLM_PROVIDER
from core.llm import create_chat_model
from core.memory import store_document
from core.observability import AgentTrace
from agents.profiler import profile_multiple_datasets
from agents.sttm_generator import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from agents.bronze_agent import execute_bronze
from agents.silver_agent import execute_silver
from agents.gold_agent import execute_gold
from agents.reporter import generate_report


# ---------------------------------------------------------------------------
# Pipeline state — keys UNCHANGED, UI reads them directly
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """State flowing through the pipeline. Keys read by Streamlit UI must not change."""
    run_id: str
    status: str
    uploaded_files: list[str]
    business_intent: str
    profile_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_sttm_approved: bool
    silver_sttm_approved: bool
    gold_sttm_approved: bool
    bronze_output_paths: list[str]
    silver_output_paths: list[str]
    gold_output_paths: list[str]
    report_path: str
    error: str


# ---------------------------------------------------------------------------
# Supervisor autonomous agent prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are the Pipeline Supervisor for an Intent-Driven Medallion data pipeline.
You are a fully autonomous ReAct agent. You do NOT follow a rigid script —
you think about the pipeline state, plan what needs to happen, dispatch specialist
agents as tools, verify their outputs, and decide when the phase is complete.

## What this pipeline does
Transforms raw retail CSV data through three quality layers:
  Bronze → raw ingestion with metadata
  Silver → cleansed, typed, deduplicated data
  Gold   → analytics-ready joined and aggregated tables (intent-driven)
Then produces a business-intent-driven executive report.

## Your operating mode — follow this sequence for every phase

1. THINK: Read the phase goal carefully. What is the current state of the pipeline?
   What data is available? What needs to be produced by the end of this phase?

2. PLAN: Write your explicit plan before calling any tool:
   - Which agents will you call, and in what order?
   - What goal will you give each agent?
   - What output do you expect from each agent, and how will you verify it?
   - What could go wrong, and how will you handle it?

3. ACT: Dispatch each specialist agent tool in your planned order.
   Give each agent a rich, specific goal description — not just "execute".
   Each tool you call launches a fully autonomous agent that will:
     * Inspect its own inputs
     * Form its own execution plan
     * Execute and verify its output
   You do not need to tell the agent HOW to do its job — just WHAT you need.

4. VERIFY: After each tool returns, check its output:
   - Did it return the expected keys (profile_path, sttm_path, output_paths, etc.)?
   - Are the paths non-empty?
   - If a tool returns an error, report it clearly and stop — do not call the next tool.

5. CONFIRM: Once all tools in the phase have completed successfully, summarise
   what was accomplished and confirm phase completion.

## Tool contract
- Every tool captures its own file paths, run IDs, and context via closure.
- You pass a `goal` parameter to each tool describing what you need.
- Tool outputs are JSON objects — read them to verify completion.
- Do NOT attempt to pass file paths from one tool to another — each tool
  resolves its own inputs from the pipeline context automatically.

## Error handling
- If any tool raises an error or returns an error key, stop immediately.
- Report the error clearly including which tool failed and what it returned.
- Do not attempt the next tool after a failure.

## Important
You are coordinating autonomous specialist agents — trust them to handle their
own execution details. Your value is in planning, sequencing, verification,
and understanding the pipeline state."""


# ---------------------------------------------------------------------------
# LLM factory — single point for provider selection
# ---------------------------------------------------------------------------

def _make_llm():
    return create_chat_model()


# ---------------------------------------------------------------------------
# Phase 1 tool factory: profiler_agent_tool + sttm_agent_tool (Bronze)
# ---------------------------------------------------------------------------

def _make_phase1_tools(uploaded_files: list[str], run_id: str):
    """Build Phase 1 tools: profiler and Bronze STTM generator.

    Both tools are intent-agnostic. Scratchpad allows sttm_agent_tool to
    automatically consume the profile_path produced by profiler_agent_tool
    without the Supervisor reproducing file paths.
    """
    scratchpad: dict = {}

    @tool
    def profiler_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Data Profiler agent.

        The profiler agent will inspect the uploaded CSV files, compute column-level
        statistics, identify semantic meanings, discover potential join keys, and note
        data quality observations. It produces a combined profile JSON used by the
        STTM agent to generate transformation rules.

        Pass a goal describing what profiling is needed and why.
        Returns JSON: {"profile_path": "path/to/profile.json"}.
        Must be called before sttm_agent_tool in Phase 1.
        """
        print(f"[ORCHESTRATOR] Dispatching profiler agent | goal: {goal[:120]}")
        profile_path = profile_multiple_datasets(
            file_paths=uploaded_files,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Files to profile: {uploaded_files}\n"
                "Inspect the files first, then compute full statistics, then return "
                "semantic analysis covering all columns, join keys, and quality notes."
            ),
        )
        scratchpad["profile_path"] = profile_path
        return json.dumps({"profile_path": profile_path})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 1: generates Bronze ingestion rules (column renames, type casts,
        metadata rows) from the data profile. Bronze is intent-agnostic — every
        source column is mapped mechanically. Requires profiler_agent_tool to have run.

        Pass a goal that clearly states: which layer's STTM to generate (Bronze)
        and what the STTM will be used for.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        if "profile_path" not in scratchpad:
            return json.dumps({"error": "profiler_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Bronze) | goal: {goal[:120]}")
        sttm_path = generate_bronze_sttm(
            profile_path=scratchpad["profile_path"],
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Bronze\n"
                f"Profile path: {scratchpad['profile_path']}\n"
                "Bronze is intent-agnostic. Inspect the profile context first, then "
                "generate a complete Bronze STTM covering every column. "
                "Add _load_timestamp and _source_file metadata rows. "
                "Do NOT add a surrogate key — that belongs in Silver."
            ),
        )
        scratchpad["sttm_bronze_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return profiler_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 2 tool factory: bronze_agent_tool + sttm_agent_tool (Silver)
# ---------------------------------------------------------------------------

def _make_phase2_tools(
    uploaded_files: list[str],
    sttm_bronze_path: str,
    run_id: str,
):
    """Build Phase 2 tools: Bronze execution and Silver STTM generator (intent-agnostic)."""
    scratchpad: dict = {}

    @tool
    def bronze_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Bronze layer ingestion agent.

        The Bronze agent will inspect the raw CSV input files and the approved STTM
        rules, form an explicit ingestion plan, apply column renaming, type casting,
        and metadata injection (_load_timestamp, _source_file), and write Bronze
        Parquet artifacts. It operates on the approved Bronze STTM rules exactly.

        Pass a goal describing what ingestion is needed — the agent handles the
        execution details autonomously.
        Returns JSON: {"bronze_output_paths": ["path1.parquet", ...]}.
        Must be called before sttm_agent_tool in Phase 2.
        """
        print(f"[ORCHESTRATOR] Dispatching Bronze agent | goal: {goal[:120]}")
        output_paths = execute_bronze(
            input_files=uploaded_files,
            sttm_path=sttm_bronze_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input CSV files: {uploaded_files}\n"
                f"Approved Bronze STTM: {sttm_bronze_path}\n"
                "Inspect the files and STTM rules first. Plan which transformations "
                "apply to each file. Then execute ingestion across all input files."
            ),
        )
        scratchpad["bronze_output_paths"] = output_paths
        return json.dumps({"bronze_output_paths": output_paths})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 2: generates Silver cleansing rules (null handling, deduplication,
        type casting, date standardisation, surrogate key injection) from the Bronze
        Parquet outputs. Silver is intent-agnostic — standard cleansing is applied
        to every Bronze column. Requires bronze_agent_tool to have run first.

        Pass a goal that clearly states: which layer's STTM to generate (Silver)
        and what cleansing is expected.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        if "bronze_output_paths" not in scratchpad:
            return json.dumps({"error": "bronze_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Silver) | goal: {goal[:120]}")
        sttm_path = generate_silver_sttm(
            bronze_output_paths=scratchpad["bronze_output_paths"],
            bronze_sttm_path=sttm_bronze_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Silver\n"
                f"Bronze output files: {scratchpad['bronze_output_paths']}\n"
                f"Approved Bronze STTM: {sttm_bronze_path}\n"
                "Silver is intent-agnostic. Inspect the Bronze Parquet metadata first. "
                "Plan null handling, type casting, deduplication, and date standardisation "
                "for every column. Add surrogate key as the first row. Then generate the "
                "complete Silver STTM."
            ),
        )
        scratchpad["sttm_silver_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return bronze_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 3 tool factory: silver_agent_tool + sttm_agent_tool (Gold)
# ---------------------------------------------------------------------------

def _make_phase3_tools(
    bronze_output_paths: list[str],
    sttm_silver_path: str,
    business_intent: str,
    run_id: str,
):
    """Build Phase 3 tools: Silver execution and Gold STTM generator."""
    scratchpad: dict = {}

    @tool
    def silver_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Silver layer cleansing agent.

        The Silver agent will inspect the Bronze Parquet inputs and approved STTM
        cleansing rules, form an explicit cleansing plan covering null handling,
        deduplication, type casting, date standardisation, and surrogate key injection,
        then execute cleansing across all Bronze inputs.

        Pass a goal describing what cleansing quality is expected — the agent handles
        execution details autonomously.
        Returns JSON: {"silver_output_paths": ["path1.parquet", ...]}.
        Must be called before sttm_agent_tool in Phase 3.
        """
        print(f"[ORCHESTRATOR] Dispatching Silver agent | goal: {goal[:120]}")
        output_paths = execute_silver(
            input_files=bronze_output_paths,
            sttm_path=sttm_silver_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input Bronze files: {bronze_output_paths}\n"
                f"Approved Silver STTM: {sttm_silver_path}\n"
                "Inspect the Bronze Parquet schemas and STTM rules first. Plan the "
                "cleansing approach for each column and file. Then execute cleansing "
                "across all Bronze inputs, producing Silver Parquet outputs."
            ),
        )
        scratchpad["silver_output_paths"] = output_paths
        return json.dumps({"silver_output_paths": output_paths})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 3: generates Gold materialisation rules (joins across Silver tables,
        renames, aggregations, surrogate key) from the Silver Parquet outputs.
        Requires silver_agent_tool to have run first.

        Pass a goal that clearly states: which layer's STTM to generate (Gold),
        what analytics-ready tables are needed, and what the business intent is.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        if "silver_output_paths" not in scratchpad:
            return json.dumps({"error": "silver_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Gold) | goal: {goal[:120]}")
        sttm_path = generate_gold_sttm(
            silver_output_paths=scratchpad["silver_output_paths"],
            silver_sttm_path=sttm_silver_path,
            business_intent=business_intent,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Gold\n"
                f"Business intent: {business_intent}\n"
                f"Silver output files: {scratchpad['silver_output_paths']}\n"
                f"Approved Silver STTM: {sttm_silver_path}\n"
                "Inspect the Silver Parquet metadata first. Plan join keys, column "
                "renames, and aggregation rules. Build queryable analytics-ready tables "
                "— do NOT pre-aggregate for the business question. Add surrogate key "
                "as the first row. Then generate the complete Gold STTM."
            ),
        )
        scratchpad["sttm_gold_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return silver_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 4 tool factory: gold_agent_tool + reporter_agent_tool
# ---------------------------------------------------------------------------

def _make_phase4_tools(
    silver_output_paths: list[str],
    sttm_gold_path: str,
    business_intent: str,
    run_id: str,
):
    """Build Phase 4 tools: Gold execution and report generation."""
    scratchpad: dict = {}

    @tool
    def gold_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Gold layer materialisation agent.

        The Gold agent will inspect the Silver Parquet inputs and approved STTM
        materialisation rules, form an explicit plan covering joins across source
        tables, column renames, aggregations, and surrogate key injection, then
        materialise all Gold target tables.

        Pass a goal describing what analytics-ready tables are needed — the agent
        handles execution autonomously. Business intent is already baked into the
        approved Gold STTM, so this dispatch is intent-agnostic.
        Returns JSON: {"gold_output_paths": ["path1.parquet", ...]}.
        Must be called before reporter_agent_tool in Phase 4.
        """
        print(f"[ORCHESTRATOR] Dispatching Gold agent | goal: {goal[:120]}")
        output_paths = execute_gold(
            input_files=silver_output_paths,
            sttm_path=sttm_gold_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input Silver files: {silver_output_paths}\n"
                f"Approved Gold STTM: {sttm_gold_path}\n"
                "Inspect the Silver Parquet schemas and Gold STTM rules first, grouped "
                "by target table. Plan joins, renames, and aggregations per Gold table. "
                "Then materialise all Gold target tables from the Silver inputs."
            ),
        )
        scratchpad["gold_output_paths"] = output_paths
        return json.dumps({"gold_output_paths": output_paths})

    @tool
    def reporter_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Reporter agent.

        The Reporter agent will inspect the available Gold tables, form an analytical
        plan to answer the business question, load the tables into DuckDB, write and
        execute SQL, then render a self-contained HTML executive report with charts.

        Pass a goal that clearly states the business question and what kind of analysis
        and visualisation is expected — the agent handles execution autonomously.
        Requires gold_agent_tool to have run first.
        Returns JSON: {"report_path": "path/to/report.html"}.
        """
        if "gold_output_paths" not in scratchpad:
            return json.dumps({"error": "gold_agent_tool must be called before reporter_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching Reporter agent | goal: {goal[:120]}")
        report_path = generate_report(
            gold_files=scratchpad["gold_output_paths"],
            business_intent=business_intent,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Business question: {business_intent}\n"
                f"Gold files: {scratchpad['gold_output_paths']}\n"
                "Inspect the Gold tables first to understand their structure. Plan your "
                "SQL approach to directly answer the business question. Load the tables, "
                "execute your query, analyse results, and produce a structured HTML report "
                "with charts that provide visual evidence for your answer."
            ),
        )
        scratchpad["report_path"] = report_path
        return json.dumps({"report_path": report_path})

    return gold_agent_tool, reporter_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Autonomous Supervisor runner — the orchestrator's own ReAct loop
# ---------------------------------------------------------------------------

def _run_supervisor(
    tools: list,
    phase_goal: str,
    phase_name: str,
    run_id: str,
) -> dict:
    """Instantiate the autonomous Supervisor agent and run it for one phase.

    The Supervisor thinks about the phase goal, plans which tools to call and
    in what order, dispatches them with rich goal descriptions, and verifies
    outputs. Full observability is captured via AgentTrace.

    Args:
        tools: The specialist agent tools available to the Supervisor this phase.
        phase_goal: High-level goal describing what this phase must accomplish.
        phase_name: Short name for logging (e.g. "phase1").
        run_id: Pipeline run identifier.

    Returns:
        dict: The full agent result including message history.
    """
    trace = AgentTrace(f"supervisor_{phase_name}", run_id)
    trace.set_input(
        phase=phase_name,
        goal=phase_goal,
        tools_available=[t.name for t in tools],
    )

    llm = _make_llm()
    agent = create_agent(llm, tools, system_prompt=SUPERVISOR_PROMPT)

    print(f"[ORCHESTRATOR] Supervisor starting {phase_name} autonomously")
    print(f"[ORCHESTRATOR] Goal: {phase_goal[:200]}")

    try:
        result = agent.invoke({"messages": [HumanMessage(content=phase_goal)]})
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract final supervisor summary from last AI message
    final_summary = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if type(msg).__name__ == "AIMessage" and isinstance(content, str) and content.strip():
            final_summary = content.strip()[:400]
            break

    trace.set_output(
        phase=phase_name,
        tools_called=[t["tool"] for t in trace.trace["tool_calls"]],
        summary=final_summary,
    ).complete()

    print(f"[ORCHESTRATOR] Supervisor completed {phase_name}")
    return result


# ---------------------------------------------------------------------------
# Pipeline entry points — signatures UNCHANGED, UI calls these directly
# ---------------------------------------------------------------------------

def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Phase 1: Supervisor profiles data and generates Bronze STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py with (saved_paths, business_intent).
    Returns PipelineState with sttm_bronze_path populated.
    """
    run_id = str(uuid.uuid4())
    audit = AuditLogger(run_id)
    audit.log(
        "orchestrator", "pipeline_started",
        intent=business_intent, status="started", phase="upload",
        rationale="User submitted files and intent; Supervisor will profile data then generate Bronze STTM.",
    )
    store_document(
        doc_id=f"intent_{run_id}",
        text=business_intent,
        metadata={"type": "business_intent", "run_id": run_id},
    )

    state: PipelineState = {
        "run_id": run_id,
        "status": "profiling",
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
        "profile_path": "",
        "sttm_bronze_path": "",
        "sttm_silver_path": "",
        "sttm_gold_path": "",
        "bronze_sttm_approved": False,
        "silver_sttm_approved": False,
        "gold_sttm_approved": False,
        "bronze_output_paths": [],
        "silver_output_paths": [],
        "gold_output_paths": [],
        "report_path": "",
        "error": "",
    }

    profiler_t, sttm_t, scratchpad = _make_phase1_tools(uploaded_files, run_id)

    try:
        audit.log(
            "orchestrator", "phase1_supervisor_started",
            status="in_progress", phase="phase1",
            rationale=(
                "Supervisor agent will autonomously decide how to profile the raw data "
                "and generate Bronze STTM ingestion rules. Bronze is intent-agnostic."
            ),
        )
        _run_supervisor(
            tools=[profiler_t, sttm_t],
            phase_goal=(
                f"Phase 1 goal for run_id='{run_id}'.\n\n"
                f"Uploaded files: {uploaded_files}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Profile the uploaded raw CSV files to understand their structure, "
                "column semantics, data quality, and potential join keys across datasets.\n"
                "2. Use that profile to generate a complete Bronze STTM CSV that covers "
                "every column with ingestion rules (renaming, type casting, metadata injection).\n\n"
                "Bronze is intent-agnostic — map every column mechanically. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase1",
            run_id=run_id,
        )
        state.update({
            "profile_path": scratchpad.get("profile_path", ""),
            "sttm_bronze_path": scratchpad.get("sttm_bronze_path", ""),
            "status": "awaiting_bronze_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase1_supervisor_completed",
            status="success", phase="phase1",
            profile_path=scratchpad.get("profile_path"),
            sttm_bronze_path=scratchpad.get("sttm_bronze_path"),
        )
    except Exception as e:
        state.update({
            "error": f"Phase 1 supervisor failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "phase1_supervisor_failed",
            status="failed", phase="phase1", detail=str(e),
        )

    return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    """Phase 2: Supervisor executes Bronze layer and generates Silver STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py after Bronze STTM approval.
    Returns PipelineState with bronze_output_paths and sttm_silver_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["bronze_sttm_approved"] = True
    state["error"] = ""

    bronze_t, sttm_t, scratchpad = _make_phase2_tools(
        uploaded_files=state["uploaded_files"],
        sttm_bronze_path=state["sttm_bronze_path"],
        run_id=state["run_id"],
    )

    try:
        audit.log(
            "orchestrator", "phase2_supervisor_started",
            status="in_progress", phase="phase2",
            rationale=(
                "User approved Bronze STTM. Supervisor will autonomously execute Bronze "
                "ingestion and generate Silver cleansing rules."
            ),
        )
        _run_supervisor(
            tools=[bronze_t, sttm_t],
            phase_goal=(
                f"Phase 2 goal for run_id='{state['run_id']}'.\n\n"
                f"Uploaded raw files: {state['uploaded_files']}\n"
                f"Approved Bronze STTM: {state['sttm_bronze_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Bronze ingestion rules to transform raw CSV files "
                "into Bronze Parquet artifacts with lineage metadata.\n"
                "2. Inspect the Bronze outputs and generate a Silver STTM that cleanses every "
                "column — handle nulls, deduplicate, cast types, standardise dates, and inject "
                "a surrogate key as the first row.\n\n"
                "Silver is intent-agnostic — standard cleansing applies to every column. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase2",
            run_id=state["run_id"],
        )
        state.update({
            "bronze_output_paths": scratchpad.get("bronze_output_paths", []),
            "sttm_silver_path": scratchpad.get("sttm_silver_path", ""),
            "status": "awaiting_silver_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase2_supervisor_completed",
            status="success", phase="phase2",
            bronze_output_paths=scratchpad.get("bronze_output_paths"),
            sttm_silver_path=scratchpad.get("sttm_silver_path"),
        )
    except Exception as e:
        state.update({
            "error": f"Phase 2 supervisor failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "phase2_supervisor_failed",
            status="failed", phase="phase2", detail=str(e),
        )

    return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
    """Phase 3: Supervisor executes Silver layer and generates Gold STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py after Silver STTM approval.
    Returns PipelineState with silver_output_paths and sttm_gold_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["silver_sttm_approved"] = True
    state["error"] = ""

    silver_t, sttm_t, scratchpad = _make_phase3_tools(
        bronze_output_paths=state["bronze_output_paths"],
        sttm_silver_path=state["sttm_silver_path"],
        business_intent=state["business_intent"],
        run_id=state["run_id"],
    )

    try:
        audit.log(
            "orchestrator", "phase3_supervisor_started",
            status="in_progress", phase="phase3",
            rationale=(
                "User approved Silver STTM. Supervisor will autonomously execute Silver "
                "cleansing and generate Gold materialisation rules."
            ),
        )
        _run_supervisor(
            tools=[silver_t, sttm_t],
            phase_goal=(
                f"Phase 3 goal for run_id='{state['run_id']}'.\n\n"
                f"Business intent: {state['business_intent']}\n"
                f"Bronze Parquet files: {state['bronze_output_paths']}\n"
                f"Approved Silver STTM: {state['sttm_silver_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Silver cleansing rules to transform Bronze Parquet "
                "files into cleansed Silver Parquet artifacts with surrogate keys.\n"
                "2. Inspect the Silver outputs and generate a Gold STTM that defines how "
                "Silver tables should be joined, renamed, aggregated, and shaped into "
                "analytics-ready Gold target tables aligned to the business intent.\n\n"
                "Think about which Silver tables need to be joined to answer the business "
                "question, and what Gold table structure would best serve the Reporter agent. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase3",
            run_id=state["run_id"],
        )
        state.update({
            "silver_output_paths": scratchpad.get("silver_output_paths", []),
            "sttm_gold_path": scratchpad.get("sttm_gold_path", ""),
            "status": "awaiting_gold_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase3_supervisor_completed",
            status="success", phase="phase3",
            silver_output_paths=scratchpad.get("silver_output_paths"),
            sttm_gold_path=scratchpad.get("sttm_gold_path"),
        )
    except Exception as e:
        state.update({
            "error": f"Phase 3 supervisor failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "phase3_supervisor_failed",
            status="failed", phase="phase3", detail=str(e),
        )

    return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
    """Phase 4: Supervisor executes Gold layer and generates the executive report.

    UI contract: called by streamlit_app.py after Gold STTM approval.
    Returns PipelineState with gold_output_paths and report_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["gold_sttm_approved"] = True
    state["error"] = ""

    gold_t, reporter_t, scratchpad = _make_phase4_tools(
        silver_output_paths=state["silver_output_paths"],
        sttm_gold_path=state["sttm_gold_path"],
        business_intent=state["business_intent"],
        run_id=state["run_id"],
    )

    try:
        audit.log(
            "orchestrator", "phase4_supervisor_started",
            status="in_progress", phase="phase4",
            rationale=(
                "User approved Gold STTM. Supervisor will autonomously execute Gold "
                "materialisation and generate the executive report."
            ),
        )
        _run_supervisor(
            tools=[gold_t, reporter_t],
            phase_goal=(
                f"Phase 4 goal for run_id='{state['run_id']}'.\n\n"
                f"Business intent: {state['business_intent']}\n"
                f"Silver Parquet files: {state['silver_output_paths']}\n"
                f"Approved Gold STTM: {state['sttm_gold_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Gold materialisation rules to produce analytics-ready "
                "Gold Parquet tables from the Silver inputs, applying all approved joins, "
                "renames, and aggregations.\n"
                "2. Dispatch the Reporter agent to inspect the Gold tables, write SQL to "
                "directly answer the business question, and produce a self-contained HTML "
                "executive report with visual evidence (charts).\n\n"
                "Think about what the business question needs and whether the Gold tables "
                "are structured to answer it. Verify the Gold tables are populated before "
                "dispatching the Reporter. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase4",
            run_id=state["run_id"],
        )
        state.update({
            "gold_output_paths": scratchpad.get("gold_output_paths", []),
            "report_path": scratchpad.get("report_path", ""),
            "status": "completed",
        })
        audit.log(
            "orchestrator", "phase4_supervisor_completed",
            status="success", phase="phase4",
            gold_output_paths=scratchpad.get("gold_output_paths"),
            report_path=scratchpad.get("report_path"),
        )
    except Exception as e:
        state.update({
            "error": f"Phase 4 supervisor failed: {e}\n{traceback.format_exc()}",
            "status": "failed",
        })
        audit.log(
            "orchestrator", "phase4_supervisor_failed",
            status="failed", phase="phase4", detail=str(e),
        )

    return state
