"""IDAMP Streamlit control surface.

This module is the human-facing entry point for the workflow application. It guides users
through the full Intent-Driven Agentic Medallion pipeline in five phases:
1) upload data and business intent,
2) review/approve Bronze STTM,
3) review/approve Silver STTM,
4) review/approve Gold STTM,
5) view and download the generated executive report.

What this module does:
- Captures user intent and uploaded CSV files.
- Persists files to the landing zone for agents.
- Invokes orchestrator checkpoints between HITL approvals.
- Renders run progress and audit events for explainability.
- Displays final HTML report output.
"""

import sys
import json
import html
from textwrap import dedent
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from core.config import LANDING_DIR, STTM_DIR
from core.audit import AuditLogger
from agents.orchestrator import (
    run_until_bronze_sttm,
    run_bronze_to_silver_sttm,
    run_silver_to_gold_sttm,
    run_gold_and_report,
)

st.set_page_config(
    page_title="IDAMP - Intent-Driven Agentic Medallion Pipeline",
    page_icon="🏗️",
    layout="wide",
)


PROGRESS_STEPS = [
    ("upload", "Upload & Intent"),
    ("bronze_sttm", "Bronze STTM Review"),
    ("bronze_load", "Bronze Layer Load"),
    ("silver_sttm", "Silver STTM Review"),
    ("silver_load", "Silver Layer Load"),
    ("gold_sttm", "Gold STTM Review"),
    ("gold_load", "Gold Layer Load"),
    ("report", "Executive Report"),
]

SELECTION_COL = "_selected_for_approval"


def render_progress_banner(current_phase: str, state: dict | None = None, report_complete: bool = False) -> None:
    # Convert logical workflow phase into a visual stepper state for the application UI.
    phase_to_index = {phase: index for index, (phase, _) in enumerate(PROGRESS_STEPS)}
    current_index = phase_to_index.get(current_phase, 0)

    # Mark report as complete as soon as a report path is present (or an explicit completion flag is set).
    report_path = (state or {}).get("report_path", "")
    report_exists = bool(report_path) and Path(str(report_path)).exists()
    if current_phase == "report" and (report_complete or report_exists):
        current_index += 1

    banner_parts = [
        "<style>"
        ".idamp-progress-shell {"
        "position: sticky;"
        "top: 0;"
        "z-index: 1000;"
        "padding: 0.35rem 0 1rem 0;"
        "background: linear-gradient(180deg, rgba(14, 17, 23, 0.96) 0%, rgba(14, 17, 23, 0.88) 85%, rgba(14, 17, 23, 0) 100%);"
        "backdrop-filter: blur(10px);"
        "}"
        ".idamp-progress {"
        "display: flex;"
        "align-items: flex-start;"
        "justify-content: center;"
        "gap: 0;"
        "width: 100%;"
        "padding: 1rem 1.25rem 0.25rem 1.25rem;"
        "border: 1px solid rgba(120, 138, 160, 0.24);"
        "border-radius: 18px;"
        "background: linear-gradient(135deg, rgba(24, 30, 41, 0.96) 0%, rgba(18, 23, 33, 0.92) 100%);"
        "box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);"
        "}"
        ".idamp-progress-step {"
        "flex: 1 1 0;"
        "min-width: 0;"
        "display: flex;"
        "flex-direction: column;"
        "align-items: center;"
        "text-align: center;"
        "position: relative;"
        "}"
        ".idamp-progress-step:not(:last-child)::after {"
        "content: '';"
        "position: absolute;"
        "top: 1.1rem;"
        "left: calc(50% + 1.35rem);"
        "width: calc(100% - 2.7rem);"
        "height: 4px;"
        "border-radius: 999px;"
        "background: rgba(91, 103, 122, 0.45);"
        "}"
        ".idamp-progress-step.is-complete:not(:last-child)::after {"
        "background: linear-gradient(90deg, #18b26b 0%, #38c983 100%);"
        "}"
        ".idamp-progress-step.is-current:not(:last-child)::after {"
        "background: linear-gradient(90deg, #f0b84b 0%, rgba(91, 103, 122, 0.45) 100%);"
        "}"
        ".idamp-progress-node {"
        "width: 2.7rem;"
        "height: 2.7rem;"
        "border-radius: 999px;"
        "display: flex;"
        "align-items: center;"
        "justify-content: center;"
        "font-size: 1rem;"
        "font-weight: 700;"
        "border: 3px solid rgba(120, 138, 160, 0.45);"
        "background: #18202c;"
        "color: #d6deeb;"
        "position: relative;"
        "z-index: 1;"
        "box-sizing: border-box;"
        "}"
        ".idamp-progress-step.is-complete .idamp-progress-node {"
        "background: linear-gradient(135deg, #159a5d 0%, #1ec978 100%);"
        "border-color: rgba(76, 230, 146, 0.5);"
        "color: #ffffff;"
        "box-shadow: 0 0 0 8px rgba(30, 201, 120, 0.12);"
        "}"
        ".idamp-progress-step.is-current .idamp-progress-node {"
        "background: linear-gradient(135deg, #f3b63e 0%, #ffcf70 100%);"
        "border-color: rgba(255, 216, 140, 0.7);"
        "color: #1b1f27;"
        "box-shadow: 0 0 0 8px rgba(243, 182, 62, 0.16);"
        "}"
        ".idamp-progress-label {"
        "margin-top: 0.75rem;"
        "font-size: 0.92rem;"
        "line-height: 1.25;"
        "color: #c8d1df;"
        "font-weight: 600;"
        "max-width: 9rem;"
        "}"
        ".idamp-progress-step.is-current .idamp-progress-label {"
        "color: #fff2c6;"
        "}"
        ".idamp-progress-step.is-complete .idamp-progress-label {"
        "color: #dff9ea;"
        "}"
        "@media (max-width: 900px) {"
        ".idamp-progress {"
        "overflow-x: auto;"
        "justify-content: flex-start;"
        "}"
        ".idamp-progress-step {"
        "min-width: 140px;"
        "}"
        "}"
        "</style>"
        "<div class='idamp-progress-shell'><div class='idamp-progress'>"
    ]

    for index, (_, label) in enumerate(PROGRESS_STEPS):
        if index < current_index:
            status_class = "is-complete"
            marker = "✓"
        elif index == current_index:
            status_class = "is-current"
            marker = str(index + 1)
        else:
            status_class = "is-pending"
            marker = str(index + 1)

        banner_parts.append(
            f"<div class='idamp-progress-step {status_class}'><div class='idamp-progress-node'>{marker}</div><div class='idamp-progress-label'>{label}</div></div>"
        )

    banner_parts.append("</div></div>")

    st.markdown("".join(banner_parts), unsafe_allow_html=True)


def _reset_analysis_session() -> None:
    st.session_state.phase = "upload"
    st.session_state.pipeline_state = None
    st.session_state.current_run_id = ""


def _prepare_sttm_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    # Add default approvals so reviewers can uncheck only rules they want to reject.
    editor_df = df.copy()
    if SELECTION_COL not in editor_df.columns:
        editor_df.insert(0, SELECTION_COL, True)
    editor_df[SELECTION_COL] = editor_df[SELECTION_COL].fillna(True).astype(bool)
    return editor_df


def _extract_selected_rows(edited_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only approved rows; this becomes the STTM passed to the next execution phase.
    if SELECTION_COL not in edited_df.columns:
        return edited_df.copy()
    return edited_df[edited_df[SELECTION_COL]].drop(columns=[SELECTION_COL], errors="ignore")


def _current_audit_logs() -> list[dict]:
    run_id = st.session_state.get("current_run_id", "")
    if not run_id:
        return []
    return AuditLogger(run_id).get_logs()


def _format_audit_entry(entry: dict) -> str:
    timestamp = str(entry.get("timestamp", ""))
    time_text = timestamp[11:19] if len(timestamp) >= 19 else "--:--:--"
    agent = str(entry.get("agent", "unknown"))
    action = str(entry.get("action", ""))

    rationale = entry.get("rationale") or entry.get("decision_basis") or entry.get("detail") or "No rationale captured"
    phase = entry.get("phase", "")
    status = entry.get("status", "")

    return f"[{time_text}] {agent} | {action}\nReason: {rationale}\nPhase: {phase or '-'} | Status: {status or '-'}"


def _status_class(entry: dict) -> str:
    status = str(entry.get("status", "")).strip().lower()
    action = str(entry.get("action", "")).strip().lower()

    if status in {"failed", "error"} or "failed" in action or "error" in action:
        return "is-failed"
    if status in {"success", "completed"}:
        return "is-success"
    if status in {"in_progress", "started", "running"}:
        return "is-progress"
    return "is-neutral"


def _render_audit_card(entry: dict, latest: bool = False) -> str:
    timestamp = str(entry.get("timestamp", ""))
    time_text = timestamp[11:19] if len(timestamp) >= 19 else "--:--:--"

    agent = html.escape(str(entry.get("agent", "unknown")))
    action = html.escape(str(entry.get("action", "")))
    rationale = html.escape(str(entry.get("rationale") or entry.get("decision_basis") or entry.get("detail") or "No rationale captured"))
    phase = html.escape(str(entry.get("phase", "-") or "-"))
    status = html.escape(str(entry.get("status", "-") or "-"))

    card_class = _status_class(entry)
    latest_badge = "<span class='idamp-audit-latest-badge'>LATEST</span>" if latest else ""

    return (
        f"<div class='idamp-audit-card {card_class}'>"
        f"<div class='idamp-audit-card-top'><span class='idamp-audit-time'>{time_text}</span>{latest_badge}</div>"
        f"<div class='idamp-audit-head'>{agent} <span class='idamp-audit-sep'>|</span> {action}</div>"
        f"<div class='idamp-audit-reason'>{rationale}</div>"
        f"<div class='idamp-audit-meta'><span>Phase: {phase}</span><span class='idamp-audit-status'>{status}</span></div>"
        "</div>"
    )


def render_audit_trail_panel() -> None:
    # Show latest-first run telemetry so users can map each UI action to agent decisions.
    st.markdown("### Audit Trail")
    st.caption("Current analysis only")

    logs = _current_audit_logs()
    if not logs:
        st.info("No audit events yet. Start the pipeline to capture orchestration and agent actions.")
        return

    latest_event = logs[-1]
    previous_events = list(reversed(logs[:-1]))

    style_block = dedent("""
    <style>
    .idamp-audit-wrap {
        width: 100%;
    }
    .idamp-audit-card {
        border: 1px solid rgba(120, 138, 160, 0.26);
        border-left-width: 5px;
        border-radius: 12px;
        padding: 0.55rem 0.65rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, rgba(24, 30, 41, 0.94) 0%, rgba(16, 21, 31, 0.92) 100%);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.18);
    }
    .idamp-audit-card.is-success { border-left-color: #1ec978; }
    .idamp-audit-card.is-progress { border-left-color: #f3b63e; }
    .idamp-audit-card.is-failed { border-left-color: #f24d66; }
    .idamp-audit-card.is-neutral { border-left-color: #8ea0b8; }
    .idamp-audit-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.2rem;
        color: #9db0c7;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .idamp-audit-time {
        letter-spacing: 0.02em;
    }
    .idamp-audit-latest-badge {
        background: linear-gradient(135deg, #f3b63e 0%, #ffd27c 100%);
        color: #171b24;
        border-radius: 999px;
        padding: 0.1rem 0.4rem;
        font-size: 0.65rem;
        font-weight: 800;
    }
    .idamp-audit-head {
        color: #e8eef8;
        font-size: 0.84rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.22rem;
        word-break: break-word;
    }
    .idamp-audit-sep { color: #8ea0b8; }
    .idamp-audit-reason {
        color: #c8d4e5;
        font-size: 0.78rem;
        line-height: 1.2;
        margin-bottom: 0.26rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        word-break: break-word;
    }
    .idamp-audit-meta {
        display: flex;
        justify-content: space-between;
        gap: 0.4rem;
        color: #9fb0c6;
        font-size: 0.72rem;
    }
    .idamp-audit-status {
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-weight: 700;
    }
    .idamp-audit-history {
        max-height: 470px;
        overflow-y: auto;
        padding-right: 0.15rem;
        margin-top: 0.4rem;
    }
    .idamp-audit-history-empty {
        color: #9fb0c6;
        font-size: 0.78rem;
        padding: 0.35rem 0.1rem;
    }
    </style>
    """)

    previous_cards = "".join(_render_audit_card(entry) for entry in previous_events)
    if not previous_cards:
        previous_cards = "<div class='idamp-audit-history-empty'>No previous events for this run.</div>"

    panel_html = (
        style_block
        + "<div class='idamp-audit-wrap'>"
        + _render_audit_card(latest_event, latest=True)
        + f"<div class='idamp-audit-history'>{previous_cards}</div>"
        + "</div>"
    )

    st.markdown(panel_html, unsafe_allow_html=True)

    run_id = st.session_state.get("current_run_id", "current")
    st.download_button(
        "Download Audit Trail",
        data=json.dumps(logs, indent=2),
        file_name=f"audit_{run_id}.json",
        mime="application/json",
        use_container_width=True,
    )

st.title("Intent-Driven Agentic Medallion Workflow")
st.markdown("Upload CSV datasets and describe your analytical goal. The multi-agent system will process data through Bronze → Silver → Gold layers with HITL approval at each stage.")

# Initialize session state
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "phase" not in st.session_state:
    st.session_state.phase = "upload"
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = ""
if "report_complete" not in st.session_state:
    st.session_state.report_complete = False

if st.session_state.pipeline_state and st.session_state.pipeline_state.get("run_id"):
    st.session_state.current_run_id = st.session_state.pipeline_state["run_id"]

render_progress_banner(
    st.session_state.phase,
    st.session_state.pipeline_state,
    st.session_state.get("report_complete", False),
)

main_col, audit_col = st.columns([3.4, 1.4], gap="large")

with audit_col:
    render_audit_trail_panel()

with main_col:
    # ========== PHASE 1: UPLOAD & INTENT ==========
    if st.session_state.phase == "upload":
        st.header("📤 Phase 1: Upload Data & Define Intent")

        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        business_intent = st.text_area(
            "Business Intent / Question",
            placeholder="What is the inference you are trying to draw from the uploaded data?",
            height=100,
        )

        if st.button("🚀 Start Workflow", disabled=not (uploaded_files and business_intent)):
            # Persist uploads before orchestration so all agents use stable file paths.
            saved_paths = []
            for uf in uploaded_files:
                if uf is None:
                    st.error("One of the uploaded files is invalid.")
                    st.stop()
                save_path = str(LANDING_DIR / uf.name)
                try:
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())
                    saved_paths.append(save_path)
                except Exception as e:
                    st.error(f"Failed to save {uf.name}: {str(e)}")
                    st.stop()

            with st.spinner("🔍 Profiling data and generating Bronze STTM..."):
                try:
                    # Transition: capture intent and produce first draft STTM for Bronze review.
                    result = run_until_bronze_sttm(saved_paths, business_intent)
                    st.session_state.pipeline_state = result
                    st.session_state.current_run_id = result.get("run_id", "")
                    if result.get("error"):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.session_state.phase = "bronze_sttm"
                        st.rerun()
                except Exception as e:
                    st.error(f"Pipeline failed: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())


    # ========== PHASE 2: BRONZE STTM REVIEW ==========
    elif st.session_state.phase == "bronze_sttm":
        st.header("🥉 Phase 2: Review Bronze Layer STTM")
        state = st.session_state.pipeline_state

        st.info("**Bronze Layer**: Raw data ingestion with metadata columns. Review and approve transformations below.")

        sttm_path = state.get("sttm_bronze_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor_df(df)

            st.write(f"**Total Transformations:** {len(df)}")

            # Editable STTM table
            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("", default=True),
                    "transformation_logic": st.column_config.TextColumn("Transformation Logic", width="large"),
                },
                key="bronze_sttm_editor",
                height=500
            )

            selected_df = _extract_selected_rows(edited_df)

            if selected_df.empty:
                st.warning("Select at least one row to continue.")

            if st.button("✅ Approve & Continue", type="primary", use_container_width=True, disabled=selected_df.empty):
                # Save edited STTM
                selected_df.to_csv(sttm_path, index=False)

                with st.spinner("⚙️ Executing Bronze layer and generating Silver STTM..."):
                    # Transition: approved Bronze STTM triggers Bronze execution and Silver STTM drafting.
                    result = run_bronze_to_silver_sttm(state)
                    st.session_state.pipeline_state = result
                    st.session_state.current_run_id = result.get("run_id", st.session_state.current_run_id)
                    if result.get("error"):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.session_state.phase = "silver_sttm"
                        st.rerun()
        else:
            st.error("Bronze STTM file not found.")


    # ========== PHASE 3: SILVER STTM REVIEW ==========
    elif st.session_state.phase == "silver_sttm":
        st.header("🥈 Phase 3: Review Silver Layer STTM")
        state = st.session_state.pipeline_state

        st.info("**Silver Layer**: Data cleansing and standardization. Review transformations below.")

        sttm_path = state.get("sttm_silver_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor_df(df)

            st.write(f"**Total Transformations:** {len(df)}")

            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("", default=True),
                    "transformation_logic": st.column_config.TextColumn("Transformation Logic", width="large"),
                },
                key="silver_sttm_editor",
                height=500
            )

            selected_df = _extract_selected_rows(edited_df)

            if selected_df.empty:
                st.warning("Select at least one row to continue.")

            if st.button("✅ Approve & Continue", type="primary", use_container_width=True, disabled=selected_df.empty):
                selected_df.to_csv(sttm_path, index=False)

                with st.spinner("⚙️ Executing Silver layer and generating Gold STTM..."):
                    # Transition: approved Silver STTM triggers Silver execution and Gold STTM drafting.
                    result = run_silver_to_gold_sttm(state)
                    st.session_state.pipeline_state = result
                    st.session_state.current_run_id = result.get("run_id", st.session_state.current_run_id)
                    if result.get("error"):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.session_state.phase = "gold_sttm"
                        st.rerun()
        else:
            st.error("Silver STTM file not found.")


    # ========== PHASE 4: GOLD STTM REVIEW ==========
    elif st.session_state.phase == "gold_sttm":
        st.header("🥇 Phase 4: Review Gold Layer STTM")
        state = st.session_state.pipeline_state

        st.info("**Gold Layer**: Business aggregations and analytics. Review final transformations below.")

        sttm_path = state.get("sttm_gold_path", "")
        if sttm_path and Path(sttm_path).exists():
            df = pd.read_csv(sttm_path)
            editor_df = _prepare_sttm_editor_df(df)

            st.write(f"**Total Transformations:** {len(df)}")

            edited_df = st.data_editor(
                editor_df,
                use_container_width=True,
                num_rows="fixed",
                hide_index=True,
                column_config={
                    SELECTION_COL: st.column_config.CheckboxColumn("", default=True),
                    "transformation_logic": st.column_config.TextColumn("Transformation Logic", width="large"),
                },
                key="gold_sttm_editor",
                height=500
            )

            selected_df = _extract_selected_rows(edited_df)

            if selected_df.empty:
                st.warning("Select at least one row to continue.")

            if st.button("✅ Approve & Execute", type="primary", use_container_width=True, disabled=selected_df.empty):
                selected_df.to_csv(sttm_path, index=False)

                with st.spinner("⚙️ Executing Gold layer and generating report..."):
                    # Transition: final approval executes Gold and starts report synthesis.
                    result = run_gold_and_report(state)
                    st.session_state.pipeline_state = result
                    st.session_state.current_run_id = result.get("run_id", st.session_state.current_run_id)
                    if result.get("error"):
                        st.error(f"❌ Error: {result['error']}")
                    else:
                        st.session_state.phase = "report"
                        st.rerun()
        else:
            st.error("Gold STTM file not found.")


    # ========== PHASE 5: REPORT ==========
    elif st.session_state.phase == "report":
        st.header("📊 Phase 5: Executive Report")
        state = st.session_state.pipeline_state
        st.session_state.report_complete = True

        report_path = state.get("report_path", "")
        if report_path and Path(report_path).exists():
            # Render final artifact inline so users can inspect narrative + evidence before download.
            with open(report_path, "r", encoding="utf-8") as f:
                report_html = f.read()
            st.components.v1.html(report_html, height=2000, scrolling=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📥 Download Report"):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download HTML Report",
                            data=f.read(),
                            file_name="report.html",
                            mime="text/html"
                        )

            with col2:
                if st.button("🔄 Start New Analysis"):
                    st.session_state.report_complete = False
                    _reset_analysis_session()
                    st.rerun()
        else:
            st.error("Report file not found. Please check the pipeline execution.")

