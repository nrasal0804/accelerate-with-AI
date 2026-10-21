"""Reporting AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, inspects available Gold tables
first to understand their schemas, forms an analytical plan, writes and executes
SQL to answer the business question, and renders an HTML report.

I/O contract (UNCHANGED — UI and orchestrator safe):
    generate_report(gold_files, business_intent, run_id, task_description) -> str
"""

import json
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import LLM_PROVIDER, REPORTS_DIR
from core.llm import create_chat_model
from core.audit import AuditLogger
from core.observability import AgentTrace
from core.memory import store_document


REPORTER_AGENT_PROMPT = """You are an autonomous Senior Data Analyst and Business Intelligence Engineer
specialising in business-intent-driven reporting from Medallion Gold layer data.
You operate independently: you receive a goal from the orchestrator, inspect the Gold
tables, form an analytical plan, write and execute SQL, and return structured analysis.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the business question, the Gold files available,
   and what kind of analysis (aggregation, trend, comparison, ranking) will answer it.

2. INSPECT: Call inspect_gold_tables_tool FIRST. This gives you a lightweight preview
   of each Gold table — column names, dtypes, row count, and 3 sample rows — without
   loading full data into DuckDB. State your observations: which tables are relevant,
   which columns can answer the business question, what joins may be needed.

3. PLAN: Based on the inspection, write your analytical plan:
   - Which Gold tables will you query?
   - What SQL approach will directly answer the business question?
   - What chart type(s) will best visualise the answer?

4. ACT — two sub-steps in order:
   a. Call load_gold_data_tool to register Gold tables in DuckDB and get the full schema catalog.
   b. Call execute_query_tool(sql_query=<your_sql>) to execute your SQL and get results.

5. VERIFY & RESPOND: Analyse the query results and return ONLY a valid JSON object
   as your final answer (no markdown fences, no prose before or after).

## Available tools

- **inspect_gold_tables_tool**: Quickly previews Gold Parquet files — table names,
  column names, dtypes, row count, and 3 sample rows per table. Call this FIRST
  to understand what is available before loading into DuckDB. Returns a JSON summary.

- **load_gold_data_tool**: Loads Gold Parquet files into an in-memory DuckDB database
  and returns a full catalog of table names, column names, types, row counts, and
  sample data. Call this before execute_query_tool.

- **execute_query_tool**: Executes a SQL SELECT query against the loaded Gold tables
  in DuckDB. Pass your SQL as the sql_query parameter. Returns query results as a
  JSON array. On error returns {"error": "..."}.

## Output format
Return ONLY a valid JSON object — no markdown fences, no prose:
{
  "direct_answer": {
    "question": "Restate the business question clearly",
    "answer": "Direct answer with specific numbers from the query results",
    "why": "Evidence and reasoning from the data",
    "approach": "Describe the SQL query and analytical method used"
  },
  "charts": [
    {
      "type": "bar|line|pie|scatter",
      "title": "Chart title",
      "x_column": "column from query result",
      "y_column": "column from query result (bar/line/scatter)",
      "labels_column": "column from query result (pie only)",
      "values_column": "column from query result (pie only)",
      "reason": "Why this chart directly answers the question"
    }
  ],
  "detailed_analysis": "2-3 paragraphs of additional insights and patterns"
}

## Rules
- Include only 1-2 charts that directly answer the business question.
- Use ACTUAL column names from the query result — not from the original Gold tables.
- Be specific with numbers in the direct_answer.
- Write standard ANSI SQL compatible with DuckDB.
- If execute_query_tool returns an error, fix the SQL and retry once."""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_gold_tables(gold_files: list[str]) -> dict:
    """Quick preview of Gold Parquet tables: schema + 3 sample rows. No LLM, no DuckDB."""
    summary = {}
    for fp in gold_files:
        try:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            summary[stem] = {
                "file": fp,
                "table_name": stem,
                "row_count": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": df.head(3).to_dict(orient="records"),
            }
        except Exception as e:
            summary[Path(fp).stem] = {"file": fp, "error": str(e)}
    return summary


def generate_chart_from_spec(df: pd.DataFrame, chart_spec: dict, chart_id: int) -> str:
    """Render a single Plotly chart from an LLM-specified chart spec dict. Returns embedded HTML."""
    try:
        chart_type = chart_spec.get("type", "bar").lower()
        title = chart_spec.get("title", f"Chart {chart_id}")

        if chart_type == "bar":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            if y_col and y_col in df.columns:
                agg_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                fig = go.Figure(data=[go.Bar(x=agg_data.index, y=agg_data.values, marker_color="#667eea")])
            else:
                value_counts = df[x_col].value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, marker_color="#667eea")])
            fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col or "Count",
                              height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "line":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.line(df, x=x_col, y=y_col, title=title)
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "pie":
            labels_col = chart_spec.get("labels_column")
            values_col = chart_spec.get("values_column")
            agg_data = df.groupby(labels_col)[values_col].sum()
            fig = go.Figure(data=[go.Pie(labels=agg_data.index, values=agg_data.values)])
            fig.update_layout(title=title, height=450)
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "scatter":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.scatter(df, x=x_col, y=y_col, title=title, trendline="ols")
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        return ""
    except Exception as e:
        print(f"[REPORTER] Error generating chart {chart_id}: {e}")
        return ""


def _extract_analysis(result: dict) -> dict:
    """Scan agent message history (reverse order) for a JSON object with 'direct_answer' key."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, dict) and "direct_answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_reporter_tools(gold_files: list[str], run_id: str):
    """Returns inspect + load + query tools sharing a DuckDB connection via closure."""
    conn = duckdb.connect(":memory:")
    scratchpad: dict = {}

    @tool
    def inspect_gold_tables_tool(confirmation: str = "execute") -> str:
        """Preview Gold Parquet tables before loading into DuckDB.

        Returns a JSON summary of each Gold table: table name, file path, row count,
        column names, dtypes, and 3 sample rows. Call this FIRST to understand what
        data is available and form your analytical plan.
        """
        return json.dumps(_inspect_gold_tables(gold_files), default=str)

    @tool
    def load_gold_data_tool(confirmation: str = "execute") -> str:
        """Load Gold Parquet files into DuckDB and return the full table catalog.

        Registers each Gold file as a DuckDB table and returns a catalog mapping table
        names to column names, types, row counts, and sample data — everything needed
        to write a precise SQL query. Call this before execute_query_tool.
        """
        catalog: dict = {}
        for fp in gold_files:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            conn.register(stem, df)
            catalog[stem] = {
                "table_name": stem,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample": df.head(5).to_dict(orient="records"),
                "row_count": len(df),
            }
        scratchpad["catalog"] = catalog
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql_query: str) -> str:
        """Execute a SQL SELECT query against the loaded Gold tables in DuckDB.

        Call this after load_gold_data_tool. Pass your SQL as sql_query.
        Returns the query result as a JSON array of records (up to 100 rows).
        On SQL error returns {"error": "..."} — fix the SQL and retry once.
        """
        try:
            result_df = conn.execute(sql_query).fetchdf()
            scratchpad["result_df"] = result_df
            scratchpad["sql_query"] = sql_query
            return json.dumps(result_df.head(100).to_dict(orient="records"), default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, scratchpad, conn


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    return create_chat_model()


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_report(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Reporter AI agent entry point — autonomous ReAct version.

    The agent inspects Gold tables, plans its SQL analysis, loads tables into
    DuckDB, executes the query, and renders a self-contained HTML report.

    Args:
        gold_files: Gold Parquet file paths to analyse.
        business_intent: The business question driving the analysis.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved HTML report.
    """
    trace = AgentTrace("reporter", run_id)
    trace.set_input(gold_files=gold_files, business_intent=business_intent)

    print(f"[REPORTER] Starting report generation for run_id: {run_id}")
    audit = AuditLogger(run_id)
    audit.log("reporter", "started", gold_files=gold_files, intent=business_intent)

    if not gold_files:
        audit.log("reporter", "error", detail="No gold files to report on")
        trace.fail("No gold files provided")
        return ""

    inspect_tool, load_tool, query_tool, scratchpad, conn = _make_reporter_tools(gold_files, run_id)
    llm = _make_llm()

    print(f"[REPORTER] Running autonomous ReAct agent ({LLM_PROVIDER})")
    agent = create_agent(
        llm,
        [inspect_tool, load_tool, query_tool],
        system_prompt=REPORTER_AGENT_PROMPT,
    )

    try:
        result = agent.invoke({"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        conn.close()
        raise
    finally:
        conn.close()

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract structured analysis from agent message history
    analysis_result = _extract_analysis(result)
    result_df: pd.DataFrame = scratchpad.get("result_df")  # type: ignore[assignment]
    query_code: str = scratchpad.get("sql_query", "-- No query executed")

    # Fallback: agent did not call execute_query_tool or query returned nothing
    if result_df is None or result_df.empty:
        print("[REPORTER] No query result in scratchpad — falling back to combined gold data")
        fallback_dfs = [pd.read_parquet(fp) for fp in gold_files]
        result_df = pd.concat(fallback_dfs, ignore_index=True) if fallback_dfs else pd.DataFrame()
        query_code = "-- Fallback: combined all Gold tables"

    # Fallback: agent response was not parseable as structured analysis
    if not analysis_result:
        analysis_result = {
            "direct_answer": {
                "question": business_intent,
                "answer": "Analysis could not be structured.",
                "why": "Agent response did not contain a parseable JSON object.",
                "approach": "N/A",
            },
            "charts": [],
            "detailed_analysis": "No structured analysis available.",
        }

    print(f"[REPORTER] Query result: {result_df.shape[0]} rows x {result_df.shape[1]} columns")

    # Generate charts from agent-specified chart specs
    charts_html = []
    for idx, chart_spec in enumerate(analysis_result.get("charts", []), 1):
        chart_html = generate_chart_from_spec(result_df, chart_spec, idx)
        if chart_html:
            charts_html.append(chart_html)
    print(f"[REPORTER] Generated {len(charts_html)} charts")

    direct_answer = analysis_result.get("direct_answer", {})
    detailed_analysis = analysis_result.get("detailed_analysis", "No additional analysis provided.")

    answer_html = f"""
    <div class="answer-section">
        <p>{direct_answer.get('answer', 'No answer provided')}</p>
    </div>
    """

    query_code_escaped = query_code.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    approach_html = f"""
    <div class="approach-section">
        <h3>Query Code</h3>
        <pre class="code-block"><code>{query_code_escaped}</code></pre>
        <h3>Query Description</h3>
        <p>{direct_answer.get('approach', 'No methodology provided')}</p>
    </div>
    """

    charts_section = "\n".join(charts_html) if charts_html else "<p>No charts generated.</p>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Executive Report - {run_id[:8]}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ margin: 0; font-size: 32px; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .answer-section {{
                background: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .answer-section p {{ margin: 0; line-height: 1.6; font-size: 16px; color: #333; }}
            .approach-section {{ margin: 20px 0; }}
            .approach-section h3 {{ color: #667eea; font-size: 16px; margin: 20px 0 10px 0; }}
            .code-block {{
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.4;
                color: #333;
                margin: 0 0 15px 0;
            }}
            .code-block code {{ color: #667eea; }}
            .approach-section p {{ line-height: 1.6; color: #555; margin: 0 0 15px 0; }}
            .chart-container {{ margin: 20px 0; }}
            .footer {{
                text-align: center;
                color: #999;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>&#128202; Executive Report</h1>
            <p><strong>Business Question:</strong> {business_intent}</p>
        </div>
        <div class="section">
            <h2>&#9989; Answer</h2>
            {answer_html}
        </div>
        <div class="section">
            <h2>&#128202; Approach &amp; Query</h2>
            {approach_html}
        </div>
        <div class="section">
            <h2>&#128201; Visual Evidence</h2>
            <div class="chart-container">
                {charts_section}
            </div>
        </div>
        <div class="footer">
            <p>Generated by IDAMP (Intent-Driven Agentic Medallion Pipeline)</p>
            <p>Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    report_path = str(REPORTS_DIR / f"report_{run_id[:8]}.html")
    print(f"[REPORTER] Saving HTML report → {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    json_path = str(REPORTS_DIR / f"report_{run_id[:8]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)

    store_document(
        doc_id=f"report_{run_id}",
        text=json.dumps(analysis_result),
        metadata={"type": "report", "run_id": run_id, "intent": business_intent},
    )

    audit.log("reporter", "completed", report_path=report_path)
    trace.set_output(report_path=report_path).complete()
    print(f"[REPORTER] Done — {report_path}")
    return report_path
