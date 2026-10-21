import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.reporter import (
    generate_chart_from_spec,
    generate_key_metrics,
    _make_reporter_tools,
    _extract_analysis,
    generate_report,
)


ANALYSIS_JSON = {
    "direct_answer": {
        "question": "Revenue by region?",
        "answer": "West: $400, East: $300",
        "why": "Direct sum from query",
        "approach": "SELECT region, SUM(revenue) GROUP BY region",
    },
    "charts": [],
    "detailed_analysis": "West leads East in revenue.",
}


def _mock_reporter_agent(gold_parquet_path: str, analysis: dict = ANALYSIS_JSON):
    """Mock create_agent that calls the real load + execute tools, then returns analysis JSON."""
    table_stem = Path(gold_parquet_path).stem.replace("-", "_").replace(" ", "_")

    def fake_create_agent(llm, tools, **kwargs):
        mock_agent = MagicMock()

        def invoke(inputs):
            load_tool, query_tool = tools[0], tools[1]
            messages = [
                MagicMock(content=load_tool.invoke({})),
                MagicMock(content=query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem}"})),
                MagicMock(content=json.dumps(analysis)),
            ]
            return {"messages": messages}

        mock_agent.invoke = invoke
        return mock_agent

    return fake_create_agent


# ---------------------------------------------------------------------------
# generate_chart_from_spec
# ---------------------------------------------------------------------------
class TestGenerateChartFromSpec:
    def _df(self):
        return pd.DataFrame({
            "region": ["East", "West", "North"],
            "revenue": [300, 400, 200],
        })

    def test_bar_chart_returns_html(self):
        spec = {"type": "bar", "title": "Revenue", "x_column": "region", "y_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 1)
        assert "<div" in html

    def test_pie_chart_returns_html(self):
        spec = {"type": "pie", "title": "Share", "labels_column": "region", "values_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 2)
        assert "<div" in html

    def test_unknown_chart_type_returns_empty(self):
        spec = {"type": "heatmap", "title": "Unknown", "x_column": "region"}
        html = generate_chart_from_spec(self._df(), spec, 3)
        assert html == ""

    def test_bad_column_does_not_raise(self):
        spec = {"type": "bar", "title": "Bad", "x_column": "nonexistent", "y_column": "revenue"}
        # Should return empty string on error, not raise
        html = generate_chart_from_spec(self._df(), spec, 4)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# generate_key_metrics
# ---------------------------------------------------------------------------
class TestGenerateKeyMetrics:
    def test_returns_expected_keys(self):
        df = pd.DataFrame({"a": [1, None], "b": ["x", "y"]})
        metrics = generate_key_metrics(df)
        assert "total_rows" in metrics
        assert "total_columns" in metrics
        assert "missing_values" in metrics
        assert metrics["total_rows"] == 2
        assert metrics["missing_values"] == 1


# ---------------------------------------------------------------------------
# _make_reporter_tools
# ---------------------------------------------------------------------------
class TestMakeReporterTools:
    def test_load_tool_returns_catalog(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_sales.parquet"
        df.to_parquet(str(p), index=False)

        load_tool, _, scratchpad, conn = _make_reporter_tools([str(p)], "run-r1")
        try:
            result = load_tool.invoke({})
            catalog = json.loads(result)
            assert "gold_sales" in catalog
            assert "columns" in catalog["gold_sales"]
            assert "id" in catalog["gold_sales"]["columns"]
        finally:
            conn.close()

    def test_execute_tool_stores_result_in_scratchpad(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_data.parquet"
        df.to_parquet(str(p), index=False)

        load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r2")
        try:
            load_tool.invoke({})
            query_tool.invoke({"sql_query": "SELECT * FROM gold_data"})
            assert "result_df" in scratchpad
            assert len(scratchpad["result_df"]) == 2
            assert scratchpad["sql_query"] == "SELECT * FROM gold_data"
        finally:
            conn.close()

    def test_execute_tool_returns_error_on_bad_sql(self, tmp_path):
        df = pd.DataFrame({"id": [1]})
        p = tmp_path / "gold_x.parquet"
        df.to_parquet(str(p), index=False)

        load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r3")
        try:
            load_tool.invoke({})
            result = query_tool.invoke({"sql_query": "SELECT * FROM nonexistent_table"})
            parsed = json.loads(result)
            assert "error" in parsed
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# _extract_analysis
# ---------------------------------------------------------------------------
class TestExtractAnalysis:
    def test_extracts_direct_answer_json(self):
        result = {"messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]}
        analysis = _extract_analysis(result)
        assert analysis["direct_answer"]["answer"] == "West: $400, East: $300"

    def test_handles_json_fences(self):
        content = f"```json\n{json.dumps(ANALYSIS_JSON)}\n```"
        result = {"messages": [MagicMock(content=content)]}
        analysis = _extract_analysis(result)
        assert "direct_answer" in analysis

    def test_returns_empty_dict_when_no_direct_answer(self):
        result = {"messages": [MagicMock(content='{"other_key": "value"}')]}
        assert _extract_analysis(result) == {}

    def test_returns_empty_dict_on_no_messages(self):
        assert _extract_analysis({"messages": []}) == {}


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:
    def test_saves_html_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)

        with patch("agents.reporter.create_agent",
                   side_effect=_mock_reporter_agent(str(gold_path))), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)], "Analyse revenue by region", "run-rpt1",
                task_description="Generate report for run-rpt1.",
            )

        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Executive Report" in content
        assert "Analyse revenue by region" in content

    def test_uses_reporter_agent_prompt(self, tmp_path):
        df = pd.DataFrame({"region": ["East"], "revenue": [300]})
        gold_path = tmp_path / "gold_p.parquet"
        df.to_parquet(str(gold_path), index=False)

        captured = {}

        def fake_create_agent(llm, tools, **kwargs):
            captured["system_prompt"] = kwargs.get("system_prompt", "")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]
            }
            return mock_agent

        with patch("agents.reporter.create_agent", side_effect=fake_create_agent), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            generate_report(
                [str(gold_path)], "intent", "run-rpt2",
                task_description="Generate report.",
            )

        assert "load_gold_data_tool" in captured["system_prompt"]

    def test_returns_empty_string_for_no_files(self, tmp_path):
        with patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            result = generate_report([], "intent", "run-rpt3",
                                     task_description="No files.")
        assert result == ""

