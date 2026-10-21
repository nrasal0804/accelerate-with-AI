"""Tests for gold_agent.py.

Strategy
--------
- ``_apply_gold_rules`` is pure Python (reads Parquet, writes Parquet, no LLM).
  These tests exercise it directly for full coverage of materialisation logic.
- ``execute_gold`` is the AI agent entry point.  We mock ``create_agent`` so
  the LLM is never called, but we verify wiring and ``task_description`` forwarding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gold_sttm(tmp_path, rows):
    sttm_path = tmp_path / "sttm_gold.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _passthrough_sttm(tmp_path, source_table="test_silver.parquet", columns=("id", "value")):
    rows = []
    for col in columns:
        rows.append({
            "source_table": source_table, "source_column": col,
            "target_column": col, "target_table": "gold_output",
            "transformation_type": "Direct", "transformation_logic": "Passthrough",
            "join_key": "", "aggregation": "",
        })
    return _make_gold_sttm(tmp_path, rows)


def _mock_agent_that_calls_tool(tool):
    """Return a mock agent whose .invoke() calls the LangChain StructuredTool via .invoke({})."""
    mock_agent = MagicMock()
    def fake_invoke(inputs):
        tool_result = tool.invoke({})
        ai_msg = MagicMock()
        ai_msg.content = tool_result
        return {"messages": [ai_msg]}
    mock_agent.invoke.side_effect = fake_invoke
    return mock_agent


# ---------------------------------------------------------------------------
# _apply_gold_rules — pure-Python core logic
# ---------------------------------------------------------------------------

class TestApplyGoldRules:

    def test_output_written_for_each_target_table(self, tmp_path, monkeypatch):
        """One Gold Parquet is produced per distinct target_table in the STTM."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).to_parquet(silver_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path)

        from agents.gold_agent import _apply_gold_rules
        with patch("agents.gold_agent.AuditLogger"):
            results = _apply_gold_rules([str(silver_path)], str(sttm_path), "analytics", "run-40")

        assert len(results) >= 1
        df = pd.read_parquet(results[0])
        assert df.shape[0] == 2

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        """Output files must be readable as Parquet."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"id": [1]}).to_parquet(silver_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path, columns=("id",))

        from agents.gold_agent import _apply_gold_rules
        with patch("agents.gold_agent.AuditLogger"):
            results = _apply_gold_rules([str(silver_path)], str(sttm_path), "analytics", "run-41")

        assert pd.read_parquet(results[0]) is not None

    def test_multiple_target_tables(self, tmp_path, monkeypatch):
        """Two distinct target_table values must produce two output files."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"id": [1, 2], "revenue": [100.0, 200.0], "cost": [50.0, 80.0]}).to_parquet(silver_path, index=False)

        sttm_path = _make_gold_sttm(tmp_path, [
            {"source_table": "test_silver.parquet", "source_column": "id",      "target_column": "id",      "target_table": "table_a", "transformation_type": "Direct", "transformation_logic": "Passthrough", "join_key": "", "aggregation": ""},
            {"source_table": "test_silver.parquet", "source_column": "revenue", "target_column": "revenue", "target_table": "table_a", "transformation_type": "Direct", "transformation_logic": "Passthrough", "join_key": "", "aggregation": ""},
            {"source_table": "test_silver.parquet", "source_column": "id",      "target_column": "id",      "target_table": "table_b", "transformation_type": "Direct", "transformation_logic": "Passthrough", "join_key": "", "aggregation": ""},
            {"source_table": "test_silver.parquet", "source_column": "cost",    "target_column": "cost",    "target_table": "table_b", "transformation_type": "Direct", "transformation_logic": "Passthrough", "join_key": "", "aggregation": ""},
        ])

        from agents.gold_agent import _apply_gold_rules
        with patch("agents.gold_agent.AuditLogger"):
            results = _apply_gold_rules([str(silver_path)], str(sttm_path), "analytics", "run-42")

        assert len(results) == 2

    def test_column_subset_applied(self, tmp_path, monkeypatch):
        """Only columns listed in the STTM must appear in the Gold output."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"keep": [1, 2], "drop": [3, 4]}).to_parquet(silver_path, index=False)

        sttm_path = _make_gold_sttm(tmp_path, [
            {"source_table": "test_silver.parquet", "source_column": "keep", "target_column": "keep", "target_table": "gold_output", "transformation_type": "Direct", "transformation_logic": "Passthrough", "join_key": "", "aggregation": ""},
        ])

        from agents.gold_agent import _apply_gold_rules
        with patch("agents.gold_agent.AuditLogger"):
            results = _apply_gold_rules([str(silver_path)], str(sttm_path), "analytics", "run-43")

        df = pd.read_parquet(results[0])
        assert "keep" in df.columns
        assert "drop" not in df.columns


# ---------------------------------------------------------------------------
# execute_gold — AI agent entry point
# ---------------------------------------------------------------------------

class TestExecuteGold:

    def test_execute_gold_returns_paths(self, tmp_path, monkeypatch):
        """execute_gold must return a non-empty list of output paths."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"id": [1], "value": [9.0]}).to_parquet(silver_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path)

        from agents.gold_agent import execute_gold

        def fake_create_agent(llm, tools, system_prompt):
            return _mock_agent_that_calls_tool(tools[0])

        with patch("agents.gold_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.gold_agent.AuditLogger"):
            results = execute_gold([str(silver_path)], str(sttm_path), "analytics", "run-50",
                                   task_description="Materialise test files for run-50.")

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_execute_gold_forwards_task_description(self, tmp_path, monkeypatch):
        """A custom task_description must reach the agent's HumanMessage."""
        monkeypatch.setattr("agents.gold_agent.GOLD_DIR", tmp_path)
        silver_path = tmp_path / "test_silver.parquet"
        pd.DataFrame({"id": [1]}).to_parquet(silver_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path, columns=("id",))

        captured = []

        def fake_create_agent(llm, tools, system_prompt):
            mock_agent = MagicMock()
            def fake_invoke(inputs):
                captured.extend(inputs["messages"])
                result = tools[0].invoke({})
                ai_msg = MagicMock()
                ai_msg.content = result
                return {"messages": [ai_msg]}
            mock_agent.invoke.side_effect = fake_invoke
            return mock_agent

        from agents.gold_agent import execute_gold
        with patch("agents.gold_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.gold_agent.AuditLogger"):
            execute_gold([str(silver_path)], str(sttm_path), "analytics", "run-51", task_description="custom gold task")

        assert any("custom gold task" in str(getattr(m, "content", "")) for m in captured)

