"""Tests for silver_agent.py.

Strategy
--------
- ``_apply_silver_rules`` is pure Python (reads Parquet, writes Parquet, no LLM).
  These tests exercise it directly for full coverage of cleansing logic.
- ``execute_silver`` is the AI agent entry point.  We mock ``create_agent`` so
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

def _make_silver_sttm(tmp_path, rows):
    import pandas as pd
    sttm_path = tmp_path / "sttm_silver.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _passthrough_sttm(tmp_path, source_table="test_bronze.parquet", columns=("id", "value")):
    rows = []
    for col in columns:
        rows.append({
            "source_table": source_table, "source_column": col, "target_column": col,
            "transformation_type": "Direct", "transformation_logic": "Passthrough",
        })
    return _make_silver_sttm(tmp_path, rows)


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
# _apply_silver_rules — pure-Python core logic
# ---------------------------------------------------------------------------

class TestApplySilverRules:

    def test_output_written_for_each_input(self, tmp_path, monkeypatch):
        """One Silver Parquet is produced per Bronze input file."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).to_parquet(bronze_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path)

        from agents.silver_agent import _apply_silver_rules
        with patch("agents.silver_agent.AuditLogger"):
            results = _apply_silver_rules([str(bronze_path)], str(sttm_path), "run-20")

        assert len(results) == 1
        df = pd.read_parquet(results[0])
        assert df.shape[0] == 2

    def test_null_handling_fill_mean(self, tmp_path, monkeypatch):
        """transformation_logic='fill null with mean' must replace NaN with the column mean."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1, 2, 3], "value": [10.0, None, 30.0]}).to_parquet(bronze_path, index=False)

        sttm_path = _make_silver_sttm(tmp_path, [
            {"source_table": "test_bronze.parquet", "source_column": "id",    "target_column": "id",    "transformation_type": "Direct", "transformation_logic": "Passthrough"},
            {"source_table": "test_bronze.parquet", "source_column": "value", "target_column": "value", "transformation_type": "Direct", "transformation_logic": "fill null with mean"},
        ])

        from agents.silver_agent import _apply_silver_rules
        with patch("agents.silver_agent.AuditLogger"):
            results = _apply_silver_rules([str(bronze_path)], str(sttm_path), "run-21")

        df = pd.read_parquet(results[0])
        assert df["value"].isnull().sum() == 0
        assert df["value"].iloc[1] == pytest.approx(20.0)  # mean of 10 and 30

    def test_dedup_removes_duplicate_rows(self, tmp_path, monkeypatch):
        """transformation_logic containing 'deduplic' must drop duplicate rows."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1, 1, 2], "value": [5.0, 5.0, 6.0]}).to_parquet(bronze_path, index=False)

        sttm_path = _make_silver_sttm(tmp_path, [
            {"source_table": "test_bronze.parquet", "source_column": "id", "target_column": "id", "transformation_type": "Direct", "transformation_logic": "deduplicate rows"},
        ])

        from agents.silver_agent import _apply_silver_rules
        with patch("agents.silver_agent.AuditLogger"):
            results = _apply_silver_rules([str(bronze_path)], str(sttm_path), "run-22")

        df = pd.read_parquet(results[0])
        assert df.shape[0] == 2  # duplicate removed

    def test_column_rename_applied(self, tmp_path, monkeypatch):
        """A rename rule must produce the target column and drop the source column."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"old_col": [1, 2]}).to_parquet(bronze_path, index=False)

        sttm_path = _make_silver_sttm(tmp_path, [
            {"source_table": "test_bronze.parquet", "source_column": "old_col", "target_column": "new_col", "transformation_type": "Indirect", "transformation_logic": "Rename", "null_handling": "", "dedup": "no"},
        ])

        from agents.silver_agent import _apply_silver_rules
        with patch("agents.silver_agent.AuditLogger"):
            results = _apply_silver_rules([str(bronze_path)], str(sttm_path), "run-23")

        df = pd.read_parquet(results[0])
        assert "new_col" in df.columns
        assert "old_col" not in df.columns

    def test_output_is_valid_parquet(self, tmp_path, monkeypatch):
        """Output file must be readable as Parquet."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1]}).to_parquet(bronze_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path, columns=("id",))

        from agents.silver_agent import _apply_silver_rules
        with patch("agents.silver_agent.AuditLogger"):
            results = _apply_silver_rules([str(bronze_path)], str(sttm_path), "run-24")

        df = pd.read_parquet(results[0])
        assert df is not None


# ---------------------------------------------------------------------------
# execute_silver — AI agent entry point
# ---------------------------------------------------------------------------

class TestExecuteSilver:

    def test_execute_silver_returns_paths(self, tmp_path, monkeypatch):
        """execute_silver must return a non-empty list of output paths."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1], "value": [9.0]}).to_parquet(bronze_path, index=False)
        sttm_path = _passthrough_sttm(tmp_path)

        from agents.silver_agent import execute_silver

        def fake_create_agent(llm, tools, system_prompt):
            return _mock_agent_that_calls_tool(tools[0])  # tools[0] is a StructuredTool

        with patch("agents.silver_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.silver_agent.AuditLogger"):
            results = execute_silver([str(bronze_path)], str(sttm_path), "run-30",
                                     task_description="Cleanse test files for run-30.")

        assert isinstance(results, list)
        assert len(results) == 1

    def test_execute_silver_forwards_task_description(self, tmp_path, monkeypatch):
        """A custom task_description must reach the agent's HumanMessage."""
        monkeypatch.setattr("agents.silver_agent.SILVER_DIR", tmp_path)
        bronze_path = tmp_path / "test_bronze.parquet"
        pd.DataFrame({"id": [1]}).to_parquet(bronze_path, index=False)
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

        from agents.silver_agent import execute_silver
        with patch("agents.silver_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.silver_agent.AuditLogger"):
            execute_silver([str(bronze_path)], str(sttm_path), "run-31", task_description="custom silver task")

        assert any("custom silver task" in str(getattr(m, "content", "")) for m in captured)

