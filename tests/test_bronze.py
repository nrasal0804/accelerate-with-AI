"""Tests for bronze_agent.py.

Strategy
--------
- ``_apply_bronze_rules`` is pure Python (reads CSV, writes Parquet, no LLM).
  These tests exercise it directly for full coverage of transformation logic.
- ``execute_bronze`` is the AI agent entry point.  We mock ``create_agent`` so
  the LLM is never called, but we verify the agent is wired up correctly and
  that ``task_description`` is forwarded when supplied.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_sttm_csv(tmp_path, extra_rows=None):
    rows = [
        {"source_table": "test.csv", "source_column": "id",    "target_column": "id",              "transformation_type": "Direct",   "transformation_logic": "Passthrough"},
        {"source_table": "test.csv", "source_column": "value", "target_column": "value",           "transformation_type": "Direct",   "transformation_logic": "Passthrough"},
        {"source_table": "test.csv", "source_column": "",      "target_column": "_load_timestamp", "transformation_type": "Indirect", "transformation_logic": "Current UTC timestamp"},
        {"source_table": "test.csv", "source_column": "",      "target_column": "_source_file",    "transformation_type": "Indirect", "transformation_logic": "Source file path"},
    ]
    if extra_rows:
        rows.extend(extra_rows)
    sttm_path = tmp_path / "sttm_bronze.csv"
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    return sttm_path


def _mock_agent_that_calls_tool(tool):
    """Return a mock agent whose .invoke() calls the LangChain tool via .invoke({})."""
    mock_agent = MagicMock()
    def fake_invoke(inputs):
        tool_result = tool.invoke({})
        ai_msg = MagicMock()
        ai_msg.content = tool_result
        return {"messages": [ai_msg]}
    mock_agent.invoke.side_effect = fake_invoke
    return mock_agent


# ---------------------------------------------------------------------------
# _apply_bronze_rules - pure-Python core logic
# ---------------------------------------------------------------------------

class TestApplyBronzeRules:

    def test_metadata_columns_added(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        from agents.bronze_agent import _apply_bronze_rules
        with patch("agents.bronze_agent.AuditLogger"):
            results = _apply_bronze_rules([str(csv_path)], str(sttm_path), "run-1")

        assert len(results) == 1
        df = pd.read_parquet(results[0])
        meta_cols = set(df.columns)
        assert meta_cols & {"_load_timestamp", "load_timestamp"}, "timestamp column missing"
        assert meta_cols & {"_source_file", "source_file"}, "source_file column missing"
        assert df.shape[0] == 2

    def test_row_count_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": range(50), "value": range(50)}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        from agents.bronze_agent import _apply_bronze_rules
        with patch("agents.bronze_agent.AuditLogger"):
            results = _apply_bronze_rules([str(csv_path)], str(sttm_path), "run-2")

        df = pd.read_parquet(results[0])
        assert df.shape[0] == 50

    def test_column_rename_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"old_name": [1, 2]}).to_csv(csv_path, index=False)

        sttm_path = tmp_path / "sttm_bronze.csv"
        pd.DataFrame([
            {"source_table": "test.csv", "source_column": "old_name", "target_column": "new_name",        "transformation_type": "Indirect", "transformation_logic": "Rename"},
            {"source_table": "test.csv", "source_column": "",         "target_column": "_load_timestamp", "transformation_type": "Indirect", "transformation_logic": "Current UTC timestamp"},
            {"source_table": "test.csv", "source_column": "",         "target_column": "_source_file",    "transformation_type": "Indirect", "transformation_logic": "Source file path"},
        ]).to_csv(sttm_path, index=False)

        from agents.bronze_agent import _apply_bronze_rules
        with patch("agents.bronze_agent.AuditLogger"):
            results = _apply_bronze_rules([str(csv_path)], str(sttm_path), "run-3")

        df = pd.read_parquet(results[0])
        assert "new_name" in df.columns
        assert "old_name" not in df.columns

    def test_multiple_files_produce_multiple_outputs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_a = tmp_path / "a.csv"
        csv_b = tmp_path / "b.csv"
        pd.DataFrame({"id": [1]}).to_csv(csv_a, index=False)
        pd.DataFrame({"id": [2]}).to_csv(csv_b, index=False)

        sttm_path = tmp_path / "sttm_bronze.csv"
        pd.DataFrame([
            {"source_table": "", "source_column": "", "target_column": "_load_timestamp", "transformation_type": "Indirect", "transformation_logic": "Current UTC timestamp"},
            {"source_table": "", "source_column": "", "target_column": "_source_file",    "transformation_type": "Indirect", "transformation_logic": "Source file path"},
        ]).to_csv(sttm_path, index=False)

        from agents.bronze_agent import _apply_bronze_rules
        with patch("agents.bronze_agent.AuditLogger"):
            results = _apply_bronze_rules([str(csv_a), str(csv_b)], str(sttm_path), "run-4")

        assert len(results) == 2

    def test_output_file_is_valid_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1]}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        from agents.bronze_agent import _apply_bronze_rules
        with patch("agents.bronze_agent.AuditLogger"):
            results = _apply_bronze_rules([str(csv_path)], str(sttm_path), "run-5")

        assert pd.read_parquet(results[0]) is not None


# ---------------------------------------------------------------------------
# execute_bronze - AI agent entry point
# ---------------------------------------------------------------------------

class TestExecuteBronze:

    def test_execute_bronze_returns_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1], "value": [9]}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        from agents.bronze_agent import execute_bronze

        def fake_create_agent(llm, tools, system_prompt):
            return _mock_agent_that_calls_tool(tools[0])

        with patch("agents.bronze_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.bronze_agent.AuditLogger"):
            results = execute_bronze([str(csv_path)], str(sttm_path), "run-10",
                                     task_description="Ingest test files for run-10.")

        assert isinstance(results, list)
        assert len(results) == 1

    def test_execute_bronze_accepts_task_description(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1]}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        captured_messages = []

        def fake_create_agent(llm, tools, system_prompt):
            mock_agent = MagicMock()
            def fake_invoke(inputs):
                captured_messages.extend(inputs["messages"])
                tool_result = tools[0].invoke({})
                ai_msg = MagicMock()
                ai_msg.content = tool_result
                return {"messages": [ai_msg]}
            mock_agent.invoke.side_effect = fake_invoke
            return mock_agent

        from agents.bronze_agent import execute_bronze
        with patch("agents.bronze_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.bronze_agent.AuditLogger"):
            execute_bronze([str(csv_path)], str(sttm_path), "run-11", task_description="custom task")

        assert any("custom task" in str(getattr(m, "content", "")) for m in captured_messages)

    def test_execute_bronze_uses_system_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.bronze_agent.BRONZE_DIR", tmp_path)
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({"id": [1]}).to_csv(csv_path, index=False)
        sttm_path = _make_sttm_csv(tmp_path)

        captured_prompt = []

        def fake_create_agent(llm, tools, system_prompt):
            captured_prompt.append(system_prompt)
            return _mock_agent_that_calls_tool(tools[0])

        from agents.bronze_agent import execute_bronze, BRONZE_AGENT_PROMPT
        with patch("agents.bronze_agent.create_agent", side_effect=fake_create_agent), \
             patch("agents.bronze_agent.AuditLogger"):
            execute_bronze([str(csv_path)], str(sttm_path), "run-12",
                           task_description="Ingest for run-12.")

        assert captured_prompt[0] == BRONZE_AGENT_PROMPT
