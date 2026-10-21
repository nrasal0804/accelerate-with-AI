import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import pandas as pd
import pytest
from core.audit import AuditLogger


class TestAuditLogger:
    def test_log_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
        logger = AuditLogger("test-run-123")
        logger.log("test_agent", "test_action", detail="hello")

        assert logger.log_file.exists()
        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["agent"] == "test_agent"
        assert logs[0]["action"] == "test_action"
        assert logs[0]["detail"] == "hello"

    def test_multiple_logs_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
        logger = AuditLogger("test-run-456")
        logger.log("agent1", "action1")
        logger.log("agent2", "action2")

        logs = logger.get_logs()
        assert len(logs) == 2

    def test_get_logs_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
        logger = AuditLogger("nonexistent")
        logs = logger.get_logs()
        assert logs == []
