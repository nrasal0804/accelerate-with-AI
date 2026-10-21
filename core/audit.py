import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from core.config import AUDIT_DIR


class AuditLogger:
    """Append-only JSONL audit trail logger."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_file = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self) -> list[dict]:
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
