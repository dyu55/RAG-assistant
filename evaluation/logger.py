"""
Query Logger.
Logs every pipeline run as structured JSONL for post-hoc analysis.
Each line in the log file is an independent JSON object.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Max bytes to read from tail of file when file is large
_MAX_TAIL_BYTES = 10_000_000  # ~10MB, roughly 20k entries at ~500 bytes/line
_ESTIMATED_LINE_BYTES = 500  # average bytes per JSONL line


class QueryLogger:
    """Append-only JSONL logger for pipeline results."""

    def __init__(self, log_dir: str | None = None):
        self.log_dir = Path(log_dir or settings.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "queries.jsonl"

    def log(self, result) -> None:
        """
        Log a PipelineResult as a JSON line.

        Args:
            result: A PipelineResult object (has .to_dict() method).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            **result.to_dict(),
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"Logged query: '{result.query[:50]}...'")
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")

    def get_recent_logs(self, n: int = 50) -> list[dict]:
        """
        Read the most recent N log entries.

        For large files (>5MB), reads only the tail to avoid loading
        the entire file into memory.
        """
        if not self.log_file.exists():
            return []

        file_size = self.log_file.stat().st_size

        if file_size > 5_000_000:
            # File is large (>5MB) — read only the tail
            # Read enough to cover n entries plus buffer
            bytes_to_read = min(n * _ESTIMATED_LINE_BYTES * 2, file_size)
            try:
                with open(self.log_file, "rb") as f:
                    f.seek(-bytes_to_read, 2)  # Seek from end
                    f.readline()  # Skip partial first line from seek
                    raw = f.read().decode("utf-8", errors="replace")
            except (OSError, ValueError):
                # Fallback: seek past beginning
                with open(self.log_file, "r", encoding="utf-8") as f:
                    raw = f.read()
            lines = raw.splitlines()
        else:
            # Small file — read entirely
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        # Parse all lines, take the last n
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug(f"Skipped malformed JSON line in {self.log_file}")
                continue

        return entries[-n:]

    def get_log_stats(self, max_entries: int = 10000) -> dict:
        """
        Get aggregate statistics from logs.

        Args:
            max_entries: Maximum number of recent entries to analyze.
        """
        entries = self.get_recent_logs(n=max_entries)

        if not entries:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "abstention_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        confidences = []
        abstentions = 0
        latencies = []

        for entry in entries:
            rel = entry.get("reliability", {})
            if rel and rel.get("confidence") is not None:
                confidences.append(rel["confidence"])
            if entry.get("should_abstain"):
                abstentions += 1
            if entry.get("total_latency_ms"):
                latencies.append(entry["total_latency_ms"])

        return {
            "total_queries": len(entries),
            "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 3),
            "abstention_rate": round(abstentions / max(len(entries), 1), 3),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
        }
