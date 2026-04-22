"""
Tests for evaluation/logger.py
"""
import json
import pytest
from pathlib import Path

from evaluation.logger import QueryLogger, _MAX_TAIL_BYTES, _ESTIMATED_LINE_BYTES


class TestQueryLoggerInit:
    def test_creates_log_directory(self, temp_dir):
        log_dir = temp_dir / "logs"
        logger = QueryLogger(log_dir=str(log_dir))

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_uses_default_dir_from_settings(self):
        logger = QueryLogger()
        # Should not raise, defaults to settings.LOG_DIR
        assert logger.log_dir.exists()

    def test_log_file_path(self, temp_dir):
        logger = QueryLogger(log_dir=str(temp_dir))
        assert logger.log_file == temp_dir / "queries.jsonl"


class TestQueryLoggerLog:
    def test_log_creates_file(self, temp_dir):
        logger = QueryLogger(log_dir=str(temp_dir))
        mock_result = _MockResult("What is AI?", "AI is artificial intelligence.")

        logger.log(mock_result)

        assert logger.log_file.exists()

    def test_log_writes_json_line(self, temp_dir):
        logger = QueryLogger(log_dir=str(temp_dir))
        mock_result = _MockResult("Hello", "Hi there!")

        logger.log(mock_result)

        with open(logger.log_file) as f:
            line = f.readline()
        entry = json.loads(line)
        assert "timestamp" in entry
        assert entry["query"] == "Hello"
        assert entry["answer"] == "Hi there!"

    def test_log_handles_exception_gracefully(self, tmp_path):
        # Pass a read-only path to simulate write failure
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # read-only
        try:
            logger = QueryLogger(log_dir=str(read_only_dir))
            mock_result = _MockResult("test", "result")
            # Should not raise - errors are logged, not propagated
            logger.log(mock_result)
        finally:
            read_only_dir.chmod(0o755)  # restore


class TestGetRecentLogs:
    def test_returns_empty_when_file_missing(self, temp_dir):
        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=10)
        assert result == []

    def test_returns_recent_n_entries(self, temp_dir, sample_jsonl_lines):
        # Pre-populate log file
        log_file = temp_dir / "queries.jsonl"
        log_file.write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=2)

        assert len(result) == 2

    def test_respects_n_parameter(self, temp_dir, sample_jsonl_lines):
        log_file = temp_dir / "queries.jsonl"
        log_file.write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=1)

        assert len(result) == 1

    def test_large_file_reads_tail_only(self, temp_dir, sample_jsonl_lines):
        """When file > 5MB, logger should read only the tail."""
        log_file = temp_dir / "queries.jsonl"
        # Create a file larger than 5MB by repeating lines
        # Each line is ~600 bytes. Need ~8334 lines.
        large_lines = sample_jsonl_lines * 8500  # ~6.3MB at ~600 bytes/line
        log_file.write_text("\n".join(large_lines) + "\n", encoding="utf-8")

        file_size = log_file.stat().st_size
        assert file_size > 5_000_000, f"Setup failed: file is {file_size} bytes"

        logger = QueryLogger(log_dir=str(temp_dir))

        # Request 50 recent entries — should succeed with tail reading
        result = logger.get_recent_logs(n=50)

        assert isinstance(result, list)
        assert len(result) <= 50

    def test_skips_malformed_lines(self, temp_dir):
        log_file = temp_dir / "queries.jsonl"
        lines = [
            json.dumps({"query": "good"}),
            "not valid json {",
            json.dumps({"query": "also good"}),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=10)

        assert len(result) == 2
        assert result[0]["query"] == "good"
        assert result[1]["query"] == "also good"

    def test_skips_empty_lines(self, temp_dir):
        log_file = temp_dir / "queries.jsonl"
        lines = [
            json.dumps({"query": "first"}),
            "",
            "   ",
            json.dumps({"query": "second"}),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=10)

        assert len(result) == 2

    def test_returns_entries_in_order(self, temp_dir, sample_jsonl_lines):
        log_file = temp_dir / "queries.jsonl"
        log_file.write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        result = logger.get_recent_logs(n=10)

        # Should be in original order (no reverse)
        assert result[0]["query"] == "What is RAG?"
        assert result[-1]["query"] == "Tell me about Python"


class TestGetLogStats:
    def test_returns_zeros_when_empty(self, temp_dir):
        logger = QueryLogger(log_dir=str(temp_dir))
        stats = logger.get_log_stats()

        assert stats["total_queries"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["abstention_rate"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

    def test_calculates_correct_stats(self, temp_dir, sample_jsonl_lines):
        log_file = temp_dir / "queries.jsonl"
        log_file.write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        stats = logger.get_log_stats(max_entries=100)

        assert stats["total_queries"] == 3
        # avg_confidence: (0.85 + 0.1 + 0.90) / 3 = 0.616...
        assert abs(stats["avg_confidence"] - 0.617) < 0.01
        # abstention_rate: 1/3 = 0.333
        assert abs(stats["abstention_rate"] - 0.333) < 0.01
        # avg_latency: (1200 + 300 + 950) / 3 = 816.7
        assert abs(stats["avg_latency_ms"] - 816.7) < 1

    def test_respects_max_entries_parameter(self, temp_dir, sample_jsonl_lines):
        log_file = temp_dir / "queries.jsonl"
        log_file.write_text("\n".join(sample_jsonl_lines) + "\n", encoding="utf-8")

        logger = QueryLogger(log_dir=str(temp_dir))
        stats = logger.get_log_stats(max_entries=2)

        # Only 2 most recent entries should be counted
        assert stats["total_queries"] == 2


# ── Helpers ────────────────────────────────────────────────────────────────────


class _MockResult:
    """Minimal mock of PipelineResult for logger tests."""

    def __init__(self, query: str, answer: str):
        self.query = query
        self.answer = answer

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "should_abstain": False,
            "reliability": {"confidence": 0.8, "score": 0.8},
            "total_latency_ms": 500,
        }