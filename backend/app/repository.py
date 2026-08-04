"""SQLite-backed persistence for reproducible benchmark history."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from .metrics import word_match_rate


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


class RunRepository:
    _FINAL_RUN_STATUSES = {"completed", "failed", "stopped"}

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connection()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    model_ids_json TEXT NOT NULL,
                    item_ids_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    transcript TEXT,
                    reference_transcript TEXT,
                    word_error_rate REAL,
                    character_error_rate REAL,
                    latency_ms REAL,
                    conversation_json TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS results_run_id_idx ON results(run_id);
                CREATE TABLE IF NOT EXISTS instruction_presets (
                    name TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.commit()

    def list_instruction_presets(self) -> list[dict[str, str]]:
        with closing(self._connection()) as connection:
            rows = connection.execute(
                "SELECT name, prompt, created_at, updated_at FROM instruction_presets ORDER BY updated_at DESC, name ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def save_instruction_preset(self, name: str, prompt: str) -> dict[str, str]:
        clean_name = name.strip()
        clean_prompt = prompt.strip()
        if not clean_name:
            raise ValueError("Instruction name is required.")
        if not clean_prompt:
            raise ValueError("Instruction text is required.")
        timestamp = _timestamp()
        with closing(self._connection()) as connection:
            connection.execute(
                """
                INSERT INTO instruction_presets (name, prompt, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET prompt = excluded.prompt, updated_at = excluded.updated_at
                """,
                (clean_name, clean_prompt, timestamp, timestamp),
            )
            row = connection.execute(
                "SELECT name, prompt, created_at, updated_at FROM instruction_presets WHERE name = ?", (clean_name,)
            ).fetchone()
            connection.commit()
        return dict(row)

    def create_run(
        self,
        *,
        model_ids: list[str],
        item_ids: list[str],
        parameters: dict[str, dict[str, Any]],
        prompt: str,
    ) -> str:
        run_id = str(uuid.uuid4())
        with closing(self._connection()) as connection:
            connection.execute(
                """
                INSERT INTO runs (id, status, started_at, model_ids_json, item_ids_json, parameters_json, prompt)
                VALUES (?, 'queued', ?, ?, ?, ?, ?)
                """,
                (run_id, _timestamp(), json.dumps(model_ids), json.dumps(item_ids), json.dumps(parameters), prompt),
            )
            connection.commit()
        return run_id

    def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        completed_at = _timestamp() if status in self._FINAL_RUN_STATUSES else None
        with closing(self._connection()) as connection:
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = ?, error = ? WHERE id = ?",
                (status, completed_at, error, run_id),
            )
            connection.commit()

    def get_run_status(self, run_id: str) -> str | None:
        with closing(self._connection()) as connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row["status"] if row is not None else None

    def control_run(self, run_id: str, action: str) -> str:
        transitions = {
            "pause": {"queued", "running"},
            "resume": {"paused"},
            "stop": {"queued", "running", "paused", "stopping"},
        }
        target_status = {"pause": "paused", "resume": "running", "stop": "stopping"}
        if action not in transitions:
            raise ValueError(f"Unsupported run control: {action}")

        with closing(self._connection()) as connection:
            row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            current_status = row["status"]
            if action == "stop" and current_status == "stopping":
                return current_status
            if current_status not in transitions[action]:
                raise ValueError(f"Cannot {action} a run with status {current_status}.")
            connection.execute(
                "UPDATE runs SET status = ?, completed_at = NULL, error = NULL WHERE id = ?",
                (target_status[action], run_id),
            )
            connection.commit()
        return target_status[action]

    def add_result(
        self,
        *,
        run_id: str,
        status: str,
        model_id: str,
        model_label: str,
        item_id: str,
        audio_path: str,
        transcript: str | None,
        reference_transcript: str | None,
        word_error_rate: float | None,
        character_error_rate: float | None,
        latency_ms: float | None,
        conversation: list[dict[str, Any]] | None,
        error: str | None,
    ) -> None:
        with closing(self._connection()) as connection:
            connection.execute(
                """
                INSERT INTO results (
                    run_id, created_at, status, model_id, model_label, item_id, audio_path, transcript,
                    reference_transcript, word_error_rate, character_error_rate, latency_ms, conversation_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _timestamp(),
                    status,
                    model_id,
                    model_label,
                    item_id,
                    audio_path,
                    transcript,
                    reference_transcript,
                    word_error_rate,
                    character_error_rate,
                    latency_ms,
                    json.dumps(conversation) if conversation is not None else None,
                    error,
                ),
            )
            connection.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        with closing(self._connection()) as connection:
            rows = connection.execute(
                """
                SELECT runs.*, COUNT(results.id) AS result_count,
                      COALESCE(SUM(CASE WHEN results.status = 'completed' THEN 1 ELSE 0 END), 0) AS successful_result_count,
                      COALESCE(SUM(CASE WHEN results.status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_result_count,
                       AVG(results.word_error_rate) AS average_word_error_rate,
                      AVG(results.latency_ms) AS average_latency_ms,
                       AVG(
                           CASE
                               WHEN results.word_error_rate IS NULL THEN NULL
                               WHEN results.word_error_rate > 1 THEN 0
                               ELSE 1 - results.word_error_rate
                           END
                       ) AS average_word_match_rate
                FROM runs
                LEFT JOIN results ON results.run_id = runs.id
                GROUP BY runs.id
                ORDER BY runs.started_at DESC
                """
            ).fetchall()
        return [self._run_summary(row) for row in rows]

    def get_history_summary(self) -> dict[str, Any]:
        with closing(self._connection()) as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM runs) AS total_run_count,
                    (SELECT COUNT(*) FROM runs WHERE status = 'completed') AS completed_run_count,
                    (SELECT COUNT(*) FROM runs WHERE status IN ('queued', 'running', 'paused', 'stopping')) AS active_run_count,
                    (SELECT COUNT(*) FROM runs WHERE status = 'stopped') AS stopped_run_count,
                    (SELECT COUNT(*) FROM results) AS result_count,
                    (SELECT COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) FROM results) AS successful_result_count,
                    (SELECT COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) FROM results) AS failed_result_count,
                    (SELECT AVG(word_error_rate) FROM results) AS average_word_error_rate,
                    (SELECT AVG(CASE WHEN word_error_rate IS NULL THEN NULL WHEN word_error_rate > 1 THEN 0 ELSE 1 - word_error_rate END) FROM results) AS average_word_match_rate,
                    (SELECT AVG(latency_ms) FROM results) AS average_latency_ms
                """
            ).fetchone()
        summary = dict(row)
        summary["indicator"] = self._history_indicator(
            summary["total_run_count"],
            summary["average_word_match_rate"],
            summary["failed_result_count"],
            summary["result_count"],
        )
        return summary

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with closing(self._connection()) as connection:
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            results = connection.execute(
                "SELECT * FROM results WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        payload = self._run_summary(run)
        result_payloads = [self._result_payload(result) for result in results]
        scored_results = [result for result in result_payloads if result["word_error_rate"] is not None]
        timed_results = [result for result in result_payloads if result["latency_ms"] is not None]
        payload["results"] = result_payloads
        payload["result_count"] = len(result_payloads)
        payload["successful_result_count"] = sum(result["status"] == "completed" for result in result_payloads)
        payload["failed_result_count"] = sum(result["status"] == "failed" for result in result_payloads)
        payload["average_word_error_rate"] = (
            sum(result["word_error_rate"] for result in scored_results) / len(scored_results)
            if scored_results
            else None
        )
        payload["average_word_match_rate"] = (
            sum(result["word_match_rate"] for result in scored_results) / len(scored_results)
            if scored_results
            else None
        )
        payload["average_latency_ms"] = (
            sum(result["latency_ms"] for result in timed_results) / len(timed_results)
            if timed_results
            else None
        )
        payload["indicator"] = self._run_indicator(
            payload["status"],
            payload["average_word_match_rate"],
            payload["failed_result_count"],
            payload["result_count"],
            payload["total_task_count"],
        )
        return payload

    @staticmethod
    def _run_summary(row: sqlite3.Row) -> dict[str, Any]:
        model_ids = json.loads(row["model_ids_json"])
        item_ids = json.loads(row["item_ids_json"])
        result_count = row["result_count"] if "result_count" in row.keys() else None
        failed_result_count = row["failed_result_count"] if "failed_result_count" in row.keys() else 0
        average_word_match_rate = (
            row["average_word_match_rate"] if "average_word_match_rate" in row.keys() else None
        )
        total_task_count = len(model_ids) * len(item_ids)
        payload = {
            "id": row["id"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "model_ids": model_ids,
            "item_ids": item_ids,
            "parameters": json.loads(row["parameters_json"]),
            "prompt": row["prompt"],
            "error": row["error"],
            "result_count": result_count,
            "total_task_count": total_task_count,
            "successful_result_count": (
                row["successful_result_count"] if "successful_result_count" in row.keys() else None
            ),
            "failed_result_count": failed_result_count,
            "average_word_error_rate": (
                row["average_word_error_rate"] if "average_word_error_rate" in row.keys() else None
            ),
            "average_word_match_rate": average_word_match_rate,
            "average_latency_ms": (
                row["average_latency_ms"] if "average_latency_ms" in row.keys() else None
            ),
        }
        payload["indicator"] = RunRepository._run_indicator(
            payload["status"],
            average_word_match_rate,
            failed_result_count or 0,
            result_count or 0,
            total_task_count,
        )
        return payload

    @staticmethod
    def _run_indicator(
        status: str,
        average_word_match_rate: float | None,
        failed_result_count: int,
        result_count: int,
        total_task_count: int,
    ) -> dict[str, str]:
        if status in {"queued", "running", "paused", "stopping"}:
            return {"label": status.title(), "tone": "active"}
        if status == "stopped":
            return {"label": "Stopped", "tone": "stopped"}
        if status == "failed":
            return {"label": "Run failed", "tone": "negative"}
        if failed_result_count:
            return {"label": "Partial errors", "tone": "warning"}
        if result_count == 0 and total_task_count:
            return {"label": "No results", "tone": "neutral"}
        if average_word_match_rate is None:
            return {"label": "No reference", "tone": "neutral"}
        if average_word_match_rate >= 0.85:
            return {"label": "Strong match", "tone": "positive"}
        if average_word_match_rate >= 0.6:
            return {"label": "Review", "tone": "warning"}
        return {"label": "Low match", "tone": "negative"}

    @staticmethod
    def _history_indicator(
        total_run_count: int,
        average_word_match_rate: float | None,
        failed_result_count: int,
        result_count: int,
    ) -> dict[str, str]:
        if total_run_count == 0:
            return {"label": "No runs yet", "tone": "neutral"}
        if failed_result_count:
            return {"label": "Errors to review", "tone": "warning"}
        if result_count == 0 or average_word_match_rate is None:
            return {"label": "No scored results", "tone": "neutral"}
        if average_word_match_rate >= 0.85:
            return {"label": "Strong overall", "tone": "positive"}
        if average_word_match_rate >= 0.6:
            return {"label": "Review overall", "tone": "warning"}
        return {"label": "Low overall match", "tone": "negative"}

    @staticmethod
    def _result_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "item_id": row["item_id"],
            "audio_path": row["audio_path"],
            "transcript": row["transcript"],
            "reference_transcript": row["reference_transcript"],
            "word_error_rate": row["word_error_rate"],
            "word_match_rate": (
                word_match_rate(row["word_error_rate"])
            ),
            "character_error_rate": row["character_error_rate"],
            "latency_ms": row["latency_ms"],
            "conversation": json.loads(row["conversation_json"]) if row["conversation_json"] else None,
            "error": row["error"],
        }