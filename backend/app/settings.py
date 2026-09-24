"""Application paths and optional Foundry runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkSettings:
    model_registry_path: Path
    voice_options_path: Path
    manifest_path: Path
    database_path: Path
    foundry_project_endpoint: str | None
    trace_enabled: bool
    trace_sensitive_data: bool
    trace_port: int
    dialects_path: Path = PROJECT_ROOT / "config" / "dialects.json"
    examples_path: Path | None = None
    refiner_deployment: str | None = None
    refiner_reasoning_effort: str | None = "low"

def load_settings() -> BenchmarkSettings:
    """Read non-secret configuration. Credentials remain with Azure identity providers."""
    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(os.getenv("BENCHMARK_DATA_DIR", PROJECT_ROOT / "data" / "swissdial"))
    runtime_dir = Path(os.getenv("BENCHMARK_RUNTIME_DIR", PROJECT_ROOT / ".runtime"))
    return BenchmarkSettings(
        model_registry_path=Path(os.getenv("BENCHMARK_MODELS_PATH", PROJECT_ROOT / "config" / "models.json")),
        voice_options_path=Path(os.getenv("BENCHMARK_VOICE_OPTIONS_PATH", PROJECT_ROOT / "config" / "voice-options.json")),
        manifest_path=Path(os.getenv("BENCHMARK_MANIFEST_PATH", data_dir / "manifest.jsonl")),
        database_path=Path(os.getenv("BENCHMARK_DATABASE_PATH", runtime_dir / "benchmark.sqlite3")),
        foundry_project_endpoint=os.getenv("FOUNDRY_PROJECT_ENDPOINT"),
        trace_enabled=os.getenv("BENCHMARK_TRACE_ENABLED", "false").lower() == "true",
        trace_sensitive_data=os.getenv("BENCHMARK_TRACE_SENSITIVE_DATA", "false").lower() == "true",
        trace_port=int(os.getenv("BENCHMARK_TRACE_PORT", "4317")),
        dialects_path=Path(os.getenv("BENCHMARK_DIALECTS_PATH", PROJECT_ROOT / "config" / "dialects.json")),
        examples_path=Path(os.getenv("BENCHMARK_EXAMPLES_PATH", data_dir / "examples.jsonl")),
        refiner_deployment=os.getenv("BENCHMARK_REFINER_DEPLOYMENT") or None,
        refiner_reasoning_effort=os.getenv("BENCHMARK_REFINER_REASONING_EFFORT", "low") or None,
    )