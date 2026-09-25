"""HTTP API for configuring and executing Swiss German benchmark runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import csv
from dataclasses import asdict
import io
import json
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import Any, Literal
import uuid
import zipfile

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .catalog import load_all_dataset_items, load_dialect_atlas, load_models, load_voice_options
from .datasets import MAX_ARCHIVE_BYTES, extract_dataset_archive
from .refiner import AzureOpenAITextRefiner, TextRefiner
from .repository import RunRepository
from .runner import BenchmarkRequest, BenchmarkRunner, DEFAULT_TRANSCRIPTION_PROMPT
from .settings import BenchmarkSettings, PROJECT_ROOT, load_settings
from .transcriber import AudioTranscriber, RoutedAudioTranscriber


class StartRunPayload(BaseModel):
    model_ids: list[str] = Field(min_length=1)
    item_ids: list[str] = Field(min_length=1)
    parameter_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT
    reference_mode: Literal["dialect", "standard-german"] = "dialect"
    strategy: Literal["baseline", "guided", "two-pass", "ensemble"] = "baseline"


class InstructionPresetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=10_000)


class SafeStaticFiles(StaticFiles):
    """Return 404 for malformed Windows paths instead of surfacing filesystem errors."""

    async def get_response(self, path: str, scope: Any) -> Response:
        try:
            response = await super().get_response(path, scope)
        except OSError:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        # Revalidate the HTML shell so versioned asset URLs take effect immediately after updates.
        if response.media_type == "text/html":
            response.headers["Cache-Control"] = "no-cache"
        return response


def _configure_observability(settings: BenchmarkSettings) -> None:
    if not settings.trace_enabled:
        return
    from agent_framework.observability import configure_otel_providers

    configure_otel_providers(
        vs_code_extension_port=settings.trace_port,
        enable_sensitive_data=settings.trace_sensitive_data,
    )


def _model_payload(model: Any) -> dict[str, Any]:
    payload = asdict(model)
    payload["capabilities"] = list(model.capabilities)
    return payload


def _item_payload(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "audio_file": item.audio_path.name,
        "audio_available": item.audio_path.is_file(),
        "reference_transcript": item.reference_transcript,
        "source": item.source,
        "metadata": item.metadata,
    }


def create_app(
    settings: BenchmarkSettings | None = None,
    transcriber: AudioTranscriber | None = None,
    refiner: TextRefiner | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    _configure_observability(settings)
    repository = RunRepository(settings.database_path)
    if refiner is None and settings.refiner_deployment:
        refiner = AzureOpenAITextRefiner(
            settings.refiner_deployment, settings.refiner_reasoning_effort, api_key=settings.foundry_api_key
        )
    runner = BenchmarkRunner(settings, repository, transcriber or RoutedAudioTranscriber(settings), refiner)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Runs only when a server actually starts (not on import), so a live server's runs are never touched.
        # No worker survives a restart, so interrupted runs are paused and can be resumed from the UI.
        repository.recover_interrupted_runs()
        yield

    app = FastAPI(title="Swiss German Test Bench", version="0.1.0", lifespan=lifespan)
    origins = [origin.strip() for origin in os.getenv("BENCHMARK_CORS_ORIGINS", "http://localhost:5173").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runner = runner
    app.state.repository = repository
    app.state.tasks = set()
    app.state.run_tasks = {}
    app.state.dataset_upload_lock = asyncio.Lock()

    def schedule_run(run_id: str) -> None:
        existing_task = app.state.run_tasks.get(run_id)
        if existing_task is not None and not existing_task.done():
            return

        task = asyncio.create_task(runner.execute_run(run_id))
        app.state.tasks.add(task)
        app.state.run_tasks[run_id] = task

        def clean_up(completed_task: asyncio.Task) -> None:
            app.state.tasks.discard(completed_task)
            if app.state.run_tasks.get(run_id) is completed_task:
                app.state.run_tasks.pop(run_id, None)

        task.add_done_callback(clean_up)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "authentication": "api-key" if settings.foundry_api_key else "rbac-default-azure-credential",
            "foundry_configured": bool(settings.foundry_project_endpoint),
            "refiner_deployment": refiner.deployment if refiner is not None else None,
            "model_count": len(load_models(settings.model_registry_path, settings.custom_model_name)),
            "voice_option_count": len(load_voice_options(settings.voice_options_path)),
            "dataset_item_count": len(load_all_dataset_items(settings.manifest_path, settings.uploaded_datasets_path)),
        }

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [_model_payload(model) for model in load_models(settings.model_registry_path, settings.custom_model_name)]

    @app.get("/api/instructions")
    def instruction_presets() -> list[dict[str, str]]:
        return repository.list_instruction_presets()

    @app.post("/api/instructions")
    def save_instruction_preset(payload: InstructionPresetPayload) -> dict[str, str]:
        try:
            return repository.save_instruction_preset(payload.name, payload.prompt)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    @app.get("/api/voice-options")
    def voice_options() -> list[dict[str, Any]]:
        return load_voice_options(settings.voice_options_path)

    @app.get("/api/dialects")
    def dialects() -> dict[str, Any]:
        return load_dialect_atlas(settings.dialects_path)

    @app.get("/api/dataset/items")
    def dataset_items() -> list[dict[str, Any]]:
        return [
            _item_payload(item)
            for item in load_all_dataset_items(settings.manifest_path, settings.uploaded_datasets_path)
        ]

    @app.get("/api/dataset/items/{item_id}/audio")
    def dataset_item_audio(item_id: str) -> FileResponse:
        items = {
            item.id: item
            for item in load_all_dataset_items(settings.manifest_path, settings.uploaded_datasets_path)
        }
        item = items.get(item_id)
        if item is None or not item.audio_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio clip not found.")

        media_type = mimetypes.guess_type(item.audio_path.name)[0] or "application/octet-stream"
        return FileResponse(item.audio_path, media_type=media_type)

    @app.post("/api/datasets", status_code=status.HTTP_201_CREATED)
    async def upload_dataset(request: Request, name: str = Query(min_length=1, max_length=80)) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/zip":
            raise HTTPException(status_code=415, detail="Upload a ZIP with Content-Type: application/zip.")
        name = name.strip()
        if not name or not name.isprintable():
            raise HTTPException(status_code=422, detail="Dataset name must be printable and non-empty.")
        if settings.uploaded_datasets_path is None:
            raise HTTPException(status_code=503, detail="Configure BENCHMARK_DATASETS_DIR before uploading.")

        root = settings.uploaded_datasets_path
        root.mkdir(parents=True, exist_ok=True)
        async with app.state.dataset_upload_lock:
            existing_items = load_all_dataset_items(settings.manifest_path, root)
            if any((item.metadata.get("dataset") or item.source).casefold() == name.casefold()
                   for item in existing_items):
                raise HTTPException(status_code=409, detail="A data source with this name already exists.")
            with tempfile.TemporaryDirectory(prefix=".upload-", dir=root) as temporary:
                staging = Path(temporary)
                archive_path = staging / "upload.zip"
                size = 0
                with archive_path.open("wb") as output:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_ARCHIVE_BYTES:
                            raise HTTPException(status_code=413, detail="ZIP exceeds the 512 MiB upload limit.")
                        output.write(chunk)
                try:
                    count = extract_dataset_archive(
                        archive_path,
                        staging / "dataset",
                        name,
                        {item.id for item in existing_items},
                    )
                except (ValueError, zipfile.BadZipFile) as error:
                    raise HTTPException(status_code=422, detail=str(error)) from error

                dataset_id = uuid.uuid4().hex
                (staging / "dataset").replace(root / dataset_id)
                return {"id": dataset_id, "name": name, "item_count": count}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return repository.list_runs()

    @app.get("/api/runs/summary")
    def history_summary() -> dict[str, Any]:
        return repository.get_history_summary()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return run

    @app.get("/api/runs/{run_id}/export.csv")
    def export_run_csv(run_id: str) -> Response:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")

        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "run_id",
                "run_status",
                "started_at",
                "completed_at",
                "model_id",
                "model_label",
                "item_id",
                "audio_path",
                "reference_utterance",
                "model_response",
                "word_match_rate",
                "word_error_rate",
                "character_error_rate",
                "chrf",
                "latency_ms",
                "time_to_first_token_ms",
                "result_status",
                "error",
                "prompt",
                "reference_mode",
                "strategy",
                "parameters",
            ],
        )
        writer.writeheader()
        for result in run["results"]:
            writer.writerow(
                {
                    "run_id": run["id"],
                    "run_status": run["status"],
                    "started_at": run["started_at"],
                    "completed_at": run["completed_at"],
                    "model_id": result["model_id"],
                    "model_label": result["model_label"],
                    "item_id": result["item_id"],
                    "audio_path": result["audio_path"],
                    "reference_utterance": result["reference_transcript"],
                    "model_response": result["transcript"],
                    "word_match_rate": result["word_match_rate"],
                    "word_error_rate": result["word_error_rate"],
                    "character_error_rate": result["character_error_rate"],
                    "chrf": result["chrf"],
                    "latency_ms": result["latency_ms"],
                    "time_to_first_token_ms": result["time_to_first_token_ms"],
                    "result_status": result["status"],
                    "error": result["error"],
                    "prompt": run["prompt"],
                    "reference_mode": run["reference_mode"],
                    "strategy": run["strategy"],
                    "parameters": json.dumps(run["parameters"], ensure_ascii=False),
                }
            )
        return Response(
            content=output.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="swiss-german-run-{run_id}.csv"'},
        )

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(payload: StartRunPayload) -> dict[str, str]:
        try:
            run_id = runner.create_run(
                BenchmarkRequest(
                    model_ids=payload.model_ids,
                    item_ids=payload.item_ids,
                    parameter_overrides=payload.parameter_overrides,
                    prompt=payload.prompt,
                    reference_mode=payload.reference_mode,
                    strategy=payload.strategy,
                )
            )
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

        schedule_run(run_id)
        return {"id": run_id, "status": "queued"}

    @app.post("/api/runs/{run_id}/pause")
    async def pause_run(run_id: str) -> dict[str, Any]:
        return await control_run(run_id, "pause")

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str) -> dict[str, Any]:
        return await control_run(run_id, "resume")

    @app.post("/api/runs/{run_id}/stop")
    async def stop_run(run_id: str) -> dict[str, Any]:
        return await control_run(run_id, "stop")

    async def control_run(run_id: str, action: str) -> dict[str, Any]:
        try:
            repository.control_run(run_id, action)
        except KeyError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.") from error
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

        active_task = app.state.run_tasks.get(run_id)
        if action == "resume":
            schedule_run(run_id)
        elif action == "stop" and (active_task is None or active_task.done()):
            repository.set_run_status(run_id, "stopped")

        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return run

    app.mount("/", SafeStaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.api:app", host="127.0.0.1", port=8001, reload=True)