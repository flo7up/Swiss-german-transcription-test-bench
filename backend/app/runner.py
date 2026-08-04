"""Execute and persist a selected model-by-audio benchmark matrix."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .catalog import load_dataset_items, load_models
from .domain import DatasetItem, ModelDefinition
from .metrics import score_transcript
from .repository import RunRepository
from .settings import BenchmarkSettings
from .transcriber import AudioTranscriber


DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe this recording in Swiss German. Return only the spoken words, without translating them."
)
HIGH_GERMAN_TRANSCRIPTION_PROMPT = (
    "Transcribe this recording into High German. Return only the spoken words, without explanation."
)
RUN_CONTROL_POLL_SECONDS = 0.1
REFERENCE_MODES = {"dialect", "standard-german"}


@dataclass(frozen=True)
class BenchmarkRequest:
    model_ids: list[str]
    item_ids: list[str]
    parameter_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT
    reference_mode: str = "dialect"


class BenchmarkRunner:
    def __init__(
        self,
        settings: BenchmarkSettings,
        repository: RunRepository,
        transcriber: AudioTranscriber,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.transcriber = transcriber

    def create_run(self, request: BenchmarkRequest) -> str:
        models = {model.id: model for model in load_models(self.settings.model_registry_path)}
        items = {item.id: item for item in load_dataset_items(self.settings.manifest_path)}
        missing_models = sorted(set(request.model_ids) - models.keys())
        missing_items = sorted(set(request.item_ids) - items.keys())
        if missing_models:
            raise ValueError(f"Unknown model IDs: {', '.join(missing_models)}")
        if missing_items:
            raise ValueError(f"Unknown dataset item IDs: {', '.join(missing_items)}")
        if not request.model_ids or not request.item_ids:
            raise ValueError("Select at least one model and one audio item.")
        if request.reference_mode not in REFERENCE_MODES:
            raise ValueError(f"Unsupported reference mode: {request.reference_mode}")
        if request.reference_mode == "standard-german":
            missing_references = sorted(
                item_id
                for item_id in request.item_ids
                if not items[item_id].metadata.get("standard_german_transcript")
            )
            if missing_references:
                raise ValueError(
                    f"High German references are unavailable for: {', '.join(missing_references)}"
                )

        resolved_parameters = {
            model_id: self._resolve_parameters(models[model_id], request.parameter_overrides.get(model_id, {}))
            for model_id in request.model_ids
        }
        prompt = request.prompt.strip()
        if request.reference_mode == "standard-german" and (not prompt or prompt == DEFAULT_TRANSCRIPTION_PROMPT):
            prompt = HIGH_GERMAN_TRANSCRIPTION_PROMPT
        return self.repository.create_run(
            model_ids=request.model_ids,
            item_ids=request.item_ids,
            parameters=resolved_parameters,
            prompt=prompt or DEFAULT_TRANSCRIPTION_PROMPT,
            reference_mode=request.reference_mode,
        )

    async def execute_run(self, run_id: str) -> None:
        run = self.repository.get_run(run_id)
        if run is None:
            raise ValueError(f"Run does not exist: {run_id}")
        if run["status"] in {"completed", "failed", "stopped"}:
            return

        models = {model.id: model for model in load_models(self.settings.model_registry_path)}
        items = {item.id: item for item in load_dataset_items(self.settings.manifest_path)}
        if run["status"] == "queued":
            self.repository.set_run_status(run_id, "running")
        try:
            for model_id in run["model_ids"]:
                model = models[model_id]
                parameters = run["parameters"][model_id]
                for item_id in run["item_ids"]:
                    if not await self._await_run_permission(run_id):
                        return
                    item = items[item_id]
                    reference_transcript = self._reference_for_item(item, run["reference_mode"])
                    started = time.perf_counter()
                    try:
                        prompt = self._prompt_for_item(run["prompt"], item.metadata, run["reference_mode"])
                        response = await self.transcriber.transcribe(model, item, parameters, prompt)
                        scores = score_transcript(reference_transcript, response.transcript)
                        self.repository.add_result(
                            run_id=run_id,
                            status="completed",
                            model_id=model.id,
                            model_label=model.label,
                            item_id=item.id,
                            audio_path=str(item.audio_path),
                            transcript=response.transcript,
                            reference_transcript=reference_transcript,
                            word_error_rate=scores.word_error_rate,
                            character_error_rate=scores.character_error_rate,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            conversation=response.conversation,
                            error=None,
                        )
                    except Exception as error:  # Individual failures belong in the result set.
                        self.repository.add_result(
                            run_id=run_id,
                            status="failed",
                            model_id=model.id,
                            model_label=model.label,
                            item_id=item.id,
                            audio_path=str(item.audio_path),
                            transcript=None,
                            reference_transcript=reference_transcript,
                            word_error_rate=None,
                            character_error_rate=None,
                            latency_ms=(time.perf_counter() - started) * 1000,
                            conversation=None,
                            error=str(error),
                        )
            if await self._await_run_permission(run_id):
                self.repository.set_run_status(run_id, "completed")
        except Exception as error:
            self.repository.set_run_status(run_id, "failed", str(error))
            raise

    async def _await_run_permission(self, run_id: str) -> bool:
        while True:
            status = self.repository.get_run_status(run_id)
            if status is None:
                raise ValueError(f"Run does not exist: {run_id}")
            if status in {"stopping", "stopped"}:
                self.repository.set_run_status(run_id, "stopped")
                return False
            if status == "paused":
                await asyncio.sleep(RUN_CONTROL_POLL_SECONDS)
                continue
            if status == "queued":
                self.repository.set_run_status(run_id, "running")
            if status in {"queued", "running"}:
                return True
            if status in {"completed", "failed"}:
                return False
            raise ValueError(f"Unexpected run status: {status}")

    @staticmethod
    def _prompt_for_item(prompt: str, metadata: dict[str, str], reference_mode: str = "dialect") -> str:
        dialect_name = metadata.get("dialect_name")
        dialect_code = metadata.get("dialect")
        dialect = dialect_name or dialect_code
        if dialect_name and dialect_code:
            dialect = f"{dialect_name} ({dialect_code.upper()})"
        if reference_mode == "standard-german":
            recording_context = f"The recording uses {dialect}. " if dialect else ""
            return (
                f"{prompt.rstrip()}\n{recording_context}Return only a High German (Standard German) transcript, "
                "translating dialectal wording when needed."
            )
        if not dialect:
            return prompt
        return f"{prompt.rstrip()}\nThe recording uses {dialect}. Preserve this dialect in the transcription."

    @staticmethod
    def _reference_for_item(item: DatasetItem, reference_mode: str) -> str | None:
        if reference_mode == "standard-german":
            return item.metadata.get("standard_german_transcript")
        return item.reference_transcript

    @staticmethod
    def _resolve_parameters(model: ModelDefinition, overrides: dict[str, Any]) -> dict[str, Any]:
        parameters = {parameter.name: parameter.default for parameter in model.parameters}
        unknown = sorted(set(overrides) - parameters.keys())
        if unknown:
            raise ValueError(f"Unsupported parameters for {model.label}: {', '.join(unknown)}")
        for parameter in model.parameters:
            value = overrides.get(parameter.name, parameter.default)
            if parameter.kind == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"{parameter.label} must be numeric.")
            if parameter.minimum is not None and value < parameter.minimum:
                raise ValueError(f"{parameter.label} must be at least {parameter.minimum}.")
            if parameter.maximum is not None and value > parameter.maximum:
                raise ValueError(f"{parameter.label} must be at most {parameter.maximum}.")
            parameters[parameter.name] = value
        return parameters