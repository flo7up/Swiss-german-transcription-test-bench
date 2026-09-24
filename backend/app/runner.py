"""Execute and persist a selected model-by-audio benchmark matrix."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable

from .catalog import load_dataset_items, load_dialect_atlas, load_models
from .domain import DatasetItem, ModelDefinition
from .examples import ExamplePool
from .metrics import score_transcript
from .refiner import TextRefiner
from .repository import RunRepository
from .settings import BenchmarkSettings
from .transcriber import AudioTranscriber, TranscriptionResponse


DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe this recording in Swiss German. Return only the spoken words, without translating them."
)
HIGH_GERMAN_TRANSCRIPTION_PROMPT = (
    "Transcribe this recording into High German. Return only the spoken words, without explanation."
)
RUN_CONTROL_POLL_SECONDS = 0.1
REFERENCE_MODES = {"dialect", "standard-german"}
STRATEGIES = {"baseline", "guided", "two-pass", "ensemble"}
FEW_SHOT_EXAMPLE_COUNT = 4
ENSEMBLE_SPELLING_EXAMPLES = 8
FUSION_INSTRUCTIONS = (
    "You are an expert in Swiss German dialects. You reconcile several imperfect speech-recognition hypotheses "
    "of the same recording into the single most likely rendering."
)
SWISS_STANDARD_GERMAN_RULES = (
    "Write Swiss Standard German (Schweizer Hochdeutsch): always use 'ss', never 'ß'. "
    "Translate meaning faithfully and completely; do not summarize, add, or drop content. "
    "Map dialect grammar to idiomatic Standard German: the relative particle 'wo' becomes der/die/das, "
    "narrative perfect tense (het gmacht) usually becomes the preterite (machte), and 'z' + verb becomes 'zu' + verb. "
    "Keep proper names, numbers, and Swiss vocabulary that is also used in Swiss Standard German (e.g. Spital, Velo)."
)


@dataclass(frozen=True)
class BenchmarkRequest:
    model_ids: list[str]
    item_ids: list[str]
    parameter_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompt: str = DEFAULT_TRANSCRIPTION_PROMPT
    reference_mode: str = "dialect"
    strategy: str = "baseline"


@dataclass(frozen=True)
class ItemPrompt:
    prompt: str
    follow_up_prompt: str | None = None


class BenchmarkRunner:
    def __init__(
        self,
        settings: BenchmarkSettings,
        repository: RunRepository,
        transcriber: AudioTranscriber,
        refiner: TextRefiner | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.transcriber = transcriber
        self.refiner = refiner

    async def _single_response(
        self,
        strategy: str,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        reference_mode: str,
        catalog: Iterable[DatasetItem],
        dialect_profiles: dict[str, dict[str, Any]],
    ) -> TranscriptionResponse:
        item_prompt = self._item_prompt(strategy, prompt, item, reference_mode, catalog, dialect_profiles)
        if item_prompt.follow_up_prompt is None:
            return await self.transcriber.transcribe(model, item, parameters, item_prompt.prompt)
        return await self.transcriber.transcribe(
            model, item, parameters, item_prompt.prompt, follow_up_prompt=item_prompt.follow_up_prompt
        )

    async def _ensemble_response(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        reference_mode: str,
        catalog: Iterable[DatasetItem],
        dialect_profiles: dict[str, dict[str, Any]],
        example_pool: ExamplePool | None,
    ) -> TranscriptionResponse:
        """Hear the clip three times independently, then let a text model reconcile the hypotheses."""
        if self.refiner is None:
            raise RuntimeError("The ensemble strategy requires BENCHMARK_REFINER_DEPLOYMENT.")
        catalog = list(catalog)
        target_prompt = prompt if reference_mode == "standard-german" else DEFAULT_TRANSCRIPTION_PROMPT
        passes = {
            "guided-dialect": self._item_prompt("guided", DEFAULT_TRANSCRIPTION_PROMPT, item, "dialect", catalog, dialect_profiles).prompt,
            "guided-standard": self._item_prompt("guided", HIGH_GERMAN_TRANSCRIPTION_PROMPT, item, "standard-german", catalog, dialect_profiles).prompt,
            "baseline": self._prompt_for_item(
                prompt if prompt.strip() else target_prompt, item.metadata, reference_mode
            ),
        }
        outcomes = await asyncio.gather(
            *(self.transcriber.transcribe(model, item, parameters, pass_prompt) for pass_prompt in passes.values()),
            return_exceptions=True,
        )
        hypotheses = {
            name: outcome.transcript
            for name, outcome in zip(passes, outcomes)
            if not isinstance(outcome, BaseException) and outcome.transcript.strip()
        }
        if not hypotheses:
            raise next(outcome for outcome in outcomes if isinstance(outcome, BaseException))

        dialect_code = (item.metadata.get("dialect") or "").upper()
        profile = dialect_profiles.get(dialect_code, {})
        examples: list[tuple[str, str]] = []
        if reference_mode == "dialect":
            query = hypotheses.get("guided-dialect") or hypotheses.get("baseline") or ""
            if example_pool is not None and dialect_code in example_pool:
                examples = example_pool.similar(
                    dialect_code, query, ENSEMBLE_SPELLING_EXAMPLES, item.metadata.get("sentence_id")
                )
            else:
                examples = self._few_shot_examples(item, catalog, ENSEMBLE_SPELLING_EXAMPLES)
        fusion_prompt = self._fusion_prompt(item, profile, reference_mode, hypotheses, examples)
        final = await self.refiner.complete(FUSION_INSTRUCTIONS, fusion_prompt)

        conversation: list[dict[str, Any]] = [
            {"role": "assistant", "pass": name, "contents": [text]} for name, text in hypotheses.items()
        ]
        failed = [name for name, outcome in zip(passes, outcomes) if isinstance(outcome, BaseException)]
        if failed:
            conversation.append({"role": "system", "pass": "failed", "contents": failed})
        conversation.append(
            {"role": "assistant", "pass": "fusion", "model": self.refiner.deployment, "contents": [final]}
        )
        return TranscriptionResponse(transcript=final, conversation=conversation)

    @staticmethod
    def _fusion_prompt(
        item: DatasetItem,
        profile: dict[str, Any],
        reference_mode: str,
        hypotheses: dict[str, str],
        examples: list[tuple[str, str]],
    ) -> str:
        dialect_code = (item.metadata.get("dialect") or "").upper()
        dialect_name = profile.get("name") or item.metadata.get("dialect_name") or "Swiss German"
        label = f"{dialect_name} ({dialect_code})" if dialect_code else dialect_name
        # Keys may carry a "#n" suffix when several samples of the same pass type are fused.
        def of_kind(kind: str) -> list[str]:
            return [text for name, text in hypotheses.items() if name.split("#")[0] == kind]

        dialect_hypotheses = of_kind("guided-dialect")
        standard_hypotheses = of_kind("guided-standard")
        (standard_hypotheses if reference_mode == "standard-german" else dialect_hypotheses).extend(of_kind("baseline"))
        lines = [
            f"A speech recognizer listened to a recording in {label} several times independently. "
            "Each hypothesis may contain different recognition errors:"
        ]
        lines += [f"- Swiss German transcript {chr(65 + index)}: {text}" for index, text in enumerate(dialect_hypotheses)]
        lines += [f"- Standard German translation {chr(65 + index)}: {text}" for index, text in enumerate(standard_hypotheses)]
        if reference_mode == "dialect":
            if examples:
                lines.append("Example sentences written in this dialect (follow their spelling conventions):")
                lines += [f"- {dialect_text}" for dialect_text, _ in examples]
            lines.append(
                "Reconstruct what the speaker most likely said and write it verbatim in this Swiss German dialect, "
                "following the spelling conventions of the examples. Prefer words that several transcripts agree on; "
                "use the translations only to understand the meaning. Do not translate into Standard German. "
                "Return only the transcript."
            )
        else:
            lines.append(
                "Reconstruct what the speaker most likely said and write it in Swiss Standard German. Prefer words "
                "and phrases that several hypotheses agree on; when they disagree, choose the reading that is most "
                "plausible for this dialect and the context. Stay close to the hypotheses' wording."
            )
            lines.append(SWISS_STANDARD_GERMAN_RULES)
            lines.append("Return only the translation.")
        return "\n".join(lines)

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
        if request.strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy: {request.strategy}")
        if request.strategy == "two-pass" and request.reference_mode != "standard-german":
            raise ValueError("The two-pass strategy requires the High German evaluation reference.")
        if request.strategy == "ensemble" and self.refiner is None:
            raise ValueError("The ensemble strategy requires BENCHMARK_REFINER_DEPLOYMENT to name a text deployment.")
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
            strategy=request.strategy,
        )

    async def execute_run(self, run_id: str) -> None:
        run = self.repository.get_run(run_id)
        if run is None:
            raise ValueError(f"Run does not exist: {run_id}")
        if run["status"] in {"completed", "failed", "stopped"}:
            return

        models = {model.id: model for model in load_models(self.settings.model_registry_path)}
        items = {item.id: item for item in load_dataset_items(self.settings.manifest_path)}
        strategy = run.get("strategy") or "baseline"
        dialect_profiles = (
            {profile["code"]: profile for profile in load_dialect_atlas(self.settings.dialects_path)["dialects"]}
            if strategy != "baseline"
            else {}
        )
        example_pool = (
            ExamplePool.load(self.settings.examples_path)
            if strategy == "ensemble" and self.settings.examples_path is not None
            else None
        )
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
                        if strategy == "ensemble":
                            response = await self._ensemble_response(
                                model,
                                item,
                                parameters,
                                run["prompt"],
                                run["reference_mode"],
                                items.values(),
                                dialect_profiles,
                                example_pool,
                            )
                        else:
                            response = await self._single_response(
                                strategy,
                                model,
                                item,
                                parameters,
                                run["prompt"],
                                run["reference_mode"],
                                items.values(),
                                dialect_profiles,
                            )
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
                            time_to_first_token_ms=response.time_to_first_token_ms,
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
                            time_to_first_token_ms=None,
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

    @classmethod
    def _item_prompt(
        cls,
        strategy: str,
        prompt: str,
        item: DatasetItem,
        reference_mode: str,
        catalog: Iterable[DatasetItem],
        dialect_profiles: dict[str, dict[str, Any]],
    ) -> ItemPrompt:
        if strategy == "baseline":
            return ItemPrompt(cls._prompt_for_item(prompt, item.metadata, reference_mode))

        dialect_code = (item.metadata.get("dialect") or "").upper()
        profile = dialect_profiles.get(dialect_code, {})
        examples = cls._few_shot_examples(item, catalog)
        context = cls._dialect_context(item, profile)
        custom_prompt = prompt.strip()
        is_default_prompt = custom_prompt in {DEFAULT_TRANSCRIPTION_PROMPT, HIGH_GERMAN_TRANSCRIPTION_PROMPT, ""}

        if reference_mode == "dialect":
            lines = [custom_prompt or DEFAULT_TRANSCRIPTION_PROMPT, context]
            if examples:
                lines.append("Examples of how this dialect is written (follow these spelling conventions):")
                lines.extend(f"- {dialect_text}" for dialect_text, _ in examples)
            lines.append(
                "Transcribe the recording verbatim in this dialect with the same spelling conventions. "
                "Do not translate into Standard German. Return only the transcript."
            )
            return ItemPrompt("\n".join(line for line in lines if line))

        pair_lines = [
            f"Dialect: {dialect_text}\nStandard German: {standard_text}" for dialect_text, standard_text in examples
        ]
        if strategy == "guided":
            lines = [custom_prompt or HIGH_GERMAN_TRANSCRIPTION_PROMPT, context]
            if pair_lines:
                lines.append("Example translations from this dialect:")
                lines.extend(pair_lines)
            lines.append(SWISS_STANDARD_GERMAN_RULES)
            lines.append("Return only the Swiss Standard German translation of the recording, with no heading or explanation.")
            return ItemPrompt("\n".join(line for line in lines if line))

        first_pass = "\n".join(
            line
            for line in [
                DEFAULT_TRANSCRIPTION_PROMPT,
                context,
                "Examples of how this dialect is written:" if examples else "",
                *(f"- {dialect_text}" for dialect_text, _ in examples),
                "Transcribe the recording verbatim in this dialect. Return only the transcript.",
            ]
            if line
        )
        follow_up = "\n".join(
            line
            for line in [
                "Now translate your dialect transcript into Swiss Standard German. "
                "Listen to the recording again to resolve unclear words.",
                "Example translations from this dialect:" if pair_lines else "",
                *pair_lines,
                SWISS_STANDARD_GERMAN_RULES,
                "" if is_default_prompt else f"Additional instructions: {custom_prompt}",
                "Return only the Swiss Standard German translation, with no heading or explanation.",
            ]
            if line
        )
        return ItemPrompt(first_pass, follow_up)

    @staticmethod
    def _dialect_context(item: DatasetItem, profile: dict[str, Any]) -> str:
        dialect_name = profile.get("name") or item.metadata.get("dialect_name")
        dialect_code = (item.metadata.get("dialect") or "").upper()
        if not dialect_name and not dialect_code:
            return ""
        label = f"{dialect_name} ({dialect_code})" if dialect_name and dialect_code else dialect_name or dialect_code
        native_name = profile.get("native_name")
        lines = [f"The recording is spoken in {label}{f', locally called {native_name}' if native_name else ''}."]
        features = profile.get("features") or []
        if features:
            lines.append("Characteristic features of this dialect:")
            lines.extend(f"- {feature}" for feature in features)
        return "\n".join(lines)

    @staticmethod
    def _few_shot_examples(
        item: DatasetItem, catalog: Iterable[DatasetItem], count: int = FEW_SHOT_EXAMPLE_COUNT
    ) -> list[tuple[str, str]]:
        """Pick deterministic same-dialect text pairs, never reusing the evaluated sentence."""
        dialect = (item.metadata.get("dialect") or "").upper()
        sentence_id = item.metadata.get("sentence_id")
        if not dialect:
            return []
        candidates = [
            candidate
            for candidate in catalog
            if candidate.id != item.id
            and (candidate.metadata.get("dialect") or "").upper() == dialect
            and (sentence_id is None or candidate.metadata.get("sentence_id") != sentence_id)
            and candidate.reference_transcript
            and candidate.metadata.get("standard_german_transcript")
        ]
        candidates.sort(key=lambda candidate: hashlib.sha1(f"{item.id}:{candidate.id}".encode()).hexdigest())
        return [
            (candidate.reference_transcript or "", candidate.metadata["standard_german_transcript"])
            for candidate in candidates[:count]
        ]

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