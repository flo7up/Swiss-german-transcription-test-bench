import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.repository import RunRepository
from backend.app.runner import BenchmarkRequest, BenchmarkRunner
from backend.app.settings import BenchmarkSettings
from backend.app.transcriber import RealtimeAudioTranscriber, TranscriptionResponse


class FakeTranscriber:
    async def transcribe(self, model, item, parameters, prompt):
        return TranscriptionResponse(
            transcript="grüezi mitenand",
            conversation=[{"role": "assistant", "contents": ["grüezi mitenand"]}],
            time_to_first_token_ms=125,
        )


class CapturingTranscriber(FakeTranscriber):
    def __init__(self) -> None:
        self.prompts = []

    async def transcribe(self, model, item, parameters, prompt):
        self.prompts.append(prompt)
        return await super().transcribe(model, item, parameters, prompt)


class HighGermanTranscriber(CapturingTranscriber):
    async def transcribe(self, model, item, parameters, prompt):
        self.prompts.append(prompt)
        return TranscriptionResponse(
            transcript="guten tag miteinander",
            conversation=[{"role": "assistant", "contents": ["guten tag miteinander"]}],
        )


class GatedTranscriber:
    def __init__(self) -> None:
        self.first_call_started = asyncio.Event()
        self.release_first_call = asyncio.Event()
        self.call_count = 0

    async def transcribe(self, model, item, parameters, prompt):
        self.call_count += 1
        if self.call_count == 1:
            self.first_call_started.set()
            await self.release_first_call.wait()
        return TranscriptionResponse(transcript="grüezi mitenand", conversation=[])


class BenchmarkRunnerTests(unittest.TestCase):
    def test_realtime_stream_records_first_non_empty_text_delta(self) -> None:
        class EventSocket:
            def __init__(self) -> None:
                self.events = [
                    {"type": "response.output_text.delta", "delta": ""},
                    {"type": "response.output_text.delta", "delta": "grüezi"},
                    {"type": "response.output_text.delta", "delta": " mitenand"},
                    {"type": "response.done", "response": {}},
                ]

            async def recv(self) -> str:
                return json.dumps(self.events.pop(0))

        with patch("backend.app.transcriber.time.perf_counter", return_value=10.25):
            transcript, time_to_first_token_ms = asyncio.run(
                RealtimeAudioTranscriber._collect_text_response(EventSocket(), 10.0)
            )

        self.assertEqual(transcript, "grüezi mitenand")
        self.assertEqual(time_to_first_token_ms, 250)

    def test_scores_against_selected_high_german_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "clip.wav").write_bytes(b"audio")
            (root / "models.json").write_text(
                '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]',
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "be-0000",
                        "audio_path": "clip.wav",
                        "reference_transcript": "grüessech mitenand",
                        "standard_german_transcript": "guten tag miteinander",
                        "source": "ETH SwissDial 1.1",
                        "dialect": "BE",
                        "dialect_name": "Bernese German",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "voice-options.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "benchmark.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
            )
            transcriber = HighGermanTranscriber()
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, transcriber)
            run_id = runner.create_run(
                BenchmarkRequest(
                    model_ids=["model-a"],
                    item_ids=["be-0000"],
                    prompt="Transcribe exactly.",
                    reference_mode="standard-german",
                )
            )

            asyncio.run(runner.execute_run(run_id))
            completed_run = repository.get_run(run_id)

        self.assertEqual(completed_run["reference_mode"], "standard-german")
        self.assertEqual(completed_run["results"][0]["reference_transcript"], "guten tag miteinander")
        self.assertEqual(completed_run["results"][0]["word_error_rate"], 0)
        self.assertIn("High German", transcriber.prompts[0])

    def test_adds_item_dialect_to_transcription_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "clip.wav").write_bytes(b"audio")
            (root / "models.json").write_text(
                '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]',
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "be-0000",
                        "audio_path": "clip.wav",
                        "reference_transcript": "grüessech mitenand",
                        "source": "ETH SwissDial 1.1",
                        "dialect": "BE",
                        "dialect_name": "Bernese German",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "voice-options.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "benchmark.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
            )
            transcriber = CapturingTranscriber()
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), transcriber)
            run_id = runner.create_run(
                BenchmarkRequest(model_ids=["model-a"], item_ids=["be-0000"], prompt="Transcribe exactly.")
            )

            asyncio.run(runner.execute_run(run_id))

        self.assertEqual(
            transcriber.prompts,
            ["Transcribe exactly.\nThe recording uses Bernese German (BE). Preserve this dialect in the transcription."],
        )

    def test_executes_matrix_and_persists_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "clips").mkdir()
            (root / "clips" / "clip-a.mp3").write_bytes(b"audio")
            (root / "models.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "model-a",
                            "label": "Model A",
                            "deployment": "deployment-a",
                            "description": "Test model",
                            "capabilities": ["audio"],
                            "parameters": [
                                {
                                    "name": "temperature",
                                    "label": "Temperature",
                                    "kind": "number",
                                    "default": 0,
                                    "minimum": 0,
                                    "maximum": 1,
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "clip-a",
                        "audio_path": "clips/clip-a.mp3",
                        "reference_transcript": "grüezi mitenand",
                        "source": "SwissDial",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "voice-options.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "benchmark.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
            )
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, FakeTranscriber())
            run_id = runner.create_run(BenchmarkRequest(model_ids=["model-a"], item_ids=["clip-a"]))

            asyncio.run(runner.execute_run(run_id))
            completed_run = repository.get_run(run_id)

        self.assertIsNotNone(completed_run)
        self.assertEqual(completed_run["status"], "completed")
        self.assertEqual(completed_run["results"][0]["word_error_rate"], 0)
        self.assertEqual(completed_run["results"][0]["word_match_rate"], 1)
        self.assertEqual(completed_run["average_word_match_rate"], 1)
        self.assertEqual(completed_run["results"][0]["time_to_first_token_ms"], 125)
        self.assertEqual(completed_run["average_time_to_first_token_ms"], 125)
        self.assertEqual(completed_run["total_task_count"], 1)
        self.assertEqual(completed_run["successful_result_count"], 1)
        self.assertEqual(completed_run["failed_result_count"], 0)
        self.assertEqual(completed_run["indicator"], {"label": "Strong match", "tone": "positive"})
        self.assertEqual(completed_run["results"][0]["status"], "completed")

    def test_rejects_unknown_parameters_before_creating_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "models.json").write_text(
                '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]', encoding="utf-8"
            )
            (root / "manifest.jsonl").write_text(
                '{"id":"clip-a","audio_path":"clip-a.mp3","source":"SwissDial"}\n', encoding="utf-8"
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "voice-options.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "benchmark.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
            )
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), FakeTranscriber())

            with self.assertRaisesRegex(ValueError, "Unsupported parameters"):
                runner.create_run(
                    BenchmarkRequest(
                        model_ids=["model-a"],
                        item_ids=["clip-a"],
                        parameter_overrides={"model-a": {"not-supported": 1}},
                    )
                )

    def test_paused_run_waits_after_current_task_and_resumes(self) -> None:
        async def scenario() -> tuple[dict, int]:
            with tempfile.TemporaryDirectory() as temp_directory:
                root = Path(temp_directory)
                (root / "clips").mkdir()
                for item_id in ("clip-a", "clip-b"):
                    (root / "clips" / f"{item_id}.mp3").write_bytes(b"audio")
                (root / "models.json").write_text(
                    '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]',
                    encoding="utf-8",
                )
                (root / "manifest.jsonl").write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "id": item_id,
                                "audio_path": f"clips/{item_id}.mp3",
                                "reference_transcript": "grüezi mitenand",
                                "source": "SwissDial",
                            }
                        )
                        for item_id in ("clip-a", "clip-b")
                    )
                    + "\n",
                    encoding="utf-8",
                )
                settings = BenchmarkSettings(
                    model_registry_path=root / "models.json",
                    voice_options_path=root / "voice-options.json",
                    manifest_path=root / "manifest.jsonl",
                    database_path=root / "benchmark.sqlite3",
                    foundry_project_endpoint=None,
                    trace_enabled=False,
                    trace_sensitive_data=False,
                    trace_port=4317,
                )
                repository = RunRepository(settings.database_path)
                transcriber = GatedTranscriber()
                runner = BenchmarkRunner(settings, repository, transcriber)
                run_id = runner.create_run(BenchmarkRequest(model_ids=["model-a"], item_ids=["clip-a", "clip-b"]))
                execution = asyncio.create_task(runner.execute_run(run_id))

                await asyncio.wait_for(transcriber.first_call_started.wait(), timeout=1)
                self.assertEqual(repository.control_run(run_id, "pause"), "paused")
                transcriber.release_first_call.set()
                for _ in range(25):
                    paused_run = repository.get_run(run_id)
                    if paused_run and paused_run["status"] == "paused" and paused_run["result_count"] == 1:
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(paused_run["status"], "paused")
                self.assertEqual(paused_run["result_count"], 1)
                self.assertEqual(transcriber.call_count, 1)

                self.assertEqual(repository.control_run(run_id, "resume"), "running")
                await asyncio.wait_for(execution, timeout=1)
                return repository.get_run(run_id), transcriber.call_count

        completed_run, call_count = asyncio.run(scenario())
        self.assertEqual(completed_run["status"], "completed")
        self.assertEqual(completed_run["result_count"], 2)
        self.assertEqual(call_count, 2)

    def test_stop_ends_run_after_current_task(self) -> None:
        async def scenario() -> tuple[dict, int]:
            with tempfile.TemporaryDirectory() as temp_directory:
                root = Path(temp_directory)
                (root / "clips").mkdir()
                for item_id in ("clip-a", "clip-b"):
                    (root / "clips" / f"{item_id}.mp3").write_bytes(b"audio")
                (root / "models.json").write_text(
                    '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]',
                    encoding="utf-8",
                )
                (root / "manifest.jsonl").write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "id": item_id,
                                "audio_path": f"clips/{item_id}.mp3",
                                "reference_transcript": "grüezi mitenand",
                                "source": "SwissDial",
                            }
                        )
                        for item_id in ("clip-a", "clip-b")
                    )
                    + "\n",
                    encoding="utf-8",
                )
                settings = BenchmarkSettings(
                    model_registry_path=root / "models.json",
                    voice_options_path=root / "voice-options.json",
                    manifest_path=root / "manifest.jsonl",
                    database_path=root / "benchmark.sqlite3",
                    foundry_project_endpoint=None,
                    trace_enabled=False,
                    trace_sensitive_data=False,
                    trace_port=4317,
                )
                repository = RunRepository(settings.database_path)
                transcriber = GatedTranscriber()
                runner = BenchmarkRunner(settings, repository, transcriber)
                run_id = runner.create_run(BenchmarkRequest(model_ids=["model-a"], item_ids=["clip-a", "clip-b"]))
                execution = asyncio.create_task(runner.execute_run(run_id))

                await asyncio.wait_for(transcriber.first_call_started.wait(), timeout=1)
                self.assertEqual(repository.control_run(run_id, "stop"), "stopping")
                transcriber.release_first_call.set()
                await asyncio.wait_for(execution, timeout=1)
                return repository.get_run(run_id), transcriber.call_count

        stopped_run, call_count = asyncio.run(scenario())
        self.assertEqual(stopped_run["status"], "stopped")
        self.assertEqual(stopped_run["result_count"], 1)
        self.assertEqual(call_count, 1)


class FollowUpCapturingTranscriber:
    def __init__(self) -> None:
        self.calls = []

    async def transcribe(self, model, item, parameters, prompt, follow_up_prompt=None):
        self.calls.append({"item": item.id, "prompt": prompt, "follow_up_prompt": follow_up_prompt})
        return TranscriptionResponse(
            transcript="guten tag miteinander",
            conversation=[
                {"role": "assistant", "pass": "dialect", "contents": ["grüessech mitenand"]},
                {"role": "assistant", "contents": ["guten tag miteinander"]},
            ],
        )


class StrategyFixture:
    def _settings(self, root: Path) -> BenchmarkSettings:
        (root / "clip.wav").write_bytes(b"audio")
        (root / "models.json").write_text(
            '[{"id":"model-a","label":"Model A","deployment":"a","description":"a"}]', encoding="utf-8"
        )
        records = [
            ("be-0001", 1, "grüessech mitenand", "guten tag miteinander"),
            ("be-0002", 2, "i ha hunger", "ich habe hunger"),
            ("be-0003", 3, "mir gö hei", "wir gehen nach hause"),
            ("be-dup1", 1, "grüessech zäme", "guten tag zusammen"),
            ("zh-0004", 4, "ich han hunger", "ich habe hunger"),
        ]
        (root / "manifest.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "id": item_id,
                        "audio_path": "clip.wav",
                        "reference_transcript": dialect_text,
                        "standard_german_transcript": standard_text,
                        "source": "ETH SwissDial 1.1",
                        "dialect": item_id[:2].upper(),
                        "dialect_name": "Bernese German" if item_id.startswith("be") else "Zurich German",
                        "sentence_id": sentence_id,
                    }
                )
                + "\n"
                for item_id, sentence_id, dialect_text, standard_text in records
            ),
            encoding="utf-8",
        )
        (root / "dialects.json").write_text(
            json.dumps(
                {
                    "cantons": {"BE": {"name": "Bern", "population": 100, "german_share": 1}},
                    "dialects": [
                        {
                            "code": "BE",
                            "name": "Bernese German",
                            "native_name": "Bärndütsch",
                            "region_cantons": ["BE"],
                            "features": ["nd becomes ng: hinger = hinter"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return BenchmarkSettings(
            model_registry_path=root / "models.json",
            voice_options_path=root / "voice-options.json",
            manifest_path=root / "manifest.jsonl",
            database_path=root / "benchmark.sqlite3",
            foundry_project_endpoint=None,
            trace_enabled=False,
            trace_sensitive_data=False,
            trace_port=4317,
            dialects_path=root / "dialects.json",
        )

class PromptStrategyTests(StrategyFixture, unittest.TestCase):
    def _run(self, strategy: str, reference_mode: str) -> tuple[dict, FollowUpCapturingTranscriber]:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings = self._settings(Path(temp_directory))
            transcriber = FollowUpCapturingTranscriber()
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, transcriber)
            run_id = runner.create_run(
                BenchmarkRequest(
                    model_ids=["model-a"],
                    item_ids=["be-0001"],
                    reference_mode=reference_mode,
                    strategy=strategy,
                )
            )
            asyncio.run(runner.execute_run(run_id))
            return repository.get_run(run_id), transcriber

    def test_guided_translation_adds_features_and_same_dialect_examples_without_test_sentence(self) -> None:
        run, transcriber = self._run("guided", "standard-german")
        prompt = transcriber.calls[0]["prompt"]

        self.assertEqual(run["strategy"], "guided")
        self.assertIsNone(transcriber.calls[0]["follow_up_prompt"])
        self.assertIn("Bärndütsch", prompt)
        self.assertIn("hinger = hinter", prompt)
        self.assertIn("Dialect: i ha hunger\nStandard German: ich habe hunger", prompt)
        self.assertIn("never 'ß'", prompt)
        self.assertNotIn("grüessech", prompt)
        self.assertNotIn("ich han hunger", prompt)
        self.assertEqual(run["results"][0]["word_error_rate"], 0)
        self.assertEqual(run["results"][0]["chrf"], 1)

    def test_guided_translation_uses_filtered_pool_not_other_evaluation_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            settings = self._settings(root)
            (root / "examples.jsonl").write_text(
                json.dumps({
                    "sentence_id": 99, "de": "Guten Morgen",
                    "dialects": {"be": "Guete Morge"},
                }) + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(**{**settings.__dict__, "examples_path": root / "examples.jsonl"})
            transcriber = FollowUpCapturingTranscriber()
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, transcriber)
            run_id = runner.create_run(BenchmarkRequest(
                model_ids=["model-a"], item_ids=["be-0001"],
                reference_mode="standard-german", strategy="guided",
            ))
            asyncio.run(runner.execute_run(run_id))
            prompt = transcriber.calls[0]["prompt"]
            self.assertIn("Dialect: Guete Morge\nStandard German: Guten Morgen", prompt)
            self.assertNotIn("i ha hunger", prompt)
            self.assertNotIn("grüessech", prompt)
            self.assertEqual(repository.get_run(run_id)["status"], "completed")

    def test_guided_dialect_transcription_shows_spelling_examples_only(self) -> None:
        _, transcriber = self._run("guided", "dialect")
        prompt = transcriber.calls[0]["prompt"]

        self.assertIn("- i ha hunger", prompt)
        self.assertNotIn("ich habe hunger", prompt)
        self.assertIn("Do not translate", prompt)

    def test_two_pass_sends_dialect_prompt_then_translation_follow_up(self) -> None:
        run, transcriber = self._run("two-pass", "standard-german")
        call = transcriber.calls[0]

        self.assertIn("Transcribe the recording verbatim in this dialect", call["prompt"])
        self.assertIn("Now translate your dialect transcript", call["follow_up_prompt"])
        self.assertIn("Standard German: ich habe hunger", call["follow_up_prompt"])
        self.assertEqual(run["results"][0]["conversation"][0]["pass"], "dialect")

    def test_two_pass_requires_high_german_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings = self._settings(Path(temp_directory))
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), FollowUpCapturingTranscriber())
            with self.assertRaisesRegex(ValueError, "two-pass"):
                runner.create_run(
                    BenchmarkRequest(model_ids=["model-a"], item_ids=["be-0001"], strategy="two-pass")
                )
            with self.assertRaisesRegex(ValueError, "Unsupported strategy"):
                runner.create_run(BenchmarkRequest(model_ids=["model-a"], item_ids=["be-0001"], strategy="magic"))

    def test_realtime_two_pass_reuses_session_for_translation_turn(self) -> None:
        class TwoPassSocket:
            def __init__(self) -> None:
                self.sent = []
                self.events = [
                    {"type": "session.created"},
                    {"type": "session.updated"},
                    {"type": "response.output_text.delta", "delta": "grüessech"},
                    {"type": "response.done", "response": {}},
                    {"type": "conversation.item.added"},
                    {"type": "response.output_text.delta", "delta": "guten tag"},
                    {"type": "response.done", "response": {}},
                ]

            async def send(self, message: str) -> None:
                self.sent.append(json.loads(message))

            async def recv(self) -> str:
                return json.dumps(self.events.pop(0))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args) -> None:
                return None

        socket = TwoPassSocket()
        transcriber = RealtimeAudioTranscriber(settings=None)
        with patch("backend.app.transcriber.websockets.connect", return_value=socket):
            transcript, _, first_pass = asyncio.run(
                transcriber._run_session("wss://example", "token", b"\x00" * 10, "dialect prompt", "translate now")
            )

        self.assertEqual(transcript, "guten tag")
        self.assertEqual(first_pass, "grüessech")
        follow_up_item = next(event for event in socket.sent if event["type"] == "conversation.item.create")
        self.assertEqual(follow_up_item["item"]["content"][0]["text"], "translate now")
        responses = [event for event in socket.sent if event["type"] == "response.create"]
        self.assertEqual([event["response"]["instructions"] for event in responses], ["dialect prompt", "translate now"])


if __name__ == "__main__":
    unittest.main()

class PassAwareTranscriber:
    def __init__(self, fail_baseline: bool = False) -> None:
        self.prompts = []
        self.fail_baseline = fail_baseline

    async def transcribe(self, model, item, parameters, prompt, follow_up_prompt=None):
        self.prompts.append(prompt)
        if "Swiss Standard German translation" in prompt:
            text = "guten tag miteinander"
        elif "verbatim in this dialect" in prompt:
            text = "grüessech mitenand"
        else:
            if self.fail_baseline:
                raise RuntimeError("server rejected WebSocket connection: HTTP 429")
            text = "grüessech mitenang"
        return TranscriptionResponse(transcript=text, conversation=[])


class FakeRefiner:
    deployment = "fake-text-model"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts = []

    async def complete(self, instructions, prompt):
        self.prompts.append(prompt)
        return self.answer


class EnsembleStrategyTests(StrategyFixture, unittest.TestCase):
    def _ensemble(self, reference_mode: str, answer: str, fail_baseline: bool = False):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            settings = self._settings(root)
            (root / "examples.jsonl").write_text(
                json.dumps({"sentence_id": 90, "de": "guten tag", "dialects": {"be": "grüessech wohl"}})
                + "\n"
                + json.dumps({"sentence_id": 1, "de": "leak", "dialects": {"be": "grüessech leak"}})
                + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(**{**settings.__dict__, "examples_path": root / "examples.jsonl"})
            transcriber = PassAwareTranscriber(fail_baseline)
            refiner = FakeRefiner(answer)
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, transcriber, refiner)
            run_id = runner.create_run(
                BenchmarkRequest(
                    model_ids=["model-a"], item_ids=["be-0001"], reference_mode=reference_mode, strategy="ensemble"
                )
            )
            asyncio.run(runner.execute_run(run_id))
            return repository.get_run(run_id), transcriber, refiner

    def test_ensemble_fuses_three_realtime_hypotheses_into_high_german(self) -> None:
        run, transcriber, refiner = self._ensemble("standard-german", "guten tag miteinander")
        result = run["results"][0]

        self.assertEqual(len(transcriber.prompts), 3)
        self.assertEqual(result["transcript"], "guten tag miteinander")
        self.assertEqual(result["word_error_rate"], 0)
        self.assertIn("Swiss German transcript A: grüessech mitenand", refiner.prompts[0])
        self.assertIn("Standard German translation A: guten tag miteinander", refiner.prompts[0])
        self.assertNotIn("Example sentences", refiner.prompts[0])
        passes = [entry["pass"] for entry in result["conversation"]]
        self.assertEqual(passes, ["guided-dialect", "guided-standard", "baseline", "fusion"])
        self.assertEqual(result["conversation"][-1]["model"], "fake-text-model")

    def test_ensemble_dialect_mode_retrieves_spelling_examples_without_evaluated_sentence(self) -> None:
        run, _, refiner = self._ensemble("dialect", "grüessech mitenand")
        prompt = refiner.prompts[0]

        self.assertEqual(run["results"][0]["word_error_rate"], 0)
        self.assertIn("- grüessech wohl", prompt)
        self.assertNotIn("leak", prompt)
        self.assertIn("Swiss German transcript B: grüessech mitenang", prompt)
        self.assertIn("Do not translate", prompt)

    def test_ensemble_continues_when_one_pass_fails(self) -> None:
        run, _, refiner = self._ensemble("standard-german", "guten tag miteinander", fail_baseline=True)
        result = run["results"][0]

        self.assertEqual(result["status"], "completed")
        self.assertIn({"role": "system", "pass": "failed", "contents": ["baseline"]}, result["conversation"])
        self.assertEqual(len(refiner.prompts), 1)

    def test_ensemble_requires_configured_refiner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings = self._settings(Path(temp_directory))
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), PassAwareTranscriber())
            with self.assertRaisesRegex(ValueError, "BENCHMARK_REFINER_DEPLOYMENT"):
                runner.create_run(BenchmarkRequest(model_ids=["model-a"], item_ids=["be-0001"], strategy="ensemble"))


class ExamplePoolAndTokenTests(unittest.TestCase):
    def test_example_pool_ranks_by_informative_overlap_and_excludes_sentence(self) -> None:
        from backend.app.examples import ExamplePool

        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "examples.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in [
                        {"sentence_id": 1, "de": "a", "dialects": {"be": "dr Hund bället"}},
                        {"sentence_id": 2, "de": "b", "dialects": {"be": "dr Chüngu frisst"}},
                        {"sentence_id": 3, "de": "c", "dialects": {"be": "dr Hund schlaft"}},
                        {"sentence_id": 4, "de": "d", "dialects": {"zh": "de Hund bellt"}},
                    ]
                ),
                encoding="utf-8",
            )
            pool = ExamplePool.load(path)

        self.assertEqual(pool.similar("BE", "dr Hund bället lut", 2), [("dr Hund bället", "a"), ("dr Hund schlaft", "c")])
        self.assertEqual(pool.similar("be", "dr Hund bället", 1, exclude_sentence_id=1), [("dr Hund schlaft", "c")])
        self.assertEqual(len(pool.sample("BE", "clip", 5, exclude_sentence_id="2")), 2)
        self.assertNotIn("VS", pool)
        self.assertIsNone(ExamplePool.load(Path("missing.jsonl")))

    def test_token_cache_reuses_token_until_close_to_expiry(self) -> None:
        import time as time_module
        from types import SimpleNamespace

        from backend.app.transcriber import TokenCache

        class Credential:
            def __init__(self) -> None:
                self.calls = 0

            async def get_token(self, scope):
                self.calls += 1
                lifetime = 3600 if self.calls == 1 else 7200
                return SimpleNamespace(token=f"token-{self.calls}", expires_on=time_module.time() + lifetime)

        async def scenario():
            cache = TokenCache()
            cache._credential = Credential()
            first = await cache.get("scope")
            second = await cache.get("scope")
            cache._tokens["scope"] = SimpleNamespace(token="stale", expires_on=time_module.time() + 60)
            third = await cache.get("scope")
            return first, second, third, cache._credential.calls

        self.assertEqual(asyncio.run(scenario()), ("token-1", "token-1", "token-2", 2))


class RecoveryTests(unittest.TestCase):
    def test_interrupted_runs_are_paused_or_stopped_on_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            repository = RunRepository(Path(temp_directory) / "benchmark.sqlite3")
            ids = {}
            for status in ("queued", "running", "stopping", "completed", "paused"):
                run_id = repository.create_run(
                    model_ids=["m"], item_ids=["i"], parameters={"m": {}}, prompt="p", reference_mode="dialect"
                )
                repository.set_run_status(run_id, status)
                ids[status] = run_id

            recovered = repository.recover_interrupted_runs()
            statuses = {status: repository.get_run_status(run_id) for status, run_id in ids.items()}

        self.assertEqual(recovered, 3)
        self.assertEqual(
            statuses,
            {"queued": "paused", "running": "paused", "stopping": "stopped", "completed": "completed", "paused": "paused"},
        )


class ModelAdapterTests(unittest.TestCase):
    def _clip(self, root: Path):
        from backend.app.domain import DatasetItem

        (root / "clip.wav").write_bytes(b"RIFFaudio")
        return DatasetItem(id="be-0001", audio_path=root / "clip.wav", reference_transcript="x", source="s", metadata={})

    def _model(self, transport: str):
        from backend.app.domain import ModelDefinition

        return ModelDefinition(id="m", label="Model", deployment="dep", description="d", capabilities=(), transport=transport)

    def test_transcription_adapter_uses_deployment_route_with_prompt_as_context(self) -> None:
        from types import SimpleNamespace

        from backend.app.transcriber import TranscriptionApiTranscriber

        calls = {}

        class FakeAzureClient:
            def __init__(self, **kwargs) -> None:
                calls["client"] = kwargs

                async def create(**request):
                    calls["request"] = request
                    return SimpleNamespace(text=" grüessech mitenand ")

                self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=create))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class Tokens:
            async def get(self, scope):
                return "token"

        with tempfile.TemporaryDirectory() as temp_directory, patch.dict(
            "os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com/openai/v1"}
        ), patch("openai.AsyncAzureOpenAI", FakeAzureClient):
            item = self._clip(Path(temp_directory))
            adapter = TranscriptionApiTranscriber(Tokens())
            response = asyncio.run(adapter.transcribe(self._model("azure-openai-transcription"), item, {}, "context"))
            with self.assertRaisesRegex(ValueError, "two-pass"):
                asyncio.run(adapter.transcribe(self._model("azure-openai-transcription"), item, {}, "context", "follow"))

        self.assertEqual(response.transcript, "grüessech mitenand")
        self.assertEqual(calls["client"]["azure_endpoint"], "https://res.openai.azure.com")
        self.assertEqual(calls["client"]["azure_ad_token"], "token")
        self.assertEqual(calls["request"]["model"], "dep")
        self.assertEqual(calls["request"]["prompt"], "context")
        self.assertEqual(calls["request"]["file"][0], "clip.wav")

    def test_audio_chat_adapter_streams_text_and_supports_follow_up_turn(self) -> None:
        from types import SimpleNamespace

        from backend.app.transcriber import AudioChatTranscriber

        requests = []
        answers = [["grüessech ", "mitenand"], ["guten tag ", "miteinander"]]

        class Stream:
            def __init__(self, parts) -> None:
                self.parts = list(parts)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.parts:
                    raise StopAsyncIteration
                return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.parts.pop(0)))])

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                async def create(**request):
                    requests.append(request)
                    return Stream(answers[len(requests) - 1])

                self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class Tokens:
            async def get(self, scope):
                return "token"

        with tempfile.TemporaryDirectory() as temp_directory, patch.dict(
            "os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"}
        ), patch("openai.AsyncOpenAI", FakeClient):
            item = self._clip(Path(temp_directory))
            response = asyncio.run(
                AudioChatTranscriber(Tokens()).transcribe(self._model("azure-openai-audio-chat"), item, {}, "dialect", "translate")
            )

        self.assertEqual(response.transcript, "guten tag miteinander")
        self.assertIsNotNone(response.time_to_first_token_ms)
        audio_part = requests[0]["messages"][1]["content"][1]
        self.assertEqual(audio_part["type"], "input_audio")
        self.assertEqual(audio_part["input_audio"]["format"], "wav")
        self.assertEqual(requests[1]["messages"][-2], {"role": "assistant", "content": "grüessech mitenand"})
        self.assertEqual(requests[1]["messages"][-1], {"role": "user", "content": "translate"})
        self.assertEqual(response.conversation[1]["pass"], "dialect")

    def test_two_pass_is_rejected_for_transcription_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "models.json").write_text(
                '[{"id":"stt","label":"STT","deployment":"d","description":"d","transport":"azure-openai-transcription"}]',
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps({"id": "a", "audio_path": "a.wav", "reference_transcript": "x", "standard_german_transcript": "y", "source": "s"}) + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "v.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "db.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
            )
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), FakeTranscriber())
            with self.assertRaisesRegex(ValueError, "conversational model"):
                runner.create_run(
                    BenchmarkRequest(model_ids=["stt"], item_ids=["a"], reference_mode="standard-german", strategy="two-pass")
                )


class MultiModelEnsembleTests(unittest.TestCase):
    def _setup(self, root: Path):
        (root / "clip.wav").write_bytes(b"audio")
        (root / "models.json").write_text(
            json.dumps(
                [
                    {"id": "rt", "label": "Realtime", "deployment": "rt", "description": "d", "transport": "azure-openai-realtime"},
                    {"id": "stt", "label": "STT", "deployment": "stt", "description": "d", "transport": "azure-openai-transcription"},
                    {
                        "id": "mix", "label": "Mix", "deployment": "fusion", "description": "d", "transport": "ensemble",
                        "members": [{"model": "rt", "pass": "target"}, {"model": "stt", "pass": "target"}, {"model": "stt", "pass": "dialect"}],
                    },
                    {"id": "bad", "label": "Bad", "deployment": "x", "description": "d", "transport": "ensemble", "members": [{"model": "mix"}]},
                ]
            ),
            encoding="utf-8",
        )
        (root / "manifest.jsonl").write_text(
            json.dumps(
                {
                    "id": "be-0001", "audio_path": "clip.wav", "reference_transcript": "grüessech mitenand",
                    "standard_german_transcript": "guten tag miteinander", "source": "s", "dialect": "BE",
                    "dialect_name": "Bernese German", "sentence_id": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return BenchmarkSettings(
            model_registry_path=root / "models.json",
            voice_options_path=root / "v.json",
            manifest_path=root / "manifest.jsonl",
            database_path=root / "db.sqlite3",
            foundry_project_endpoint=None,
            trace_enabled=False,
            trace_sensitive_data=False,
            trace_port=4317,
            examples_path=root / "missing-examples.jsonl",
        )

    def test_members_are_heard_once_each_and_fused(self) -> None:
        class MemberTranscriber:
            def __init__(self) -> None:
                self.calls = []

            async def transcribe(self, model, item, parameters, prompt, follow_up_prompt=None):
                dialect = "verbatim in this dialect" in prompt
                self.calls.append((model.id, "dialect" if dialect else "standard"))
                return TranscriptionResponse(transcript=f"{model.id}-{'ch' if dialect else 'de'}", conversation=[])

        with tempfile.TemporaryDirectory() as temp_directory:
            settings = self._setup(Path(temp_directory))
            transcriber = MemberTranscriber()
            refiner = FakeRefiner("guten tag miteinander")
            repository = RunRepository(settings.database_path)
            runner = BenchmarkRunner(settings, repository, transcriber, refiner)
            run_id = runner.create_run(
                BenchmarkRequest(model_ids=["mix"], item_ids=["be-0001"], reference_mode="standard-german", strategy="ensemble")
            )
            asyncio.run(runner.execute_run(run_id))
            result = repository.get_run(run_id)["results"][0]

        self.assertEqual(sorted(transcriber.calls), [("rt", "standard"), ("stt", "dialect"), ("stt", "standard")])
        self.assertEqual(result["word_error_rate"], 0)
        self.assertIn("Standard German translation A: rt-de", refiner.prompts[0])
        self.assertIn("Standard German translation B: stt-de", refiner.prompts[0])
        self.assertIn("Swiss German transcript A: stt-ch", refiner.prompts[0])
        self.assertEqual([entry.get("model") for entry in result["conversation"]], ["Realtime", "STT", "STT", "fake-text-model"])

    def test_multi_model_ensemble_requires_ensemble_strategy_and_valid_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings = self._setup(Path(temp_directory))
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), FakeTranscriber(), FakeRefiner("x"))
            with self.assertRaisesRegex(ValueError, "select the Ensemble strategy"):
                runner.create_run(BenchmarkRequest(model_ids=["mix"], item_ids=["be-0001"], strategy="guided"))
            with self.assertRaisesRegex(ValueError, "nested member"):
                runner.create_run(BenchmarkRequest(model_ids=["bad"], item_ids=["be-0001"], strategy="ensemble"))
