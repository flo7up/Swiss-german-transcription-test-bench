import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.repository import RunRepository
from backend.app.runner import BenchmarkRequest, BenchmarkRunner
from backend.app.settings import BenchmarkSettings
from backend.app.transcriber import TranscriptionResponse


class FakeTranscriber:
    async def transcribe(self, model, item, parameters, prompt):
        return TranscriptionResponse(
            transcript="grüezi mitenand",
            conversation=[{"role": "assistant", "contents": ["grüezi mitenand"]}],
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


if __name__ == "__main__":
    unittest.main()