import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.runner import BenchmarkRequest
from backend.app.settings import BenchmarkSettings
from backend.app.transcriber import TranscriptionResponse


class FakeTranscriber:
    async def transcribe(self, model, item, parameters, prompt):
        return TranscriptionResponse(transcript="grüezi", conversation=[], time_to_first_token_ms=125)


class ApiTests(unittest.TestCase):
    def test_exposes_models_dataset_and_queues_valid_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "clips").mkdir()
            (root / "clips" / "clip.mp3").write_bytes(b"audio")
            (root / "models.json").write_text(
                '[{"id":"model-a","label":"Model A","deployment":"a","description":"Audio","capabilities":["audio"]}]',
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "id": "clip-a",
                        "audio_path": "clips/clip.mp3",
                        "reference_transcript": "grüezi",
                        "standard_german_transcript": "guten tag",
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
            (root / "voice-options.json").write_text(
                '[{"id":"voice-live","label":"Voice Live API","availability":"managed-api"}]',
                encoding="utf-8",
            )
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                health = client.get("/api/health").json()
                self.assertEqual(health["dataset_item_count"], 1)
                self.assertEqual(health["authentication"], "rbac-default-azure-credential")
                self.assertEqual(health["voice_option_count"], 1)
                self.assertEqual(client.get("/api/models").json()[0]["id"], "model-a")
                self.assertEqual(client.get("/api/voice-options").json()[0]["id"], "voice-live")
                saved_instruction = client.post(
                    "/api/instructions",
                    json={"name": "Swiss German verbatim", "prompt": "Write only the Swiss German transcript."},
                )
                self.assertEqual(saved_instruction.status_code, 200)
                self.assertEqual(saved_instruction.json()["name"], "Swiss German verbatim")
                self.assertEqual(client.get("/api/instructions").json()[0]["prompt"], "Write only the Swiss German transcript.")
                item = client.get("/api/dataset/items").json()[0]
                self.assertTrue(item["audio_available"])
                self.assertEqual(item["reference_transcript"], "grüezi")
                audio = client.get("/api/dataset/items/clip-a/audio")
                self.assertEqual(audio.status_code, 200)
                self.assertEqual(audio.headers["content-type"], "audio/mpeg")
                self.assertEqual(audio.content, b"audio")
                self.assertEqual(client.get("/api/dataset/items/missing/audio").status_code, 404)
                self.assertEqual(client.get("/%3Aundefined").status_code, 404)
                self.assertEqual(client.get("/api/v2/%24%7Bjndi%3Adns%3Aexample%7D").status_code, 404)
                response = client.post("/api/runs", json={"model_ids": ["model-a"], "item_ids": ["clip-a"]})
                self.assertEqual(response.status_code, 202)
                run = client.get(f"/api/runs/{response.json()['id']}").json()
                history_summary = client.get("/api/runs/summary").json()
                export = client.get(f"/api/runs/{response.json()['id']}/export.csv")
                high_german_response = client.post(
                    "/api/runs",
                    json={
                        "model_ids": ["model-a"],
                        "item_ids": ["clip-a"],
                        "reference_mode": "standard-german",
                    },
                )
                self.assertEqual(high_german_response.status_code, 202)
                high_german_run = client.get(f"/api/runs/{high_german_response.json()['id']}").json()
                guided_response = client.post(
                    "/api/runs",
                    json={
                        "model_ids": ["model-a"],
                        "item_ids": ["clip-a"],
                        "reference_mode": "standard-german",
                        "strategy": "guided",
                    },
                )
                self.assertEqual(guided_response.status_code, 202)
                guided_run = client.get(f"/api/runs/{guided_response.json()['id']}").json()
                guided_export = client.get(f"/api/runs/{guided_response.json()['id']}/export.csv")
                invalid_strategy = client.post(
                    "/api/runs", json={"model_ids": ["model-a"], "item_ids": ["clip-a"], "strategy": "two-pass"}
                )
                dialect_atlas = client.get("/api/dialects").json()

        self.assertEqual(guided_run["strategy"], "guided")
        self.assertIn("strategy", guided_export.text.splitlines()[0])
        self.assertIn("chrf", guided_export.text.splitlines()[0])
        self.assertEqual(invalid_strategy.status_code, 422)
        self.assertEqual(len(dialect_atlas["dialects"]), 8)
        self.assertIn("share_of_german_speakers", dialect_atlas["dialects"][0])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["reference_mode"], "dialect")
        self.assertEqual(run["strategy"], "baseline")
        self.assertEqual(run["results"][0]["word_error_rate"], 0)
        self.assertEqual(run["results"][0]["word_match_rate"], 1)
        self.assertEqual(run["results"][0]["time_to_first_token_ms"], 125)
        self.assertEqual(run["average_time_to_first_token_ms"], 125)
        self.assertEqual(history_summary["total_run_count"], 1)
        self.assertEqual(history_summary["successful_result_count"], 1)
        self.assertEqual(history_summary["indicator"], {"label": "Strong overall", "tone": "positive"})
        self.assertEqual(export.status_code, 200)
        self.assertIn("text/csv", export.headers["content-type"])
        self.assertIn("reference_utterance", export.text)
        self.assertIn("time_to_first_token_ms", export.text)
        self.assertIn("grüezi", export.text)
        self.assertEqual(high_german_run["reference_mode"], "standard-german")
        self.assertIn("High German", high_german_run["prompt"])
        self.assertEqual(high_german_run["results"][0]["reference_transcript"], "guten tag")

    def test_rejects_unknown_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "models.json").write_text("[]", encoding="utf-8")
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
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                response = client.post("/api/runs", json={"model_ids": ["missing"], "item_ids": ["missing"]})

        self.assertEqual(response.status_code, 422)
        self.assertIn("Unknown model IDs", response.json()["detail"])

    def test_run_control_endpoints_pause_resume_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "clips").mkdir()
            (root / "clips" / "clip.mp3").write_bytes(b"audio")
            (root / "models.json").write_text(
                '[{"id":"model-a","label":"Model A","deployment":"a","description":"Audio"}]',
                encoding="utf-8",
            )
            (root / "manifest.jsonl").write_text(
                '{"id":"clip-a","audio_path":"clips/clip.mp3","source":"SwissDial"}\n',
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
            app = create_app(settings, FakeTranscriber())
            with TestClient(app) as client:
                resumable_id = app.state.runner.create_run(
                    BenchmarkRequest(model_ids=["model-a"], item_ids=["clip-a"])
                )
                paused = client.post(f"/api/runs/{resumable_id}/pause")
                self.assertEqual(paused.status_code, 200)
                self.assertEqual(paused.json()["status"], "paused")

                resumed = client.post(f"/api/runs/{resumable_id}/resume")
                self.assertEqual(resumed.status_code, 200)
                self.assertIn(resumed.json()["status"], {"running", "completed"})

                stoppable_id = app.state.runner.create_run(
                    BenchmarkRequest(model_ids=["model-a"], item_ids=["clip-a"])
                )
                self.assertEqual(client.post(f"/api/runs/{stoppable_id}/pause").status_code, 200)
                stopped = client.post(f"/api/runs/{stoppable_id}/stop")
                self.assertEqual(stopped.status_code, 200)
                self.assertEqual(stopped.json()["status"], "stopped")
                self.assertEqual(client.post(f"/api/runs/{stoppable_id}/resume").status_code, 409)
                self.assertEqual(client.post("/api/runs/missing/pause").status_code, 404)


if __name__ == "__main__":
    unittest.main()

class StartupRecoveryTests(unittest.TestCase):
    def test_interrupted_runs_are_recovered_on_server_start_not_on_app_creation(self) -> None:
        from backend.app.repository import RunRepository

        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "models.json").write_text("[]", encoding="utf-8")
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
            run_id = repository.create_run(
                model_ids=["m"], item_ids=["i"], parameters={"m": {}}, prompt="p", reference_mode="dialect"
            )
            repository.set_run_status(run_id, "running")

            app = create_app(settings, FakeTranscriber())
            status_after_create = repository.get_run_status(run_id)
            with TestClient(app):
                status_after_start = repository.get_run_status(run_id)

        self.assertEqual(status_after_create, "running")
        self.assertEqual(status_after_start, "paused")
