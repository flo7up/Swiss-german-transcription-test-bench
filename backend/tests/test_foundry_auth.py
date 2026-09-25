import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.catalog import load_models
from backend.app.domain import DatasetItem, ModelDefinition
from backend.app.refiner import AzureOpenAITextRefiner
from backend.app.settings import BenchmarkSettings
from backend.app.transcriber import (
    AudioChatTranscriber,
    FoundryAudioTranscriber,
    RealtimeAudioTranscriber,
    TranscriptionApiTranscriber,
)


class NoTokens:
    async def get(self, scope):
        raise AssertionError("API-key mode must not request an Entra token")


class FoundryAuthTests(unittest.TestCase):
    def test_custom_model_is_selectable_and_runnable_without_a_registry_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models.json").write_text("[]", encoding="utf-8")
            (root / "clip.wav").write_bytes(b"audio")
            (root / "manifest.jsonl").write_text(
                json.dumps({"id": "clip", "audio_path": "clip.wav", "reference_transcript": "grüezi", "source": "test"})
                + "\n",
                encoding="utf-8",
            )
            settings = BenchmarkSettings(
                model_registry_path=root / "models.json",
                voice_options_path=root / "voice.json",
                manifest_path=root / "manifest.jsonl",
                database_path=root / "db.sqlite3",
                foundry_project_endpoint=None,
                trace_enabled=False,
                trace_sensitive_data=False,
                trace_port=4317,
                foundry_api_key="secret",
                custom_model_name="my-audio-deployment",
            )

            class FakeTranscriber:
                async def transcribe(self, model, item, parameters, prompt):
                    self_model.append(model.deployment)
                    return SimpleNamespace(transcript="grüezi", conversation=[], time_to_first_token_ms=None)

            self_model = []
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                health = client.get("/api/health").json()
                models = client.get("/api/models").json()
                response = client.post("/api/runs", json={"model_ids": ["custom-foundry-model"], "item_ids": ["clip"]})
                run = client.get(f"/api/runs/{response.json()['id']}").json()
            self.assertEqual(health["authentication"], "api-key")
            self.assertNotIn("secret", str(health) + str(models) + str(run))
            self.assertEqual(health["model_count"], 1)
            self.assertEqual(models[0]["deployment"], "my-audio-deployment")
            self.assertEqual(response.status_code, 202)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(self_model, ["my-audio-deployment"])

    def test_custom_model_id_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(
                '[{"id":"custom-foundry-model","label":"a","deployment":"a","description":"a"}]', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "reserved ID"):
                load_models(path, "custom")

    def test_key_based_foundry_adapter_uses_direct_endpoint_and_custom_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clip.wav"
            audio.write_bytes(b"audio")
            item = DatasetItem("clip", audio, "hi", "test", {})
            model = ModelDefinition("custom", "custom", "my-deployment", "test", ())
            settings = BenchmarkSettings(
                root / "models.json", root / "voice.json", root / "manifest.jsonl", root / "db.sqlite3",
                None, False, False, 4317, foundry_api_key="secret",
            )
            seen = {}

            class FakeOpenAI:
                def __init__(self, **kwargs):
                    seen["openai"] = kwargs

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return None

            class FakeChatClient:
                def __init__(self, **kwargs):
                    seen["chat"] = kwargs

            async def fake_run(client, item, parameters, prompt, follow_up_prompt, media_type):
                return SimpleNamespace(transcript="hi", conversation=[])

            with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"}), patch(
                "openai.AsyncOpenAI", FakeOpenAI
            ), patch("agent_framework.openai.OpenAIChatClient", FakeChatClient), patch.object(
                FoundryAudioTranscriber, "_run", staticmethod(fake_run)
            ):
                response = asyncio.run(FoundryAudioTranscriber(settings).transcribe(model, item, {}, "transcribe"))
            self.assertEqual(response.transcript, "hi")
            self.assertEqual(seen["openai"]["api_key"], "secret")
            self.assertEqual(seen["openai"]["base_url"], "https://res.openai.azure.com/openai/v1/")
            self.assertEqual(seen["chat"]["model"], "my-deployment")

    def test_realtime_key_is_sent_in_api_key_header(self):
        seen = {}

        class Socket:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        def connect(url, **kwargs):
            seen.update(kwargs)
            return Socket()

        adapter = RealtimeAudioTranscriber(None, NoTokens())
        adapter._api_key = "secret"
        with patch("backend.app.transcriber.websockets.connect", connect), patch.object(
            adapter, "_receive_event", side_effect=RuntimeError("stop after connecting")
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after connecting"):
                asyncio.run(adapter._run_session("wss://example/realtime", "secret", b"audio", "prompt"))
        self.assertEqual(seen["additional_headers"], {"api-key": "secret"})

    def test_transcription_key_is_passed_to_azure_client(self):
        seen = {}

        class FakeClient:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.audio = SimpleNamespace(
                    transcriptions=SimpleNamespace(create=self.create)
                )

            async def create(self, **kwargs):
                return SimpleNamespace(text="hi")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"}
        ), patch("openai.AsyncAzureOpenAI", FakeClient):
            audio = Path(directory) / "clip.wav"
            audio.write_bytes(b"audio")
            item = DatasetItem("clip", audio, "hi", "test", {})
            model = ModelDefinition("m", "m", "custom-stt", "test", ())
            result = asyncio.run(TranscriptionApiTranscriber(NoTokens(), "secret").transcribe(model, item, {}, "context"))
        self.assertEqual(result.transcript, "hi")
        self.assertEqual(seen["api_key"], "secret")
        self.assertNotIn("azure_ad_token", seen)

    def test_text_refiner_uses_key_without_credential_chain(self):
        seen = {}

        class FakeClient:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.responses = SimpleNamespace(create=self.create)

            async def create(self, **kwargs):
                seen["model"] = kwargs["model"]
                return SimpleNamespace(output_text=" hi ")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"}), patch(
            "openai.AsyncOpenAI", FakeClient
        ):
            result = asyncio.run(AzureOpenAITextRefiner("custom-text", tokens=NoTokens(), api_key="secret").complete("i", "p"))
        self.assertEqual(result, "hi")
        self.assertEqual(seen["api_key"], "secret")
        self.assertEqual(seen["model"], "custom-text")

    def test_audio_chat_key_uses_custom_deployment(self):
        seen = {}

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if seen.get("read"):
                    raise StopAsyncIteration
                seen["read"] = True
                return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])

        class FakeClient:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            async def create(self, **kwargs):
                seen["model"] = kwargs["model"]
                return Stream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com"}
        ), patch("openai.AsyncOpenAI", FakeClient):
            audio = Path(directory) / "clip.wav"
            audio.write_bytes(b"audio")
            item = DatasetItem("clip", audio, "hi", "test", {})
            model = ModelDefinition("m", "m", "custom-audio", "test", ())
            result = asyncio.run(AudioChatTranscriber(NoTokens(), "secret").transcribe(model, item, {}, "prompt"))
        self.assertEqual(result.transcript, "hi")
        self.assertEqual(seen["api_key"], "secret")
        self.assertEqual(seen["model"], "custom-audio")
