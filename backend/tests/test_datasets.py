import asyncio
from dataclasses import replace
import io
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.api import create_app
from backend.app.catalog import load_all_dataset_items
from backend.app.domain import DatasetItem, ModelDefinition
from backend.app.examples import ExamplePair, ExamplePool
from backend.app.repository import RunRepository
from backend.app.runner import BenchmarkRunner
from backend.app.settings import BenchmarkSettings, load_settings
from backend.app.transcriber import TranscriptionResponse


def archive(records, audio_files=None, extra_files=None):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as output:
        output.writestr("manifest.jsonl", "".join(json.dumps(record) + "\n" for record in records))
        for name, data in (audio_files or {"clips/one.wav": b"RIFFaudio"}).items():
            output.writestr(name, data)
        for name, data in (extra_files or {}).items():
            output.writestr(name, data)
    return content.getvalue()


class FakeTranscriber:
    async def transcribe(self, model, item, parameters, prompt):
        return TranscriptionResponse(transcript="grüezi", conversation=[])


class CaptureRefiner:
    deployment = "test-refiner"

    async def complete(self, instructions, prompt):
        self.prompt = prompt
        return "grüezi"


class DatasetUploadTests(unittest.TestCase):
    def _settings(self, root):
        (root / "models.json").write_text(
            '[{"id":"test","label":"Test","deployment":"test","description":"Test"}]', encoding="utf-8"
        )
        (root / "manifest.jsonl").write_text(
            json.dumps({"id": "base", "audio_path": "base.wav", "reference_transcript": "grüezi", "source": "SwissDial"})
            + "\n", encoding="utf-8"
        )
        (root / "base.wav").write_bytes(b"RIFFbase")
        return BenchmarkSettings(
            model_registry_path=root / "models.json",
            voice_options_path=root / "voice.json",
            manifest_path=root / "manifest.jsonl",
            database_path=root / "runs.sqlite3",
            foundry_project_endpoint=None,
            trace_enabled=False,
            trace_sensitive_data=False,
            trace_port=4317,
            uploaded_datasets_path=root / "uploads",
        )

    def test_imports_multiple_archives_and_runs_from_uploaded_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            first = [{"id": "own-1", "audio_path": "clips/one.wav", "reference_transcript": "grüezi",
                      "standard_german_transcript": "hallo", "dialect": "BE"}]
            second = [{"id": "own-2", "audio_path": "clips/two.mp3", "reference_transcript": "grüezi"}]
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                response = client.post(
                    "/api/datasets?name=My%20recordings", content=archive(first), headers={"Content-Type": "application/zip"}
                )
                self.assertEqual(response.status_code, 201, response.text)
                self.assertEqual(response.json()["item_count"], 1)
                self.assertEqual(client.get("/api/health").json()["dataset_item_count"], 2)
                self.assertEqual(client.get("/api/dataset/items/own-1/audio").content, b"RIFFaudio")
                other = client.post(
                    "/api/datasets?name=Second", content=archive(
                        second, audio_files={"clips/two.mp3": b"MP3audio"}
                    ), headers={"Content-Type": "application/zip"}
                )
                self.assertEqual(other.status_code, 201, other.text)
                duplicate_name = client.post(
                    "/api/datasets?name=my%20RECORDINGS",
                    content=archive([{"id": "own-3", "audio_path": "clips/one.wav"}]),
                    headers={"Content-Type": "application/zip"},
                )
                self.assertEqual(duplicate_name.status_code, 409, duplicate_name.text)
                items = {item["id"]: item for item in client.get("/api/dataset/items").json()}
                self.assertEqual(set(items), {"base", "own-1", "own-2"})
                self.assertEqual(items["own-1"]["metadata"]["dataset"], "My recordings")
                run = client.post("/api/runs", json={"model_ids": ["test"], "item_ids": ["own-1"]})
                self.assertEqual(run.status_code, 202, run.text)
                self.assertEqual(client.get(f"/api/runs/{run.json()['id']}").json()["results"][0]["status"], "completed")

            self.assertEqual(len(load_all_dataset_items(settings.manifest_path, settings.uploaded_datasets_path)), 3)
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                self.assertEqual(client.get("/api/health").json()["dataset_item_count"], 3)
                self.assertEqual(client.get(f"/api/runs/{run.json()['id']}").json()["results"][0]["transcript"], "grüezi")

    def test_rejects_unsafe_archives_without_publishing_them(self):
        cases = [
            ("missing audio", archive([{"id": "own", "audio_path": "clips/missing.wav"}]), "missing:"),
            ("escape path", archive([{"id": "own", "audio_path": "../base.wav"}]), "Invalid archive path"),
            ("zip traversal", archive([{"id": "own", "audio_path": "clips/one.wav"}],
                                      extra_files={"../outside.wav": b"bad"}), "Invalid archive path"),
            ("extra file", archive([{"id": "own", "audio_path": "clips/one.wav"}],
                                   extra_files={"clips/secret.txt": b"bad"}), "unreferenced:"),
            ("existing id", archive([{"id": "base", "audio_path": "clips/one.wav"}]), "Duplicate dataset item ID"),
            ("duplicate id", archive([{"id": "own", "audio_path": "clips/one.wav"},
                                      {"id": "own", "audio_path": "clips/one.wav"}]), "Duplicate dataset item ID"),
            ("empty audio", archive([{"id": "own", "audio_path": "clips/one.wav"}],
                                    audio_files={"clips/one.wav": b""}), "Audio file is empty"),
            ("windows device", archive([{"id": "own", "audio_path": "clips/CON.wav"}],
                                       audio_files={"clips/CON.wav": b"RIFFaudio"}), "Invalid archive path"),
            ("nontext reference", archive([{"id": "own", "audio_path": "clips/one.wav",
                                            "reference_transcript": 1}]), "non-text reference_transcript"),
            ("file used as directory", archive([
                {"id": "one", "audio_path": "clips/one.wav"},
                {"id": "two", "audio_path": "clips/one.wav/two.wav"},
            ], audio_files={"clips/one.wav": b"RIFFone", "clips/one.wav/two.wav": b"RIFFtwo"}),
             "cannot also be a directory"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                for label, payload, message in cases:
                    with self.subTest(label=label):
                        response = client.post("/api/datasets?name=Bad", content=payload,
                                               headers={"Content-Type": "application/zip"})
                        self.assertEqual(response.status_code, 422, response.text)
                        self.assertIn(message, response.json()["detail"])
                self.assertEqual(client.get("/api/health").json()["dataset_item_count"], 1)
            self.assertFalse(list(settings.uploaded_datasets_path.iterdir()))
            self.assertFalse((Path(directory).parent / "outside.wav").exists())

    def test_rejects_duplicate_zip_entries_symlinks_and_wrong_content_type(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            payload = archive([{"id": "own", "audio_path": "clips/one.wav"}])
            duplicate = io.BytesIO()
            with zipfile.ZipFile(duplicate, "w") as output:
                output.writestr("manifest.jsonl", b'{"id":"own","audio_path":"clips/one.wav"}\n')
                output.writestr("clips/one.wav", b"RIFFone")
                output.writestr("clips/one.wav", b"RIFFtwo")
            symlink = io.BytesIO()
            with zipfile.ZipFile(symlink, "w") as output:
                output.writestr("manifest.jsonl", b'{"id":"own","audio_path":"clips/one.wav"}\n')
                link = zipfile.ZipInfo("clips/one.wav")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(link, b"../../base.wav")
            with TestClient(create_app(settings, FakeTranscriber())) as client:
                self.assertEqual(client.post("/api/datasets?name=Bad", content=payload).status_code, 415)
                self.assertEqual(client.post("/api/datasets?name=Bad", content=b"not a ZIP",
                                             headers={"Content-Type": "application/zip"}).status_code, 422)
                self.assertEqual(client.post("/api/datasets?name=Bad", content=duplicate.getvalue(),
                                             headers={"Content-Type": "application/zip"}).status_code, 422)
                self.assertEqual(client.post("/api/datasets?name=Bad", content=symlink.getvalue(),
                                             headers={"Content-Type": "application/zip"}).status_code, 422)
                with patch("backend.app.api.MAX_ARCHIVE_BYTES", 32):
                    self.assertEqual(client.post("/api/datasets?name=Bad", content=payload,
                                                 headers={"Content-Type": "application/zip"}).status_code, 413)
                with patch("backend.app.datasets.MAX_EXTRACTED_BYTES", 8):
                    self.assertEqual(client.post("/api/datasets?name=Bad", content=payload,
                                                 headers={"Content-Type": "application/zip"}).status_code, 422)

    def test_custom_items_do_not_borrow_swissdial_few_shot_examples(self):
        swissdial = DatasetItem("base", Path("base.wav"), "base", "SwissDial",
                                {"dialect": "ZH", "standard_german_transcript": "base de"})
        custom = DatasetItem("own", Path("own.wav"), "own", "Other",
                             {"dialect": "ZH", "standard_german_transcript": "own de", "dataset": "Other"})
        self.assertEqual(BenchmarkRunner._few_shot_examples(custom, [swissdial, custom]), [])
        self.assertEqual(BenchmarkRunner._few_shot_examples(swissdial, [swissdial, custom]), [])

    def test_uploaded_ensemble_uses_own_examples_instead_of_swissdial_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            settings = replace(settings, examples_path=Path(directory) / "examples.jsonl")
            item = DatasetItem("own", Path("own.wav"), "grüezi", "Own",
                               {"dialect": "ZH", "dataset": "Own"})
            peer = DatasetItem("peer", Path("peer.wav"), "own example", "Own",
                               {"dialect": "ZH", "dataset": "Own", "standard_german_transcript": "own translation"})
            pool = ExamplePool({"ZH": [ExamplePair(12, "SwissDial-only", "base de", frozenset({"grüezi"}))]})
            refiner = CaptureRefiner()
            runner = BenchmarkRunner(settings, RunRepository(settings.database_path), FakeTranscriber(), refiner)
            model = ModelDefinition("test", "Test", "test", "Test", ("audio",))
            asyncio.run(runner._hear_and_fuse(
                item, [("guided-dialect", model, {}, "prompt")], "dialect",
                [item, peer], {}, pool,
            ))
            self.assertIn("own example", refiner.prompt)
            self.assertNotIn("SwissDial-only", refiner.prompt)

    def test_custom_base_manifest_disables_default_swissdial_example_pool(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {
                "BENCHMARK_MANIFEST_PATH": str(Path(directory) / "own.jsonl"),
                "BENCHMARK_EXAMPLES_PATH": "",
            }
        ):
            settings = load_settings()
        self.assertIsNone(settings.examples_path)
