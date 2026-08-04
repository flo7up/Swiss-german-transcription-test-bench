import json
import tempfile
import unittest
from pathlib import Path

from backend.app.catalog import load_dataset_items, load_models


class CatalogTests(unittest.TestCase):
    def test_loads_model_definitions_and_manifest_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            directory = Path(temp_directory)
            model_path = directory / "models.json"
            model_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "model-a",
                            "label": "Model A",
                            "deployment": "deployment-a",
                            "description": "Audio model",
                            "capabilities": ["audio"],
                            "transport": "azure-openai-realtime",
                            "parameters": [
                                {
                                    "name": "temperature",
                                    "label": "Temperature",
                                    "kind": "number",
                                    "default": 0,
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            manifest_path = directory / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "clip-a",
                        "audio_path": "clips/clip-a.mp3",
                        "reference_transcript": "grüezi mitenand",
                        "source": "SwissDial",
                        "canton": "ZH",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            models = load_models(model_path)
            items = load_dataset_items(manifest_path)

        self.assertEqual(models[0].deployment, "deployment-a")
        self.assertEqual(models[0].transport, "azure-openai-realtime")
        self.assertEqual(models[0].parameters[0].default, 0)
        self.assertEqual(items[0].id, "clip-a")
        self.assertEqual(items[0].audio_path, directory / "clips" / "clip-a.mp3")
        self.assertEqual(items[0].metadata["canton"], "ZH")

    def test_missing_manifest_returns_empty_dataset(self) -> None:
        self.assertEqual(load_dataset_items(Path("does-not-exist.jsonl")), [])


if __name__ == "__main__":
    unittest.main()