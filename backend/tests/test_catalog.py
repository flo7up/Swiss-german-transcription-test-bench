import json
import tempfile
import unittest
from pathlib import Path

from backend.app.catalog import load_dataset_items, load_dialect_atlas, load_models


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

    def test_dialect_atlas_derives_speaker_shares_from_canton_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "dialects.json"
            path.write_text(
                json.dumps(
                    {
                        "cantons": {
                            "ZH": {"name": "Zurich", "population": 1000, "german_share": 0.8},
                            "SZ": {"name": "Schwyz", "population": 500, "german_share": 1.0},
                            "LU": {"name": "Lucerne", "population": 500, "german_share": 0.6},
                            "GE": {"name": "Geneva", "population": 1000, "german_share": 0.0},
                        },
                        "dialects": [
                            {"code": "ZH", "name": "Zurich German", "region_cantons": ["ZH"]},
                            {"code": "LU", "name": "Lucerne German", "region_cantons": ["LU", "SZ"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            atlas = load_dialect_atlas(path)

        self.assertEqual(atlas["total_population"], 3000)
        self.assertEqual(atlas["total_german_speakers"], 1600)
        self.assertEqual([dialect["code"] for dialect in atlas["dialects"]], ["ZH", "LU"])
        lucerne = atlas["dialects"][1]
        self.assertEqual(lucerne["core_speakers"], 300)
        self.assertEqual(lucerne["region_speakers"], 800)
        self.assertAlmostEqual(lucerne["share_of_german_speakers"], 0.5)
        self.assertAlmostEqual(atlas["coverage_share"], 1.0)

    def test_repository_dialect_atlas_is_consistent(self) -> None:
        atlas = load_dialect_atlas(Path(__file__).resolve().parents[2] / "config" / "dialects.json")
        codes = {dialect["code"] for dialect in atlas["dialects"]}

        self.assertEqual(codes, {"AG", "BE", "BS", "GR", "LU", "SG", "VS", "ZH"})
        self.assertEqual(len(atlas["cantons"]), 26)
        for dialect in atlas["dialects"]:
            self.assertTrue(set(dialect["region_cantons"]) <= set(atlas["cantons"]))
            self.assertTrue(dialect["features"])
        self.assertGreater(atlas["coverage_share"], 0.8)
        self.assertLess(atlas["coverage_share"], 1)

    def test_missing_dialect_atlas_returns_empty_payload(self) -> None:
        self.assertEqual(load_dialect_atlas(Path("does-not-exist.json"))["dialects"], [])


if __name__ == "__main__":
    unittest.main()