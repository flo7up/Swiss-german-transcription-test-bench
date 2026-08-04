import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import zipfile


IMPORTER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "import_swissdial_archive.py"
SPEC = importlib.util.spec_from_file_location("swissdial_importer", IMPORTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load the SwissDial importer.")
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


class SwissDialImporterTests(unittest.TestCase):
    def test_imports_balanced_official_sample_with_dialect_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            dataset_dir = root / "data1.1"
            sentences = []
            for sentence_id in range(3):
                sentences.append(
                    {
                        "id": sentence_id,
                        "de": f"Standard German {sentence_id}",
                        "ch_ag": f"Aargau {sentence_id}",
                        "ch_be": f"Bern {sentence_id}",
                        "thema": "demo",
                    }
                )
            dataset_dir.mkdir()
            (dataset_dir / "sentences_ch_de_numerics.json").write_text(
                json.dumps(sentences), encoding="utf-8"
            )
            for dialect in ("ag", "be"):
                (dataset_dir / dialect).mkdir()
                for sentence_id in range(3):
                    (dataset_dir / dialect / f"ch_{dialect}_{sentence_id:04d}.wav").write_bytes(b"audio")

            output_dir = root / "dataset"
            count = IMPORTER.import_official_dataset(
                dataset_dir,
                output_dir,
                catalog_size=4,
                dialects=["ag", "be"],
                seed=7,
            )
            records = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 4)
        self.assertEqual([record["dialect"] for record in records], ["AG", "BE", "AG", "BE"])
        self.assertTrue(all(record["reference_transcript"].startswith(("Aargau", "Bern")) for record in records))
        self.assertTrue(all(record["standard_german_transcript"].startswith("Standard German") for record in records))

    def test_imports_flat_archive_clips_referenced_with_speaker_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "sample/export.tsv",
                    "clip_id\tclip_path\tsentence\tclip_is_valid\tclient_id\tcanton\n"
                    "clip-1\tspeaker-a/clip-1.mp3\tgrüezi mitenand\ttrue\tspeaker-a\tZH\n",
                )
                archive.writestr("sample/clips/clip-1.mp3", b"audio")

            output_dir = root / "dataset"
            count = IMPORTER.import_archive(archive_path, output_dir)
            record = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(record["audio_path"], "clips/clip-1.mp3")
        self.assertEqual(record["reference_transcript"], "grüezi mitenand")
        self.assertEqual(record["canton"], "ZH")


if __name__ == "__main__":
    unittest.main()