import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from scripts.check_publication import publication_issues, screenshot_issues


class PublicationTests(unittest.TestCase):
    def test_offline_checks_do_not_load_local_environment_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PYTHON_DOTENV_DISABLED": "1"}, clear=True
        ):
            dotenv = Path(directory) / ".env"
            dotenv.write_text("BENCHMARK_TEST_DOTENV_MARKER=should-stay-unloaded\n", encoding="utf-8")
            self.assertFalse(load_dotenv(dotenv))
            self.assertNotIn("BENCHMARK_TEST_DOTENV_MARKER", os.environ)

    def test_rejects_local_artifacts_even_if_force_added_to_git(self):
        for name in (
            ".env", ".env.production", ".runtime/benchmark.sqlite3", ".venv/example.py",
            "frontend/node_modules/example.js", ".pytest_cache/README.md",
            "data/swissdial/manifest.jsonl", "data/custom/examples.jsonl",
            "clips/recording.wav", "sample.zip", "history.sqlite3-wal", "credentials.key",
            "LOCAL\\Recording.MP3",
        ):
            with self.subTest(name=name):
                self.assertEqual(len(publication_issues([name])), 1)

    def test_allows_examples_code_and_documentation_images(self):
        self.assertEqual(
            publication_issues([
                ".env.example", "data/swissdial/manifest.example.jsonl",
                "backend/tests/test_api.py", "docs/images/results-table.png",
                ".github/workflows/ci.yml", "requirements-dev.txt",
            ]),
            [],
        )

    def test_detects_missing_and_invalid_screenshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs" / "images").mkdir(parents=True)
            (root / "README.md").write_text(
                '![Missing](docs/images/missing.png)\n<img src="docs/images/invalid.png">',
                encoding="utf-8",
            )
            (root / "docs" / "images" / "invalid.png").write_text("not an image", encoding="utf-8")
            issues = screenshot_issues(root)
        self.assertEqual(len(issues), 2)
        self.assertTrue(any("Missing README screenshot" in issue for issue in issues))
        self.assertTrue(any("Invalid PNG screenshot" in issue for issue in issues))
