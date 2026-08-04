import unittest

from backend.app.metrics import normalize_transcript, score_transcript, word_match_rate


class TranscriptMetricTests(unittest.TestCase):
    def test_normalizes_case_punctuation_and_swiss_german_letters(self) -> None:
        self.assertEqual(normalize_transcript("  Grüezi, ZÄME!  "), "grüezi zäme")

    def test_calculates_word_and_character_error_rates(self) -> None:
        scores = score_transcript("hoi zäme", "hoi")

        self.assertEqual(scores.reference_word_count, 2)
        self.assertEqual(scores.hypothesis_word_count, 1)
        self.assertEqual(scores.word_error_rate, 0.5)
        self.assertGreater(scores.character_error_rate or 0, 0)

    def test_omits_rates_when_no_reference_transcript_exists(self) -> None:
        scores = score_transcript(None, "hoi zäme")

        self.assertIsNone(scores.word_error_rate)
        self.assertIsNone(scores.character_error_rate)
        self.assertEqual(scores.hypothesis_word_count, 2)

    def test_word_match_rate_is_bounded_and_unavailable_without_reference(self) -> None:
        self.assertEqual(word_match_rate(0), 1)
        self.assertEqual(word_match_rate(0.25), 0.75)
        self.assertEqual(word_match_rate(1.5), 0)
        self.assertIsNone(word_match_rate(None))


if __name__ == "__main__":
    unittest.main()