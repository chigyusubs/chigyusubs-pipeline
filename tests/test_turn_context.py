from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chigyusubs.turn_context import discover_words_json_path


class DiscoverWordsJsonPathTests(unittest.TestCase):
    def test_discovers_run_id_ctc_words_for_reflow_vtt_in_translation_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode_dir = root / "samples" / "episodes" / "demo_ep"
            transcription_dir = episode_dir / "transcription"
            translation_dir = episode_dir / "translation"
            transcription_dir.mkdir(parents=True)
            translation_dir.mkdir(parents=True)

            input_path = translation_dir / "rba19229d_en.vtt"
            input_path.write_text("WEBVTT\n", encoding="utf-8")

            reflow_path = transcription_dir / "rba19229d_reflow.vtt"
            reflow_path.write_text("WEBVTT\n", encoding="utf-8")

            words_path = transcription_dir / "rba19229d_ctc_words.json"
            words_path.write_text("[]\n", encoding="utf-8")

            discovered = discover_words_json_path(
                input_path=reflow_path,
                words_path="",
            )

            self.assertEqual(discovered, words_path)


if __name__ == "__main__":
    unittest.main()
