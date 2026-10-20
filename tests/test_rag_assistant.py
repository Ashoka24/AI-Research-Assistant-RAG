import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag_assistant import ResearchAssistant, chunk_text


class TestRetriever(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(chunk_text("  "), [])

    def test_chunking_splits_long_text(self):
        parts = chunk_text(" ".join(["research"] * 300), 120, 20)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 120 for part in parts))

    def test_relevant_source_is_ranked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "methods.md").write_text("TF-IDF ranks passages using word overlap.", encoding="utf-8")
            (root / "other.txt").write_text("Tomatoes need sun and water.", encoding="utf-8")
            assistant = ResearchAssistant(root, chunk_size=400, overlap=30)
            results = assistant.retrieve("How does TF-IDF rank passages?", top_k=2)
            self.assertTrue(results)
            self.assertEqual(results[0].source, "methods.md")

    def test_missing_directory_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            assistant = ResearchAssistant(Path(folder) / "missing")
            self.assertEqual(assistant.retrieve("question"), [])


if __name__ == "__main__":
    unittest.main()
