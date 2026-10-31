"""Unit tests for RunManager step persistence and fault recovery."""

from pathlib import Path
import tempfile
import unittest

from utils.run_manager import RunManager


class TestRunManager(unittest.TestCase):
    def test_run_manager_initialization(self):
        """Verify RunManager directory creation and manifest file initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RunManager(run_id="test_run_123", base_dir=base_path)

            self.assertEqual(manager.run_id, "test_run_123")
            self.assertTrue(manager.run_dir.exists())
            self.assertTrue(manager.manifest_file.exists())

    def test_step_save_and_load_json(self):
        """Verify saving and reloading step outputs in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RunManager(run_id="test_run_json", base_dir=base_path)

            topics_data = {
                "topics": [
                    {"id": 1, "topic": "Quantum Computing Basics"},
                    {"id": 2, "topic": "Error Correction"},
                ]
            }

            # Step not yet saved
            self.assertFalse(manager.has_step("01_topics"))

            # Save step
            saved_path = manager.save_step("01_topics", topics_data)
            self.assertTrue(saved_path.exists())
            self.assertTrue(manager.has_step("01_topics"))

            # Load step back
            loaded = manager.load_step("01_topics")
            self.assertEqual(loaded, topics_data)
            self.assertIn("01_topics", manager.list_completed_steps())

    def test_step_save_and_load_text(self):
        """Verify saving and reloading markdown text step outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RunManager(run_id="test_run_txt", base_dir=base_path)

            script_text = "# Script Title\nScene 1: Introduction..."
            saved_path = manager.save_step("03_script", script_text, filename="script.md")

            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.name, "script.md")
            self.assertTrue(manager.has_step("03_script"))

            loaded = manager.load_step("03_script")
            self.assertEqual(loaded, script_text)

    def test_user_input_persistence(self):
        """Verify initial user parameters are saved to 00_user_input.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RunManager(run_id="test_run_input", base_dir=base_path)

            manager.save_user_input(
                idea="AI-driven urban gardening",
                tone="humorous",
                length="medium",
            )

            self.assertTrue(manager.has_step("user_input"))
            loaded = manager.load_step("user_input")
            self.assertEqual(loaded["idea"], "AI-driven urban gardening")
            self.assertEqual(loaded["tone"], "humorous")


if __name__ == "__main__":
    unittest.main()
