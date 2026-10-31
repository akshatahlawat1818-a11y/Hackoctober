"""Tests for main.py CLI runner and workflow orchestration."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from main import parse_arguments, run_pipeline


class TestMainCLI(unittest.TestCase):
    def test_parse_arguments_defaults(self):
        """Verify default CLI argument values."""
        with patch.object(sys, "argv", ["main.py"]):
            args = parse_arguments()
            self.assertIsNone(args.idea)
            self.assertEqual(args.tone, "serious")
            self.assertEqual(args.length, "60 seconds")
            self.assertIsNone(args.resume)
            self.assertFalse(args.no_video)
            self.assertFalse(args.dry_run)
            self.assertEqual(args.mode, "draft")

    def test_parse_arguments_custom(self):
        """Verify custom CLI flag parsing for idea, tone, length, and options."""
        test_args = [
            "main.py",
            "--idea", "AI quiz generator",
            "--tone", "humorous",
            "--length", "90 seconds",
            "--resume", "run_12345",
            "--no-video",
            "--dry-run",
            "--mode", "final",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_arguments()
            self.assertEqual(args.idea, "AI quiz generator")
            self.assertEqual(args.tone, "humorous")
            self.assertEqual(args.length, "90 seconds")
            self.assertEqual(args.resume, "run_12345")
            self.assertTrue(args.no_video)
            self.assertTrue(args.dry_run)
            self.assertEqual(args.mode, "final")

    def test_resume_skips_existing_steps(self):
        """Verify that --resume skips already generated step artifacts on disk."""
        run_id = "test_main_resume_run"
        run_dir = config.OUTPUTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Pre-populate step files to test skip logic
        user_input_file = run_dir / "00_user_input.json"
        user_input_file.write_text(
            json.dumps({"idea": "Cached Idea", "tone": "serious", "length": "60 seconds"}),
            encoding="utf-8",
        )

        topics_file = run_dir / "topics.json"
        topics_file.write_text(
            json.dumps([{"title": "Topic 1", "why_it_matters": "Why", "search_queries": ["query 1"]}]),
            encoding="utf-8",
        )

        research_file = run_dir / "research.json"
        research_file.write_text(
            json.dumps([{"topic": "Topic 1", "key_facts": [], "tools_or_tech_mentioned": []}]),
            encoding="utf-8",
        )

        script_file = run_dir / "final_script.md"
        script_file.write_text("# Cached Script\n\nThis is a cached script.", encoding="utf-8")

        tools_file = run_dir / "tools.json"
        tools_file.write_text(
            json.dumps([{"name": "Tool A", "purpose": "Testing", "cost": "free", "setup_steps": [], "link": "https://example.com"}]),
            encoding="utf-8",
        )

        storyboard_file = run_dir / "storyboard.json"
        storyboard_file.write_text(
            json.dumps({"voiceover_text": "Sample voiceover text.", "scenes": []}),
            encoding="utf-8",
        )

        # With all step files existing and --no-video, run_pipeline should execute without calling agents
        result = run_pipeline(
            resume_run_id=run_id,
            no_video=True,
        )

        self.assertEqual(result["run_id"], run_id)
        self.assertTrue(Path(result["script_file"]).exists())
        self.assertTrue(Path(result["tools_file"]).exists())
        self.assertTrue(Path(result["storyboard_file"]).exists())
        self.assertIsNone(result["promo_file"])


if __name__ == "__main__":
    unittest.main()
