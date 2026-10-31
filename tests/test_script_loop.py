"""Tests for Writer Agent, Critic Agent, and LangGraph Script Loop."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.critic import critique_script
from agents.writer import write_script
from config import OUTPUTS_DIR
from pipeline.script_loop import run_script_loop

FAKE_RESEARCH = [
    {
        "topic": "OCR & Document Parsing for Handwritten Student Notes",
        "key_facts": [
            {
                "fact": "TrOCR and specialized CNN-LSTM models achieve over 92% character accuracy on messy student handwriting.",
                "source_url": "https://en.wikipedia.org/wiki/Optical_character_recognition",
            },
            {
                "fact": "Multimodal LLMs parse diagram layouts and formulas alongside handwritten text.",
                "source_url": "https://en.wikipedia.org/wiki/Multimodal_learning",
            },
        ],
        "tools_or_tech_mentioned": ["TrOCR", "PyMuPDF", "OpenCV"],
    }
]

SAMPLE_IDEA = "AI app that turns messy college notes into viral study quizzes"


class TestScriptLoop(unittest.TestCase):
    @patch("agents.writer.call_gemini")
    def test_writer_incorporates_research_and_feedback(self, mock_gemini):
        """Verify write_script integrates research facts and addresses critic feedback."""
        mock_gemini.return_value = "# Scene 1\nHost: Turn messy notes into quiz gold with TrOCR!"

        # Initial write
        draft = write_script(
            idea=SAMPLE_IDEA,
            tone="humorous",
            length="short",
            research=FAKE_RESEARCH,
        )
        self.assertIn("TrOCR", draft)
        self.assertTrue(mock_gemini.called)

        # Revision with feedback
        feedback = ["Punch up the hook in Scene 1", "Mention PyMuPDF explicitly"]
        revised = write_script(
            idea=SAMPLE_IDEA,
            tone="humorous",
            length="short",
            research=FAKE_RESEARCH,
            feedback=feedback,
        )
        self.assertIsInstance(revised, str)
        # Check that feedback instructions were included in the prompt
        called_prompt = mock_gemini.call_args[1]["prompt"]
        self.assertIn("CRITICAL EDITORIAL REVISION REQUIRED", called_prompt)
        self.assertIn("Punch up the hook in Scene 1", called_prompt)

    @patch("agents.critic.call_groq")
    def test_critic_calculates_python_average_and_checks_rubric(self, mock_groq):
        """Verify critic recomputes the average in Python rather than trusting LLM math."""
        # Model returns deliberate scores and its own (untrusted) average
        mock_groq.return_value = {
            "scores": {
                "hook": 8.0,
                "clarity": 9.0,
                "tone_match": 9.0,
                "accuracy": 10.0,
                "length_fit": 8.0,
                "originality": 8.0,
            },
            # Note: sum is 52.0 / 6 = 8.67
            "average": 99.9,  # Deliberately bogus average to ensure Python recomputes it
            "feedback": ["Great comedic timing in intro", "Shorten scene 3 transition"],
            "weak_lines": ["'This is an app.' -> too bland, make it snappier."],
        }

        critique = critique_script(
            script="Sample script text here...",
            tone="humorous",
            research=FAKE_RESEARCH,
        )

        scores = critique["scores"]
        self.assertEqual(len(scores), 6)
        self.assertEqual(scores["hook"], 8.0)
        self.assertEqual(scores["accuracy"], 10.0)

        # Verify average was computed in Python: 52 / 6 = 8.67
        self.assertEqual(critique["average"], 8.67)
        self.assertNotEqual(critique["average"], 99.9)
        self.assertIn("feedback", critique)

    @patch("pipeline.script_loop.write_script")
    @patch("pipeline.script_loop.critique_script")
    def test_langgraph_loop_passes_on_high_score(self, mock_critique, mock_writer):
        """Verify LangGraph script loop terminates on round 1 when score meets pass criteria."""
        mock_writer.return_value = "# Perfect Script\nHost: Welcome to quiz automation."
        # Passing scores: average >= 9.0 and all >= 8.0
        mock_critique.return_value = {
            "scores": {
                "hook": 9.5,
                "clarity": 9.0,
                "tone_match": 9.5,
                "accuracy": 9.0,
                "length_fit": 9.0,
                "originality": 9.0,
            },
            "average": 9.17,
            "feedback": ["Outstanding draft, ready for production."],
            "weak_lines": [],
        }

        run_id = "test_loop_pass_run"
        result = run_script_loop(
            idea=SAMPLE_IDEA,
            tone="humorous",
            length="short",
            research=FAKE_RESEARCH,
            run_id=run_id,
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["round"], 1)

        # Verify files created in drafts and final_script.md
        draft_file = OUTPUTS_DIR / run_id / "drafts" / "draft_1.md"
        critique_file = OUTPUTS_DIR / run_id / "drafts" / "critique_1.json"
        final_file = OUTPUTS_DIR / run_id / "final_script.md"

        self.assertTrue(draft_file.exists())
        self.assertTrue(critique_file.exists())
        self.assertTrue(final_file.exists())

    @patch("pipeline.script_loop.write_script")
    @patch("pipeline.script_loop.critique_script")
    def test_langgraph_loop_retries_and_exhausts_to_best_effort(
        self, mock_critique, mock_writer
    ):
        """Verify LangGraph retries up to 3 times and selects highest-scoring draft on exhaustion."""
        mock_writer.side_effect = [
            "# Draft 1: Mediocre",
            "# Draft 2: Better",
            "# Draft 3: Peak Draft",
            "# Draft 4: Slight Dip",
        ]

        # Simulating 4 rounds that never hit the 9.0 threshold
        # Max retries is 3, so round 1 + 3 retries = 4 total rounds
        mock_critique.side_effect = [
            {"scores": {"hook": 7.0, "clarity": 7.0, "tone_match": 7.0, "accuracy": 7.0, "length_fit": 7.0, "originality": 7.0}, "average": 7.0, "feedback": ["Fix hook"]},
            {"scores": {"hook": 8.0, "clarity": 8.0, "tone_match": 8.0, "accuracy": 8.0, "length_fit": 8.0, "originality": 8.0}, "average": 8.0, "feedback": ["Add energy"]},
            {"scores": {"hook": 8.8, "clarity": 8.8, "tone_match": 8.8, "accuracy": 8.8, "length_fit": 8.8, "originality": 8.8}, "average": 8.8, "feedback": ["Almost there"]},  # Peak (8.8)
            {"scores": {"hook": 8.2, "clarity": 8.2, "tone_match": 8.2, "accuracy": 8.2, "length_fit": 8.2, "originality": 8.2}, "average": 8.2, "feedback": ["Over-edited"]},
        ]

        run_id = "test_loop_exhausted_run"
        result = run_script_loop(
            idea=SAMPLE_IDEA,
            tone="humorous",
            length="short",
            research=FAKE_RESEARCH,
            run_id=run_id,
        )

        self.assertEqual(result["status"], "best_effort")
        self.assertEqual(result["round"], 4)

        # Ensure highest scoring version (Draft 3 with score 8.8) was chosen
        self.assertEqual(result["average"], 8.8)
        self.assertEqual(result["script"], "# Draft 3: Peak Draft")

        # Verify all 4 drafts and critiques were saved
        drafts_dir = OUTPUTS_DIR / run_id / "drafts"
        for r in range(1, 5):
            self.assertTrue((drafts_dir / f"draft_{r}.md").exists())
            self.assertTrue((drafts_dir / f"critique_{r}.json").exists())

        # Verify final script matches peak draft
        final_script = (OUTPUTS_DIR / run_id / "final_script.md").read_text(encoding="utf-8")
        self.assertEqual(final_script, "# Draft 3: Peak Draft")


def run_standalone_demo():
    """Demonstrate the script loop with fake research input."""
    print("=" * 65)
    print("Demonstrating LangGraph Script Loop with Fake Research")
    print(f"Idea: '{SAMPLE_IDEA}'")
    print("=" * 65)

    demo_run_id = "demo_script_loop_run"

    # Simulate: Round 1 scores 8.2 (needs revision), Round 2 scores 9.2 (passes!)
    with patch("pipeline.script_loop.write_script") as mock_writer, \
         patch("pipeline.script_loop.critique_script") as mock_critique:

        mock_writer.side_effect = [
            "# Draft 1: The College Quiz Hack\nScene 1: Staring at 50 pages of handwritten notes...",
            "# Draft 2 (Revised): The College Quiz Revolution\nScene 1: [Visual: High-energy montage of highlighter markers] Host: What if your notes studied YOU?",
        ]

        mock_critique.side_effect = [
            {
                "scores": {"hook": 7.8, "clarity": 8.5, "tone_match": 8.0, "accuracy": 8.5, "length_fit": 8.5, "originality": 8.0},
                "average": 8.22,
                "feedback": [
                    "Punch up the opening hook; make the relatable agony of exam cramming more vivid.",
                    "Highlight TrOCR accuracy explicitly in narration.",
                ],
                "weak_lines": ["'Notes are hard to read' -> make it sharper."],
            },
            {
                "scores": {"hook": 9.5, "clarity": 9.2, "tone_match": 9.3, "accuracy": 9.5, "length_fit": 9.0, "originality": 9.0},
                "average": 9.25,
                "feedback": ["Outstanding comedic energy and precise factual citations."],
                "weak_lines": [],
            },
        ]

        final_state = run_script_loop(
            idea=SAMPLE_IDEA,
            tone="humorous",
            length="short",
            research=FAKE_RESEARCH,
            run_id=demo_run_id,
        )

    print("\n" + "=" * 65)
    print("Workflow Execution Summary:")
    print(f"Status: {final_state['status']}")
    print(f"Final Score Average: {final_state['average']:.2f}/10")
    print("Final Script Preview:")
    print("-" * 65)
    print(final_state["script"][:300] + "...")
    print("=" * 65)


if __name__ == "__main__":
    run_standalone_demo()
