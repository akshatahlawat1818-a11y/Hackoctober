"""Tests for Planner Agent and Pipeline Approval Step."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.planner import plan_topics
from config import API_CONFIG, OUTPUTS_DIR
from pipeline.approval import approve_topics, save_topics_to_disk

SAMPLE_QUIZ_IDEA = "AI app that turns my college notes into quizzes"

MOCK_PLANNER_RESPONSE = {
    "topics": [
        {
            "title": "OCR & Document Parsing for Handwritten Student Notes",
            "why_it_matters": "Raw student notes (handwritten and scanned PDFs) require accurate text and layout extraction.",
            "search_queries": [
                "best open source ocr for handwriting 2025",
                "multimodal llm document parser python",
            ],
        },
        {
            "title": "Automated Question Generation Algorithms using LLMs",
            "why_it_matters": "Transforming unstructured study material into rigorous multiple choice and flashcard questions.",
            "search_queries": [
                "few-shot prompt engineering question generation llm",
                "bloom taxonomy quiz generator ai",
            ],
        },
        {
            "title": "RAG Architecture for Long Lecture Notes and Textbooks",
            "why_it_matters": "Need chunking and vector retrieval so questions cite exact source pages in the notes.",
            "search_queries": [
                "rag chunking strategies for college notes",
                "dense passage retrieval student flashcards",
            ],
        },
        {
            "title": "Spaced Repetition & Cognitive Science (Leitner / SM-2)",
            "why_it_matters": "Increases retention by scheduling quiz questions at optimal memory decay intervals.",
            "search_queries": [
                "sm2 algorithm python implementation",
                "spaced repetition active recall quiz app architecture",
            ],
        },
        {
            "title": "Multi-Modal Parsing for Math Equations and Science Diagrams",
            "why_it_matters": "STEM notes frequently include charts, formulas, and diagrams that standard text parsers miss.",
            "search_queries": [
                "latex math equation extraction from notes ai",
                "diagram understanding vision models quiz",
            ],
        },
        {
            "title": "Gamification and Student Retention Mechanics",
            "why_it_matters": "Streak mechanics and adaptive difficulty maintain daily study habits for college users.",
            "search_queries": [
                "edtech gamification study streak case studies",
                "adaptive testing difficulty algorithms edtech",
            ],
        },
    ]
}


class TestPlanner(unittest.TestCase):
    @patch("agents.planner.call_gemini")
    def test_plan_topics_schema_and_counts(self, mock_call_gemini):
        """Verify plan_topics returns 5-8 topics adhering to required schema."""
        mock_call_gemini.return_value = MOCK_PLANNER_RESPONSE

        topics = plan_topics(
            idea=SAMPLE_QUIZ_IDEA,
            tone="humorous",
            length="medium",
        )

        # Must return between 5 and 8 topics
        self.assertGreaterEqual(len(topics), 5)
        self.assertLessEqual(len(topics), 8)

        # Verify topic structure
        for topic in topics:
            self.assertIn("title", topic)
            self.assertIn("why_it_matters", topic)
            self.assertIn("search_queries", topic)
            self.assertIsInstance(topic["title"], str)
            self.assertIsInstance(topic["why_it_matters"], str)
            self.assertIsInstance(topic["search_queries"], list)
            self.assertGreaterEqual(len(topic["search_queries"]), 2)
            self.assertLessEqual(len(topic["search_queries"]), 3)

    def test_approval_workflow_remove_add_and_accept(self):
        """Verify approve_topics commands for remove, add, and ok acceptance."""
        initial_topics = list(MOCK_PLANNER_RESPONSE["topics"])
        run_id = "test_run_approval_123"

        # Simulate user typing: "remove 2", "add Export to Anki & Quizlet", "ok"
        command_inputs = iter(["remove 2", "add Export to Anki & Quizlet", "ok"])

        approved = approve_topics(
            topics=initial_topics,
            idea=SAMPLE_QUIZ_IDEA,
            run_id=run_id,
            input_func=lambda _: next(command_inputs),
        )

        # Initial was 6 topics, removed 1, added 1 -> total 6
        self.assertEqual(len(approved), 6)
        # Verify the newly added topic is present
        titles = [t["title"] for t in approved]
        self.assertIn("Export to Anki & Quizlet", titles)
        self.assertNotIn("Automated Question Generation Algorithms using LLMs", titles)

        # Verify saved file in outputs/<run_id>/topics.json
        expected_file = OUTPUTS_DIR / run_id / "topics.json"
        self.assertTrue(expected_file.exists())

        with open(expected_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        self.assertEqual(len(saved_data), len(approved))
        self.assertEqual(saved_data[-1]["title"], "Export to Anki & Quizlet")


def run_standalone_demo():
    """Interactive / standalone demo for Planner and Approval step."""
    print("=" * 65)
    print("Testing Planner with Idea:")
    print(f"'{SAMPLE_QUIZ_IDEA}'")
    print("=" * 65)

    has_live_key = bool(
        (API_CONFIG.gemini_api_key and "your_gemini" not in API_CONFIG.gemini_api_key)
        or (os.getenv("GEMINI_API_KEY") and "your_gemini" not in os.getenv("GEMINI_API_KEY", ""))
    )

    if has_live_key:
        print("\n--- Calling live Gemini model for topic planning ---")
        try:
            topics = plan_topics(idea=SAMPLE_QUIZ_IDEA, tone="humorous", length="medium")
        except Exception as e:
            print(f"Live Gemini call failed ({e}). Falling back to mock topics.")
            topics = MOCK_PLANNER_RESPONSE["topics"]
    else:
        print("\n--- Using Mocked Planner Response (No GEMINI_API_KEY set) ---")
        topics = MOCK_PLANNER_RESPONSE["topics"]

    print(f"\nGenerated {len(topics)} topics successfully.")

    # Demonstrate automated approval simulation
    demo_run_id = "demo_college_notes_run"
    print("\n--- Simulating Approval Commands: 'remove 1', 'add Flashcards Export', 'ok' ---")
    simulated_inputs = iter(["remove 1", "add Flashcards Export", "ok"])

    final_topics = approve_topics(
        topics=topics,
        idea=SAMPLE_QUIZ_IDEA,
        tone="humorous",
        length="medium",
        run_id=demo_run_id,
        input_func=lambda prompt: next(simulated_inputs),
    )

    saved_path = OUTPUTS_DIR / demo_run_id / "topics.json"
    print(f"\nFinal approved topics count: {len(final_topics)}")
    print(f"Saved file verified at: {saved_path}")
    print("\nPreview of saved topics.json:")
    with open(saved_path, "r", encoding="utf-8") as f:
        print(f.read()[:500] + "\n...")


if __name__ == "__main__":
    run_standalone_demo()
