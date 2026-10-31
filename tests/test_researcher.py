"""Tests for Researcher Agent and Source URL Verification."""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.researcher import research_all, research_topic
from config import API_CONFIG, OUTPUTS_DIR

SAMPLE_TOPIC = {
    "title": "Spaced Repetition & Cognitive Science (Leitner / SM-2)",
    "why_it_matters": "Increases retention by scheduling quiz questions at optimal memory decay intervals.",
    "search_queries": [
        "sm2 algorithm python implementation",
        "spaced repetition active recall quiz app architecture",
    ],
}


class TestResearcher(unittest.TestCase):
    @patch("agents.researcher.fetch_duckduckgo_results")
    @patch("agents.researcher.fetch_wikipedia_summary")
    @patch("agents.researcher.call_gemini")
    def test_research_topic_drops_hallucinated_urls(
        self, mock_gemini, mock_wiki, mock_ddg
    ):
        """Verify that any facts citing unverified or hallucinated URLs are pruned."""
        real_url_1 = "https://en.wikipedia.org/wiki/Spaced_repetition"
        real_url_2 = "https://supermemo.guru/wiki/SM-2"
        fake_url = "https://hallucinated-source.com/made-up-study"

        # Mock search results
        mock_ddg.return_value = [
            {"title": "SM-2 Algorithm", "href": real_url_2, "body": "Details of the SM-2 spaced repetition algorithm."},
        ]
        mock_wiki.return_value = {
            "title": "Wikipedia: Spaced repetition",
            "href": real_url_1,
            "body": "Spaced repetition is an evidence-based learning technique...",
        }

        # Mock Gemini returning 2 valid facts and 1 fabricated fact
        mock_gemini.return_value = {
            "topic": SAMPLE_TOPIC["title"],
            "key_facts": [
                {
                    "fact": "Spaced repetition spaces learning intervals to combat the forgetting curve.",
                    "source_url": real_url_1,
                },
                {
                    "fact": "SM-2 computes ease factors based on grade responses 0 through 5.",
                    "source_url": real_url_2,
                },
                {
                    "fact": "Invented study claiming 999% retention improvement.",
                    "source_url": fake_url,  # Must be dropped!
                },
            ],
            "tools_or_tech_mentioned": ["Anki", "SuperMemo", "Anki-sync"],
        }

        result = research_topic(SAMPLE_TOPIC)

        self.assertEqual(result["topic"], SAMPLE_TOPIC["title"])
        self.assertEqual(len(result["key_facts"]), 2)

        # Ensure all facts cite ONLY real URLs
        for item in result["key_facts"]:
            self.assertIn(item["source_url"], [real_url_1, real_url_2])
            self.assertNotEqual(item["source_url"], fake_url)

        self.assertIn("Anki", result["tools_or_tech_mentioned"])

    @patch("agents.researcher.research_topic")
    def test_research_all_persistence(self, mock_research_topic):
        """Verify sequential research execution and per-topic JSON file persistence."""
        mock_summary = {
            "topic": SAMPLE_TOPIC["title"],
            "key_facts": [
                {
                    "fact": "Active recall and spaced repetition improve long term retention.",
                    "source_url": "https://en.wikipedia.org/wiki/Spaced_repetition",
                }
            ],
            "tools_or_tech_mentioned": ["Anki", "FSRS"],
        }
        mock_research_topic.return_value = mock_summary

        run_id = "test_run_research_persistence"
        results = research_all([SAMPLE_TOPIC], run_id=run_id, pause_seconds=0.01)

        self.assertEqual(len(results), 1)

        # Verify research_1.json exists in outputs/<run_id>/
        saved_file = OUTPUTS_DIR / run_id / "research_1.json"
        self.assertTrue(saved_file.exists())

        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["topic"], SAMPLE_TOPIC["title"])
        self.assertEqual(len(data["key_facts"]), 1)
        self.assertEqual(data["key_facts"][0]["source_url"], "https://en.wikipedia.org/wiki/Spaced_repetition")


def run_standalone_demo():
    """Run research on one topic and print the result."""
    print("=" * 65)
    print("Testing Researcher Agent on One Topic:")
    print(f"'{SAMPLE_TOPIC['title']}'")
    print("=" * 65)

    has_live_key = bool(
        (API_CONFIG.gemini_api_key and "your_gemini" not in API_CONFIG.gemini_api_key)
        or (os.getenv("GEMINI_API_KEY") and "your_gemini" not in os.getenv("GEMINI_API_KEY", ""))
    )

    demo_run_id = "demo_research_one_topic"

    if has_live_key:
        print("\n--- Running live DuckDuckGo + Wikipedia + Gemini synthesis ---")
        try:
            results = research_all([SAMPLE_TOPIC], run_id=demo_run_id, pause_seconds=1.0)
        except Exception as e:
            print(f"Live call encountered error: {e}. Running with mock.")
            results = None
    else:
        results = None

    if not results:
        print("\n--- Running with Mocked Gemini (No GEMINI_API_KEY configured) ---")
        real_wiki_url = "https://en.wikipedia.org/wiki/Spaced_repetition"
        real_ddg_url = "https://en.wikipedia.org/wiki/SuperMemo"

        with patch("agents.researcher.fetch_duckduckgo_results") as mock_ddg, \
             patch("agents.researcher.fetch_wikipedia_summary") as mock_wiki, \
             patch("agents.researcher.call_gemini") as mock_gemini:

            mock_ddg.return_value = [
                {"title": "SuperMemo SM2", "href": real_ddg_url, "body": "SuperMemo is a learning system based on computing optimal repetitions."},
            ]
            mock_wiki.return_value = {
                "title": "Wikipedia: Spaced repetition",
                "href": real_wiki_url,
                "body": "Spaced repetition is an evidence-based learning technique that incorporates increasing intervals of time between subsequent review.",
            }
            mock_gemini.return_value = {
                "topic": SAMPLE_TOPIC["title"],
                "key_facts": [
                    {
                        "fact": "Spaced repetition systematically increases intervals between reviews according to memory decay curves.",
                        "source_url": real_wiki_url,
                    },
                    {
                        "fact": "The SM-2 algorithm calculates ease factors based on graded student feedback.",
                        "source_url": real_ddg_url,
                    },
                    {
                        "fact": "Fake fact with fabricated link that should be pruned.",
                        "source_url": "https://invented-site.org/hallucinated-paper",
                    },
                ],
                "tools_or_tech_mentioned": ["Anki", "SuperMemo", "FSRS-4.5"],
            }

            results = research_all([SAMPLE_TOPIC], run_id=demo_run_id, pause_seconds=0.1)

    saved_file = OUTPUTS_DIR / demo_run_id / "research_1.json"
    print("\n" + "=" * 65)
    print(f"Result verified in: {saved_file}")
    print("=" * 65)
    with open(saved_file, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    run_standalone_demo()
