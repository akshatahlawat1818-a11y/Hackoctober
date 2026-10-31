"""Tests for Tools Agent and Promo Agent."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.promo_agent import build_storyboard
from agents.tools_agent import build_tools_list
from config import OUTPUTS_DIR

SAMPLE_IDEA = "AI app that turns my college notes into quizzes"
SAMPLE_SCRIPT = """# College Note Quizzer
## Scene 1: The Study Struggle
Host: Tired of staring at 60 pages of organic chemistry notes?
Let's turn that chaos into interactive quizzes using TrOCR and LangChain.
## Scene 2: The Magic
Visual: Document scanned, questions created in seconds.
## Scene 3: The Win
Host: Study faster, remember longer.
"""

SAMPLE_RESEARCH = [
    {
        "topic": "OCR & Document Parsing for Handwritten Student Notes",
        "key_facts": [
            {
                "fact": "TrOCR achieves state of the art accuracy on handwritten text.",
                "source_url": "https://en.wikipedia.org/wiki/Optical_character_recognition",
            }
        ],
        "tools_or_tech_mentioned": ["TrOCR", "Tesseract", "PyMuPDF"],
    }
]


class TestToolsAndPromoAgents(unittest.TestCase):
    @patch("agents.tools_agent.call_gemini")
    def test_build_tools_list_schema_and_saving(self, mock_gemini):
        """Verify build_tools_list generates valid tool schemas and persists tools.json."""
        mock_gemini.return_value = {
            "tools": [
                {
                    "name": "PyMuPDF",
                    "purpose": "Extract text, layout, and images from student PDF lecture notes.",
                    "cost": "free",
                    "setup_steps": ["pip install PyMuPDF", "import fitz"],
                    "link": "https://github.com/pymupdf/PyMuPDF",
                },
                {
                    "name": "TrOCR",
                    "purpose": "Transform handwriting into clean digital text.",
                    "cost": "free",
                    "setup_steps": ["pip install transformers", "from transformers import TrOCRProcessor"],
                    "link": "https://huggingface.co/microsoft/trocr-base-handwritten",
                },
                {
                    "name": "Google Gemini 3.8 Flash",
                    "purpose": "Generate active recall questions and answer explanations.",
                    "cost": "freemium",
                    "setup_steps": ["pip install google-genai", "export GEMINI_API_KEY=..."],
                    "link": "https://ai.google.dev/",
                },
            ]
        }

        run_id = "test_run_tools"
        tools = build_tools_list(
            idea=SAMPLE_IDEA,
            script=SAMPLE_SCRIPT,
            research=SAMPLE_RESEARCH,
            run_id=run_id,
        )

        self.assertEqual(len(tools), 3)
        for t in tools:
            self.assertIn("name", t)
            self.assertIn("purpose", t)
            self.assertIn("cost", t)
            self.assertIn(t["cost"], ["free", "paid", "freemium"])
            self.assertIn("setup_steps", t)
            self.assertIsInstance(t["setup_steps"], list)
            self.assertIn("link", t)
            self.assertTrue(t["link"].startswith("http"))

        # Verify saved file outputs/<run_id>/tools.json
        tools_path = OUTPUTS_DIR / run_id / "tools.json"
        self.assertTrue(tools_path.exists())
        with open(tools_path, "r", encoding="utf-8") as f:
            saved_tools = json.load(f)
        self.assertEqual(len(saved_tools), 3)

    @patch("agents.promo_agent.call_gemini")
    def test_build_storyboard_schema_and_saving(self, mock_gemini):
        """Verify build_storyboard produces 4 scenes, valid voiceover, and persists storyboard.json."""
        mock_gemini.return_value = {
            "voiceover_text": (
                "Drowning in stacks of messy college lecture notes before final exams? "
                "Watch your handwritten pages transform into interactive flashcards and smart recall quizzes in seconds. "
                "Study half the time, retain twice as much, and ace every test with confidence. "
                "Turn your notes into top grades right now."
            ),
            "scenes": [
                {
                    "scene_no": 1,
                    "visual_prompt": "Cinematic camera push-in on an exhausted college student staring at messy notebooks under warm desk lamp glow, shallow depth of field, photorealistic lighting, no text.",
                    "caption": "Drowning in messy study notes?",
                },
                {
                    "scene_no": 2,
                    "visual_prompt": "Lateral tracking camera shot as the same student scans notes with a smartphone, warm glowing particles rising from paper, consistent lighting and character, no text.",
                    "caption": "Instant AI extraction in seconds.",
                },
                {
                    "scene_no": 3,
                    "visual_prompt": "Medium shot orbiting the happy student tapping interactive quiz cards on a tablet with campus cafe background, vibrant colors, same character, no text.",
                    "caption": "Retain more with active recall.",
                },
                {
                    "scene_no": 4,
                    "visual_prompt": "Dynamic upward shot of the student smiling with an A-plus paper on sunny campus quad, golden hour backlight, cinematic lens flare, no text.",
                    "caption": "Start acing your exams today.",
                },
            ],
        }

        run_id = "test_run_promo"
        storyboard = build_storyboard(
            idea=SAMPLE_IDEA,
            script=SAMPLE_SCRIPT,
            tone="humorous",
            run_id=run_id,
        )

        self.assertIn("voiceover_text", storyboard)
        self.assertIn("scenes", storyboard)

        scenes = storyboard["scenes"]
        self.assertEqual(len(scenes), 4)

        for idx, scene in enumerate(scenes, start=1):
            self.assertEqual(scene["scene_no"], idx)
            self.assertIn("visual_prompt", scene)
            self.assertIn("caption", scene)
            # Ensure no text instruction is in visual prompt
            self.assertIn("no text", scene["visual_prompt"].lower())

        # Verify word count of voiceover text (~45-50 words)
        words = storyboard["voiceover_text"].split()
        self.assertGreaterEqual(len(words), 30)
        self.assertLessEqual(len(words), 65)

        # Verify saved file outputs/<run_id>/storyboard.json
        storyboard_path = OUTPUTS_DIR / run_id / "storyboard.json"
        self.assertTrue(storyboard_path.exists())
        with open(storyboard_path, "r", encoding="utf-8") as f:
            saved_storyboard = json.load(f)
        self.assertEqual(len(saved_storyboard["scenes"]), 4)


def run_standalone_demo():
    """Run interactive demonstration of Tools Agent and Promo Agent."""
    print("=" * 65)
    print("Demonstrating Tools Agent and Promo Agent")
    print(f"Idea: '{SAMPLE_IDEA}'")
    print("=" * 65)

    demo_run_id = "demo_tools_and_promo"

    with patch("agents.tools_agent.call_gemini") as mock_tools_gemini, \
         patch("agents.promo_agent.call_gemini") as mock_promo_gemini:

        mock_tools_gemini.return_value = {
            "tools": [
                {
                    "name": "TrOCR",
                    "purpose": "State-of-the-art handwritten text recognition from note scans.",
                    "cost": "free",
                    "setup_steps": ["pip install transformers", "from transformers import TrOCRProcessor"],
                    "link": "https://huggingface.co/microsoft/trocr-base-handwritten",
                },
                {
                    "name": "PyMuPDF",
                    "purpose": "Fast extraction of PDF text, images, and layout chunks.",
                    "cost": "free",
                    "setup_steps": ["pip install PyMuPDF"],
                    "link": "https://github.com/pymupdf/PyMuPDF",
                },
                {
                    "name": "Google Gemini 2.5 Flash",
                    "purpose": "Instant generation of adaptive multiple-choice questions.",
                    "cost": "freemium",
                    "setup_steps": ["pip install google-genai"],
                    "link": "https://ai.google.dev/",
                },
            ]
        }

        mock_promo_gemini.return_value = {
            "voiceover_text": (
                "Drowning in stacks of messy college lecture notes before final exams? "
                "Watch your handwritten pages transform into interactive flashcards and smart recall quizzes in seconds. "
                "Study half the time, retain twice as much, and ace every test with confidence. "
                "Turn your notes into top grades right now."
            ),
            "scenes": [
                {
                    "scene_no": 1,
                    "visual_prompt": "Cinematic slow camera tracking shot pushing in on an exhausted college student surrounded by towering stacks of handwritten notebooks under cozy midnight desk lamp glow, shallow depth of field, realistic textures, no text.",
                    "caption": "Drowning in endless college lecture notes?",
                },
                {
                    "scene_no": 2,
                    "visual_prompt": "Smooth lateral camera pan as the student points a smartphone camera toward the pages, soft glowing holographic particles rising from the paper, warm cinematic lighting, consistent character and room, no text.",
                    "caption": "Instant AI extraction scans every page.",
                },
                {
                    "scene_no": 3,
                    "visual_prompt": "Medium close-up shot orbiting the student now laughing and tapping interactive flashcards on a tablet with friends in a sunlit modern campus cafe, vibrant colors, same student, no text.",
                    "caption": "Master exams with personalized active recall.",
                },
                {
                    "scene_no": 4,
                    "visual_prompt": "Dynamic upward tracking shot of the confident student stepping out into campus daylight, holding up an A-plus exam paper with a triumph smile, golden hour backlight, cinematic lens flare, no text.",
                    "caption": "Turn your notes into top grades today.",
                },
            ],
        }

        tools = build_tools_list(
            idea=SAMPLE_IDEA,
            script=SAMPLE_SCRIPT,
            research=SAMPLE_RESEARCH,
            run_id=demo_run_id,
        )

        storyboard = build_storyboard(
            idea=SAMPLE_IDEA,
            script=SAMPLE_SCRIPT,
            tone="humorous",
            run_id=demo_run_id,
        )

    print("\n" + "=" * 65)
    print(f"Tools Generated: {len(tools)}")
    for t in tools:
        print(f"  • {t['name']} ({t['cost']}): {t['purpose']}")

    print("\n" + "=" * 65)
    print(f"Storyboard Generated ({len(storyboard['scenes'])} scenes):")
    print(f"Voiceover ({len(storyboard['voiceover_text'].split())} words):")
    print(f"  \"{storyboard['voiceover_text']}\"")
    print("\nScenes:")
    for s in storyboard["scenes"]:
        print(f"  Scene {s['scene_no']}: [{s['caption']}]")
        print(f"    Visual: {s['visual_prompt'][:90]}...")
    print("=" * 65)


if __name__ == "__main__":
    run_standalone_demo()
