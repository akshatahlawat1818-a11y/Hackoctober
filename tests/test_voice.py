"""Tests for Voice Pipeline using edge-tts, ffprobe, and ffmpeg."""

import os
from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline.voice import (
    adjust_audio_speed,
    get_audio_duration,
    get_voice_for_tone,
    make_voiceover,
    scale_srt_timestamps,
)

SAMPLE_VOICEOVER = (
    "Drowning in messy college lecture notes before final exams? "
    "Watch your handwritten pages transform into interactive flashcards and smart recall quizzes in seconds. "
    "Study half the time, retain twice as much, and ace every test with confidence. "
    "Turn your notes into top grades right now."
)


class TestVoicePipeline(unittest.TestCase):
    def test_voice_selection_by_tone(self):
        """Verify tone mapping selects calm voice for serious and energetic for humorous."""
        calm_voice = get_voice_for_tone("serious")
        energetic_voice = get_voice_for_tone("humorous")

        self.assertEqual(calm_voice, config.PROMO_CONFIG.voice_serious)
        self.assertEqual(energetic_voice, config.PROMO_CONFIG.voice_humorous)
        self.assertNotEqual(calm_voice, energetic_voice)

    def test_srt_rescaling_math(self):
        """Verify SRT timestamp mathematical scaling when adjusting tempo."""
        original_srt = (
            "1\n00:00:00,000 --> 00:00:10,000\nHello world!\n\n"
            "2\n00:00:10,000 --> 00:00:20,000\nSecond line.\n"
        )
        # Speed up by 2.0x -> 10s becomes 5s, 20s becomes 10s
        scaled = scale_srt_timestamps(original_srt, speed_factor=2.0)
        self.assertIn("00:00:05,000", scaled)
        self.assertIn("00:00:10,000", scaled)

    def test_make_sample_voiceover(self):
        """Verify edge-tts voice synthesis, SRT generation, and duration measurement."""
        # Generate real voiceover via edge-tts (free, no API key needed)
        run_id = "test_voiceover_run"
        out_mp3 = config.OUTPUTS_DIR / run_id / "voiceover.mp3"

        result = make_voiceover(
            text=SAMPLE_VOICEOVER,
            tone="humorous",
            out_path=out_mp3,
        )

        # Verify MP3 exists and has non-zero audio content
        self.assertTrue(out_mp3.exists())
        self.assertGreater(out_mp3.stat().st_size, 5000)

        # Verify SRT subtitles exist and contain text
        srt_path = Path(result["srt_path"])
        self.assertTrue(srt_path.exists())
        srt_content = srt_path.read_text(encoding="utf-8")
        self.assertGreater(len(srt_content), 20)

        # Verify audio duration measurement
        duration = result["duration"]
        self.assertGreater(duration, 0.0)
        # Should be paced close to 20s
        self.assertLessEqual(duration, 25.0)

        print(f"\n[Test Voiceover Created] Duration: {duration:.2f}s | Voice: {result['voice']}")


def run_standalone_demo():
    """Run interactive demo of make_voiceover with both tones."""
    print("=" * 65)
    print("Testing Voiceover Generation with edge-tts and duration check")
    print("=" * 65)

    demo_dir = config.OUTPUTS_DIR / "demo_voiceover"

    # 1. Humorous (energetic voice)
    humorous_mp3 = demo_dir / "voiceover_humorous.mp3"
    print("\n--- 1. Generating Humorous Voiceover (Energetic Voice) ---")
    res_humor = make_voiceover(
        text=SAMPLE_VOICEOVER,
        tone="humorous",
        out_path=humorous_mp3,
    )
    print(f"File: {res_humor['audio_path']}")
    print(f"Duration: {res_humor['duration']:.2f}s (Sped up: {res_humor['was_sped_up']})")

    # 2. Serious (calm voice)
    serious_mp3 = demo_dir / "voiceover_serious.mp3"
    print("\n--- 2. Generating Serious Voiceover (Calm Voice) ---")
    res_serious = make_voiceover(
        text="Academic retention hinges upon systematic spaced repetition and active cognitive recall. Convert your lecture materials into rigorous testing instruments today.",
        tone="serious",
        out_path=serious_mp3,
    )
    print(f"File: {res_serious['audio_path']}")
    print(f"Duration: {res_serious['duration']:.2f}s")

    print("\n" + "=" * 65)
    print("Subtitles Preview (from humorous run):")
    print("-" * 65)
    with open(res_humor["srt_path"], "r", encoding="utf-8") as f:
        print(f.read().strip())
    print("=" * 65)


if __name__ == "__main__":
    run_standalone_demo()
