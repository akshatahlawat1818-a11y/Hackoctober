"""Tests for Video Stitching Pipeline (pipeline/stitch.py).

Tests FFmpeg availability, resolution configuration, clip discovery,
and video stitching with voiceover, background music, and burned-in subtitles.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline.stitch import (
    find_scene_clips,
    format_subtitles_filter_path,
    get_export_resolution,
    make_promo,
    verify_ffmpeg_installed,
)
from pipeline.video_gen import create_dummy_clip
from pipeline.voice import find_binary, get_audio_duration


def create_dummy_audio(out_path: Path, duration_sec: float = 20.0, freq: int = 440) -> Path:
    """Generate a dummy MP3 audio file with FFmpeg using sine tone."""
    ffmpeg_bin = find_binary("ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq}:duration={duration_sec}",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


class TestStitchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare test workspace with dummy clips, voiceover, BGM, and subtitles."""
        cls.test_dir = config.OUTPUTS_DIR / "test_stitch_workspace"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 dummy 5-second scene clips
        cls.clips_dir = cls.test_dir / "clips"
        cls.clips_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 5):
            clip_file = cls.clips_dir / f"scene_{i}.mp4"
            create_dummy_clip(scene_no=i, out_path=clip_file, duration_seconds=5)

        # Generate dummy 20-second voiceover
        cls.voiceover_file = cls.test_dir / "voiceover.mp3"
        create_dummy_audio(cls.voiceover_file, duration_sec=20.0, freq=440)

        # Generate dummy background music file
        cls.bgm_file = cls.test_dir / "music.mp3"
        create_dummy_audio(cls.bgm_file, duration_sec=10.0, freq=220)

        # Create dummy SRT subtitles file
        cls.subtitles_file = cls.test_dir / "subtitles.srt"
        cls.subtitles_file.write_text(
            "1\n00:00:01,000 --> 00:00:04,500\nTurn college notes into quizzes.\n\n"
            "2\n00:00:05,000 --> 00:00:09,500\nInstant flashcards and active recall.\n\n"
            "3\n00:00:10,000 --> 00:00:14,500\nRetain twice as much in half the time.\n\n"
            "4\n00:00:15,000 --> 00:00:19,500\nStart acing your exams right now!\n",
            encoding="utf-8",
        )

    def test_verify_ffmpeg_installed(self):
        """Verify FFmpeg binary detection succeeds on installed systems."""
        ffmpeg_path = verify_ffmpeg_installed()
        self.assertTrue(Path(ffmpeg_path).exists())
        self.assertIn("ffmpeg", ffmpeg_path.lower())

    def test_verify_ffmpeg_missing_raises_runtime_error(self):
        """Verify helpful installation error message when FFmpeg is not found."""
        with patch("pipeline.stitch.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError) as ctx:
                verify_ffmpeg_installed()
            self.assertIn("FFmpeg is not installed", str(ctx.exception))
            self.assertIn("winget install", str(ctx.exception))

    def test_get_export_resolution(self):
        """Verify export resolution parsing for 16:9, 9:16, and custom sizes."""
        # Default landscape
        w, h = get_export_resolution()
        self.assertEqual((w, h), (1920, 1080))

        # Vertical format
        with patch("pipeline.stitch.PROMO_CONFIG", config.PromoVideoConfig(resolution="1080x1920")):
            self.assertEqual(get_export_resolution(), (1080, 1920))

        # Custom resolution
        with patch("pipeline.stitch.PROMO_CONFIG", config.PromoVideoConfig(resolution="1280x720")):
            self.assertEqual(get_export_resolution(), (1280, 720))

    def test_format_subtitles_filter_path(self):
        """Verify Windows drive letter escaping for the FFmpeg subtitles filter."""
        test_p = Path("C:/Users/Test/file.srt")
        escaped = format_subtitles_filter_path(test_p)
        self.assertNotIn("\\\\", escaped)
        if ":" in str(test_p.resolve()):
            self.assertIn("\\:", escaped)

    def test_find_scene_clips(self):
        """Verify discovery and sorting of 4 scene video clips."""
        found = find_scene_clips(self.clips_dir)
        self.assertEqual(len(found), 4)
        self.assertEqual([f.name for f in found], ["scene_1.mp4", "scene_2.mp4", "scene_3.mp4", "scene_4.mp4"])

    def test_find_scene_clips_missing_dir_raises(self):
        """Verify FileNotFoundError when scene clips directory does not exist."""
        missing = self.test_dir / "nonexistent_clips_dir"
        with self.assertRaises(FileNotFoundError):
            find_scene_clips(missing)

    def test_make_promo_clean_stitch(self):
        """Test stitching 4 dummy clips with voiceover only (no subs, no BGM)."""
        out_mp4 = self.test_dir / "promo_clean.mp4"
        if out_mp4.exists():
            out_mp4.unlink()

        result = make_promo(
            clips_dir=self.clips_dir,
            voiceover_path=self.voiceover_file,
            out_path=out_mp4,
        )

        self.assertTrue(out_mp4.exists())
        self.assertGreater(out_mp4.stat().st_size, 50000)
        self.assertEqual(result["video_path"], str(out_mp4))
        self.assertFalse(result["has_subtitles"])
        self.assertFalse(result["has_bgm"])

        # Check duration is ~20s
        duration = get_audio_duration(out_mp4)
        self.assertAlmostEqual(duration, 20.0, delta=1.0)

    def test_make_promo_full_mix_with_subtitles_and_bgm(self):
        """Test stitching with subtitles burned in and background music mixed."""
        out_mp4 = self.test_dir / "promo_full_mix.mp4"
        if out_mp4.exists():
            out_mp4.unlink()

        result = make_promo(
            clips_dir=self.clips_dir,
            voiceover_path=self.voiceover_file,
            subtitle_path=self.subtitles_file,
            music_path=self.bgm_file,
            out_path=out_mp4,
        )

        self.assertTrue(out_mp4.exists())
        self.assertGreater(out_mp4.stat().st_size, 50000)
        self.assertTrue(result["has_subtitles"])
        self.assertTrue(result["has_bgm"])

        duration = get_audio_duration(out_mp4)
        self.assertAlmostEqual(duration, 20.0, delta=1.0)


def run_standalone_demo():
    """Run interactive demonstration of make_promo with generated dummy clips."""
    print("=" * 65)
    print("Demonstrating Promo Stitching Pipeline with Dummy Clips")
    print("=" * 65)

    test_dir = config.OUTPUTS_DIR / "demo_stitch_output"
    test_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = test_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Generating 4 colored test scene clips (5s each)...")
    for i in range(1, 5):
        clip_path = clips_dir / f"scene_{i}.mp4"
        create_dummy_clip(i, clip_path, duration_seconds=5)

    print("\n2. Generating test voiceover tone and background music...")
    voice_path = test_dir / "voiceover.mp3"
    create_dummy_audio(voice_path, duration_sec=20.0, freq=350)

    bgm_path = test_dir / "music.mp3"
    create_dummy_audio(bgm_path, duration_sec=12.0, freq=200)

    sub_path = test_dir / "subtitles.srt"
    sub_path.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\n[Scene 1] Drowning in notes?\n\n"
        "2\n00:00:05,000 --> 00:00:10,000\n[Scene 2] AI generates flashcards in seconds!\n\n"
        "3\n00:00:10,000 --> 00:00:15,000\n[Scene 3] Master active recall easily.\n\n"
        "4\n00:00:15,000 --> 00:00:19,800\n[Scene 4] Ace every test with confidence!\n",
        encoding="utf-8",
    )

    promo_out = test_dir / "final_demo_promo.mp4"
    print("\n3. Stitching final promo video...")
    res = make_promo(
        clips_dir=clips_dir,
        voiceover_path=voice_path,
        subtitle_path=sub_path,
        music_path=bgm_path,
        out_path=promo_out,
    )

    print("\nStitching completed successfully!")
    print(f"  Result: {res}")


if __name__ == "__main__":
    unittest.main()
