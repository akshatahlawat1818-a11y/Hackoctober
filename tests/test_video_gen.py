"""Tests for Video Generation Pipeline (Vertex AI Veo 3.1 & Dry Run)."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from pipeline.video_gen import (
    estimate_generation_cost,
    generate_clips,
    get_clips_dir,
)

SAMPLE_STORYBOARD = {
    "voiceover_text": "Sample voiceover text for testing video generation pipeline.",
    "scenes": [
        {
            "scene_no": 1,
            "visual_prompt": "Cinematic camera push-in on an exhausted college student with books, no text.",
            "caption": "Scene 1: Late Night Study",
        },
        {
            "scene_no": 2,
            "visual_prompt": "Lateral pan of student scanning notebook with smartphone, no text.",
            "caption": "Scene 2: Instant Scanning",
        },
        {
            "scene_no": 3,
            "visual_prompt": "Medium shot of student smiling at interactive flashcards, no text.",
            "caption": "Scene 3: Active Recall",
        },
        {
            "scene_no": 4,
            "visual_prompt": "Wide shot of student on campus quad holding A-plus paper, no text.",
            "caption": "Scene 4: Top Grades",
        },
    ],
}


class TestVideoGenPipeline(unittest.TestCase):
    def test_cost_estimation(self):
        """Verify video generation cost calculation for draft and final quality modes."""
        # 4 scenes x 5s = 20s
        # Draft: 20s x $0.05 = $1.00
        cost, seconds, scenes, rate = estimate_generation_cost(SAMPLE_STORYBOARD, mode="draft")
        self.assertEqual(seconds, 20)
        self.assertEqual(len(scenes), 4)
        self.assertEqual(rate, config.VEO_CONFIG.draft_price_per_second)
        self.assertEqual(cost, 1.00)

        # Final: 20s x $0.20 = $4.00
        cost_f, seconds_f, scenes_f, rate_f = estimate_generation_cost(SAMPLE_STORYBOARD, mode="final")
        self.assertEqual(seconds_f, 20)
        self.assertEqual(cost_f, 4.00)

    def test_safety_limit_rejection(self):
        """Verify hard safety rejection when projected cost exceeds MAX_COST_USD."""
        # Override MAX_COST_USD to very low $0.50
        with patch("pipeline.video_gen.MAX_COST_USD", 0.50):
            with self.assertRaises(ValueError) as ctx:
                generate_clips(
                    storyboard=SAMPLE_STORYBOARD,
                    mode="draft",  # Cost is $1.00 > $0.50 limit
                    run_id="test_safety_limit",
                    dry_run=False,
                    input_func=lambda _: "yes",
                )
            self.assertIn("SAFETY LIMIT REFUSAL", str(ctx.exception))

    def test_skip_already_existing_scenes(self):
        """Verify that existing clips on disk are skipped from cost estimation."""
        run_id = "test_skip_existing_run"
        clips_dir = get_clips_dir(run_id)

        # Create dummy scene_1.mp4 beforehand
        scene_1_file = clips_dir / "scene_1.mp4"
        scene_1_file.write_bytes(b"\x00" * 4096)

        cost, seconds, scenes_to_gen, _ = estimate_generation_cost(
            SAMPLE_STORYBOARD, mode="draft", clips_dir=clips_dir
        )

        # Only 3 remaining scenes to generate: 3 x 5s = 15s -> $0.75
        self.assertEqual(len(scenes_to_gen), 3)
        self.assertEqual(seconds, 15)
        self.assertEqual(cost, 0.75)

    def test_dry_run_generates_all_4_clips(self):
        """Verify dry-run mode generates 4 valid test MP4 clips with FFmpeg."""
        run_id = "test_dry_run_run"
        clips = generate_clips(
            storyboard=SAMPLE_STORYBOARD,
            mode="draft",
            run_id=run_id,
            dry_run=True,
        )

        self.assertEqual(len(clips), 4)
        for idx, clip_path in enumerate(clips, 1):
            self.assertTrue(clip_path.exists())
            self.assertEqual(clip_path.name, f"scene_{idx}.mp4")
            self.assertGreater(clip_path.stat().st_size, 1000)

    @patch("pipeline.video_gen.genai.Client")
    def test_veo_polling_and_download(self, mock_client_cls):
        """Verify Vertex AI Veo polling and clip downloading with mocked SDK client."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Mock long-running operation
        mock_op = MagicMock()
        mock_op.done = True
        mock_op.error = None

        mock_video = MagicMock()
        mock_video.video.save = MagicMock()

        mock_op.response.generated_videos = [mock_video]
        mock_client.models.generate_videos.return_value = mock_op

        run_id = "test_veo_mock_run"
        single_scene_storyboard = {
            "scenes": [
                {"scene_no": 1, "visual_prompt": "Cinematic shot", "caption": "Scene 1"}
            ]
        }

        with patch("pipeline.video_gen.API_CONFIG", config.APIConfig(google_cloud_project="test-project")):
            clips = generate_clips(
                storyboard=single_scene_storyboard,
                mode="draft",
                run_id=run_id,
                dry_run=False,
                input_func=lambda _: "yes",
            )

        self.assertEqual(len(clips), 1)
        self.assertTrue(mock_client.models.generate_videos.called)
        self.assertTrue(mock_video.video.save.called)


def run_standalone_demo():
    """Run demonstration with --dry-run option."""
    print("=" * 65)
    print("Demonstrating Veo Video Generation Pipeline (Dry-Run Mode)")
    print("=" * 65)

    demo_run_id = "demo_veo_pipeline"
    clips = generate_clips(
        storyboard=SAMPLE_STORYBOARD,
        mode="draft",
        run_id=demo_run_id,
        dry_run=True,
    )

    print(f"\nGenerated {len(clips)} test clips:")
    for c in clips:
        size_kb = c.stat().st_size / 1024
        print(f"  • {c.name} ({size_kb:.1f} KB): {c}")
    print("=" * 65)


if __name__ == "__main__":
    run_standalone_demo()
