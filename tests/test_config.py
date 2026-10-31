"""Unit tests verifying configuration defaults and paths."""

from pathlib import Path
import unittest
import config


class TestConfig(unittest.TestCase):
    def test_config_paths_exist(self):
        """Verify root directories and outputs folders are configured correctly."""
        self.assertIsInstance(config.BASE_DIR, Path)
        self.assertIsInstance(config.OUTPUTS_DIR, Path)
        self.assertTrue(config.OUTPUTS_DIR.exists())

    def test_critic_rubric_defaults(self):
        """Verify critic rubric scoring thresholds and retry limits."""
        self.assertEqual(config.CRITIC_CONFIG.max_retries, 3)
        self.assertEqual(config.CRITIC_CONFIG.pass_score, 9.0)
        self.assertEqual(config.CRITIC_CONFIG.min_criterion_score, 8.0)
        self.assertGreaterEqual(len(config.CRITIC_CONFIG.criteria), 5)

    def test_models_configured(self):
        """Verify all agent models are configured and critic differs from writer."""
        self.assertTrue(bool(config.MODELS.planner))
        self.assertTrue(bool(config.MODELS.research))
        self.assertTrue(bool(config.MODELS.writer))
        self.assertTrue(bool(config.MODELS.critic))
        self.assertTrue(bool(config.MODELS.tools))
        self.assertTrue(bool(config.MODELS.promo))
        self.assertTrue(bool(config.MODELS.veo))
        self.assertTrue(bool(config.GEMINI_MODEL))
        self.assertTrue(bool(config.GROQ_MODEL))

        # Critic must use a different model family/provider than the writer
        self.assertNotEqual(config.MODELS.critic, config.MODELS.writer)

    def test_promo_video_specs(self):
        """Verify storyboard duration, scene count, and tone-specific voices."""
        self.assertEqual(config.PROMO_CONFIG.total_scenes, 4)
        self.assertEqual(config.PROMO_CONFIG.scene_duration_seconds, 5)
        self.assertEqual(config.PROMO_CONFIG.total_duration_seconds, 20)
        self.assertTrue(bool(config.PROMO_CONFIG.voice_serious))
        self.assertTrue(bool(config.PROMO_CONFIG.voice_humorous))
        self.assertNotEqual(config.PROMO_CONFIG.voice_serious, config.PROMO_CONFIG.voice_humorous)

    def test_veo_config(self):
        """Verify Veo 3.1 model IDs, pricing rates, and max spending limits."""
        self.assertTrue(bool(config.VEO_CONFIG.draft_model))
        self.assertTrue(bool(config.VEO_CONFIG.final_model))
        self.assertEqual(config.VEO_CONFIG.draft_resolution, "720p")
        self.assertEqual(config.VEO_CONFIG.final_resolution, "1080p")
        self.assertGreater(config.VEO_CONFIG.draft_price_per_second, 0.0)
        self.assertGreater(config.VEO_CONFIG.final_price_per_second, 0.0)
        self.assertEqual(config.MAX_COST_USD, 5.0)


if __name__ == "__main__":
    unittest.main()
