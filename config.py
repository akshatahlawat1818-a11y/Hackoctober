"""Configuration settings for the Multi-Agent AI Workflow.

All model names, scoring thresholds, retry limits, and output paths are defined
here so they can be easily modified in a single location or overridden via
environment variables.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Graceful fallback if python-dotenv has not yet been installed
    pass

# Base project directory paths
BASE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = BASE_DIR / "agents"
PIPELINE_DIR = BASE_DIR / "pipeline"
UTILS_DIR = BASE_DIR / "utils"
OUTPUTS_DIR = BASE_DIR / "outputs"
TESTS_DIR = BASE_DIR / "tests"

# Ensure outputs directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# Default model configuration from environment
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class ModelConfig:
    """Central registry of model identifiers used across all agents and pipelines.

    Configurable through GEMINI_MODEL and GROQ_MODEL in .env, with defaults:
      - Gemini: gemini-3.8-flash
      - Groq: openai/gpt-oss-120b
    """

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Gemini models (google-genai SDK)
    planner: str = os.getenv("MODEL_PLANNER", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    research: str = os.getenv("MODEL_RESEARCH", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    writer: str = os.getenv("MODEL_WRITER", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    tools: str = os.getenv("MODEL_TOOLS", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    promo: str = os.getenv("MODEL_PROMO", os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))

    # Groq model for critic agent (distinct model from writer)
    critic: str = os.getenv("MODEL_CRITIC", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

    # Vertex AI models for video generation (Veo 3.1)
    veo: str = os.getenv("MODEL_VEO", "veo-3.1-generate-001")
    veo_draft: str = os.getenv("MODEL_VEO_DRAFT", "veo-3.1-lite-generate-001")
    veo_final: str = os.getenv("MODEL_VEO_FINAL", "veo-3.1-generate-001")


@dataclass(frozen=True)
class VeoConfig:
    """Configuration for Vertex AI Veo 3.1 video generation.

    Includes current official model IDs, resolutions, rates per second, and safety limits.
    """

    draft_model: str = os.getenv("MODEL_VEO_DRAFT", "veo-3.1-lite-generate-001")
    final_model: str = os.getenv("MODEL_VEO_FINAL", "veo-3.1-generate-001")
    draft_resolution: str = "720p"
    final_resolution: str = "1080p"
    draft_price_per_second: float = float(os.getenv("VEO_DRAFT_PRICE_PER_SEC", "0.05"))
    final_price_per_second: float = float(os.getenv("VEO_FINAL_PRICE_PER_SEC", "0.20"))
    max_cost_usd: float = float(os.getenv("MAX_COST_USD", "5.0"))


@dataclass(frozen=True)
class CriticRubricConfig:
    """Critic agent evaluation parameters and retry thresholds."""

    # Maximum number of retries for writer revision based on critic feedback
    max_retries: int = int(os.getenv("MAX_CRITIC_RETRIES", "3"))

    # Minimum average score across all criteria required to pass
    pass_score: float = float(os.getenv("CRITIC_PASS_SCORE", "9.0"))

    # Minimum score required on every individual criterion
    min_criterion_score: float = float(os.getenv("CRITIC_MIN_CRITERION_SCORE", "8.0"))

    # Standard rubric criteria evaluated by the critic
    criteria: list[str] = field(
        default_factory=lambda: [
            "Content Quality & Depth",
            "Clarity & Structure",
            "Tone Adherence",
            "Fact-Checking & Sourcing",
            "Engagement & Pacing",
        ]
    )


@dataclass(frozen=True)
class PromoVideoConfig:
    """Specifications for the 20-second promotional video generation."""

    total_scenes: int = 4
    scene_duration_seconds: int = 5
    total_duration_seconds: int = 20
    aspect_ratio: str = os.getenv("PROMO_ASPECT_RATIO", "16:9")
    resolution: str = os.getenv("PROMO_RESOLUTION", "1920x1080")
    tts_voice: str = os.getenv("TTS_VOICE", "en-US-ChristopherNeural")
    voice_serious: str = os.getenv("VOICE_SERIOUS", "en-US-ChristopherNeural")
    voice_humorous: str = os.getenv("VOICE_HUMOROUS", "en-US-AriaNeural")


@dataclass(frozen=True)
class APIConfig:
    """API credentials and cloud project configuration."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    google_cloud_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


# Instantiated global config objects for convenience
MODELS = ModelConfig()
CRITIC_CONFIG = CriticRubricConfig()
PROMO_CONFIG = PromoVideoConfig()
API_CONFIG = APIConfig()
VEO_CONFIG = VeoConfig()
MAX_COST_USD = VEO_CONFIG.max_cost_usd


def get_run_dir(run_id: str) -> Path:
    """Return and create the specific output directory for a given run ID."""
    run_dir = OUTPUTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
