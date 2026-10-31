"""Pipeline package for the multi-agent AI workflow."""

from pipeline.approval import approve_topics, display_topics
from pipeline.script_loop import build_script_graph, run_script_loop
from pipeline.stitch import get_export_resolution, make_promo, verify_ffmpeg_installed
from pipeline.video_gen import create_dummy_clip, estimate_generation_cost, generate_clips
from pipeline.voice import get_audio_duration, get_voice_for_tone, make_voiceover

__all__ = [
    "approve_topics",
    "build_script_graph",
    "create_dummy_clip",
    "display_topics",
    "estimate_generation_cost",
    "generate_clips",
    "get_audio_duration",
    "get_export_resolution",
    "get_voice_for_tone",
    "make_promo",
    "run_script_loop",
    "verify_ffmpeg_installed",
]
