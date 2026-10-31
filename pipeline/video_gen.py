"""Video Generation Pipeline using Vertex AI Veo (google-genai SDK).

Generates 4 commercial scene clips using Google Veo 3.1 models via Vertex AI.
Includes cost estimation, user confirmation, safety spending limits, resume caching,
graceful error handling, and a --dry-run option for local development.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Optional, Union

from google import genai
from google.genai import types as genai_types

import config
from config import (
    API_CONFIG,
    MAX_COST_USD,
    OUTPUTS_DIR,
    PROMO_CONFIG,
    VEO_CONFIG,
    get_run_dir,
)
from pipeline.voice import find_binary

# Scene color palette for dummy clip generation in dry-run mode
DUMMY_SCENE_COLORS = {
    1: "navy",
    2: "darkolivegreen",
    3: "darkslateblue",
    4: "maroon",
}


def get_clips_dir(run_id: Optional[str] = None) -> Path:
    """Return and create outputs/<run_id>/clips/ folder."""
    base_dir = get_run_dir(run_id) if run_id else OUTPUTS_DIR
    clips_dir = base_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    return clips_dir


def log_video_cost(
    model: str,
    duration_sec: int,
    cost_usd: float,
    scene_no: int,
    status: str = "success",
) -> None:
    """Log individual clip generation cost to outputs/cost_log.csv."""
    cost_log_path = OUTPUTS_DIR / "cost_log.csv"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = cost_log_path.exists() and cost_log_path.stat().st_size > 0

    with open(cost_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "provider",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "status",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            "vertex_veo",
            model,
            duration_sec,
            0,
            duration_sec,
            f"{status} (scene {scene_no}, cost: ${cost_usd:.2f} USD)",
        ])


def create_dummy_clip(
    scene_no: int,
    out_path: Path,
    duration_seconds: int = 5,
    caption: str = "",
) -> Path:
    """Generate a 5-second test video clip using FFmpeg for dry runs."""
    ffmpeg_bin = find_binary("ffmpeg")
    color = DUMMY_SCENE_COLORS.get(scene_no, "black")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=1280x720:d={duration_seconds}",
        "-t",
        str(duration_seconds),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"  [Dry-Run] Created dummy clip for Scene {scene_no} ({color}, {duration_seconds}s): {out_path.name}")
    except Exception as e:
        print(f"  [Dry-Run Error] Failed to generate dummy clip with FFmpeg: {e}")
        # Create empty placeholder file if ffmpeg fails
        out_path.write_bytes(b"\x00" * 1024)

    return out_path


def estimate_generation_cost(
    storyboard: dict[str, Any],
    mode: str = "draft",
    clips_dir: Optional[Path] = None,
) -> tuple[float, int, list[dict[str, Any]], float]:
    """Calculate projected cost for scenes that have not yet been generated.

    Returns:
        (total_cost_usd, total_seconds, scenes_to_generate, rate_per_second)
    """
    is_draft = mode.lower() == "draft"
    rate_per_sec = VEO_CONFIG.draft_price_per_second if is_draft else VEO_CONFIG.final_price_per_second

    scenes = storyboard.get("scenes", [])
    scenes_to_generate = []

    for scene in scenes:
        scene_no = scene.get("scene_no", 1)
        if clips_dir:
            clip_file = clips_dir / f"scene_{scene_no}.mp4"
            if clip_file.exists() and clip_file.stat().st_size > 1024:
                continue
        scenes_to_generate.append(scene)

    scene_duration = PROMO_CONFIG.scene_duration_seconds
    total_seconds = len(scenes_to_generate) * scene_duration
    total_cost = round(total_seconds * rate_per_sec, 2)

    return total_cost, total_seconds, scenes_to_generate, rate_per_sec


def generate_clips(
    storyboard: dict[str, Any],
    mode: str = "draft",
    run_id: Optional[str] = None,
    dry_run: bool = False,
    input_func: Callable[[str], str] = input,
) -> list[Path]:
    """Generate 4 synchronized video clips for the storyboard using Vertex AI Veo 3.1.

    Args:
        storyboard: Dictionary containing scenes with visual prompts and captions.
        mode: 'draft' (cheaper Lite model at 720p) or 'final' (Fast/Standard at 1080p).
        run_id: Optional run ID for output paths.
        dry_run: If True, skips Vertex AI API and generates 5s dummy clips with FFmpeg.
        input_func: Input function for user confirmation prompt.

    Returns:
        List of Path objects for all 4 generated scene video clips.
    """
    mode = mode.lower()
    is_draft = mode == "draft"
    model_id = VEO_CONFIG.draft_model if is_draft else VEO_CONFIG.final_model
    resolution = VEO_CONFIG.draft_resolution if is_draft else VEO_CONFIG.final_resolution
    scene_duration = PROMO_CONFIG.scene_duration_seconds

    clips_dir = get_clips_dir(run_id)
    scenes = storyboard.get("scenes", [])
    if not scenes:
        raise ValueError("Storyboard does not contain any scenes to generate.")

    total_cost, total_seconds, scenes_to_generate, rate = estimate_generation_cost(
        storyboard=storyboard,
        mode=mode,
        clips_dir=clips_dir,
    )

    # 1. DRY RUN EXECUTION
    if dry_run:
        print("\n" + "=" * 65)
        print("          VEO VIDEO GENERATION: DRY RUN MODE")
        print("=" * 65)
        print("Skipping Vertex AI API calls. Generating 5-second dummy clips with FFmpeg...")
        all_clips = []
        for scene in scenes:
            scene_no = scene.get("scene_no", 1)
            clip_path = clips_dir / f"scene_{scene_no}.mp4"

            if clip_path.exists() and clip_path.stat().st_size > 1024:
                print(f"  [Scene {scene_no}] Clip already exists ({clip_path.name}). Skipping.")
            else:
                create_dummy_clip(
                    scene_no=scene_no,
                    out_path=clip_path,
                    duration_seconds=scene_duration,
                    caption=scene.get("caption", ""),
                )
                log_video_cost(
                    model="ffmpeg-dummy",
                    duration_sec=scene_duration,
                    cost_usd=0.0,
                    scene_no=scene_no,
                    status="dry_run_success",
                )
            all_clips.append(clip_path)

        print("\nAll dry-run scene clips ready in:", clips_dir)
        print("=" * 65 + "\n")
        return all_clips

    # 2. HARD SAFETY LIMIT CHECK
    if total_cost > MAX_COST_USD:
        error_msg = (
            f"SAFETY LIMIT REFUSAL: Estimated cost (${total_cost:.2f} USD) exceeds "
            f"maximum spending threshold MAX_COST_USD (${MAX_COST_USD:.2f} USD). "
            f"Generation halted to prevent unexpected billing."
        )
        print(f"\n[ERROR] {error_msg}\n")
        raise ValueError(error_msg)

    # Check if all clips are already cached
    if not scenes_to_generate:
        print("\nAll scene clips already exist on disk. Skipping generation to save API costs.")
        return [clips_dir / f"scene_{s.get('scene_no', idx)}.mp4" for idx, s in enumerate(scenes, 1)]

    # 3. COST ESTIMATION & USER CONFIRMATION
    print("\n" + "=" * 65)
    print("             VEO 3.1 VIDEO GENERATION ESTIMATE")
    print("=" * 65)
    print(f"Mode:             {mode.upper()} ({resolution})")
    print(f"Model ID:         {model_id}")
    print(f"Scenes to create: {len(scenes_to_generate)} of {len(scenes)} ({total_seconds} seconds total)")
    print(f"Pricing Rate:     ${rate:.2f} / second")
    print(f"Estimated Cost:   ${total_cost:.2f} USD")
    print(f"Max Cost Limit:   ${MAX_COST_USD:.2f} USD")
    print("=" * 65)

    try:
        user_approval = input_func("Type 'yes' to proceed with Veo video generation: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        user_approval = "no"

    if user_approval != "yes":
        print("\nVeo video generation cancelled by user. Returning existing cached clips.")
        return [clips_dir / f"scene_{s.get('scene_no', idx)}.mp4" for idx, s in enumerate(scenes, 1) if (clips_dir / f"scene_{s.get('scene_no', idx)}.mp4").exists()]

    # 4. INITIALIZE VERTEX AI CLIENT VIA google-genai SDK
    project = API_CONFIG.google_cloud_project or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = API_CONFIG.google_cloud_location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT is not configured. Veo on Vertex AI requires a valid Google Cloud project ID in .env."
        )

    print(f"\nInitializing Vertex AI client (Project: {project}, Location: {location})...")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )

    all_clips: list[Path] = []

    # 5. GENERATE SCENES WITH VEO
    for scene in scenes:
        scene_no = scene.get("scene_no", 1)
        clip_path = clips_dir / f"scene_{scene_no}.mp4"

        # Resume check: Skip if already exists on disk
        if clip_path.exists() and clip_path.stat().st_size > 1024:
            print(f"\n[Scene {scene_no}/{len(scenes)}] Clip already exists ({clip_path.name}). Skipping.")
            all_clips.append(clip_path)
            continue

        prompt = scene.get("visual_prompt", "")
        print(f"\n[Scene {scene_no}/{len(scenes)}] Generating with Veo ({model_id}, {resolution})...")
        print(f"  Prompt: {prompt[:120]}...")

        scene_cost = round(scene_duration * rate, 2)

        try:
            # Submit generation request
            operation = client.models.generate_videos(
                model=model_id,
                prompt=prompt,
                config=genai_types.GenerateVideosConfig(
                    number_of_videos=1,
                    duration_seconds=scene_duration,
                    aspect_ratio=PROMO_CONFIG.aspect_ratio,
                    resolution=resolution,
                ),
            )

            print("  Operation submitted. Polling for completion...")
            start_poll = time.time()
            max_poll_seconds = 420  # 7 minutes max wait
            poll_interval = 12

            # Poll long-running operation
            while not operation.done:
                if time.time() - start_poll > max_poll_seconds:
                    raise TimeoutError(f"Generation for Scene {scene_no} timed out after {max_poll_seconds}s.")
                time.sleep(poll_interval)
                operation = client.operations.get(operation)
                print(f"  Still rendering Scene {scene_no} (elapsed: {int(time.time() - start_poll)}s)...")

            # Check operation errors / safety blocks
            if operation.error:
                error_msg = f"Veo generation error for Scene {scene_no}: {operation.error}"
                print(f"\n  [ERROR] {error_msg}")
                log_video_cost(
                    model=model_id,
                    duration_sec=scene_duration,
                    cost_usd=0.0,
                    scene_no=scene_no,
                    status=f"failed: {operation.error}",
                )
                continue

            # Download and save completed clip
            if operation.response and operation.response.generated_videos:
                generated_video = operation.response.generated_videos[0]
                if hasattr(client, "files") and hasattr(client.files, "download"):
                    try:
                        client.files.download(file=generated_video.video)
                    except Exception:
                        pass

                generated_video.video.save(str(clip_path))
                print(f"  Downloaded Scene {scene_no} successfully to: {clip_path}")
                log_video_cost(
                    model=model_id,
                    duration_sec=scene_duration,
                    cost_usd=scene_cost,
                    scene_no=scene_no,
                    status="success",
                )
                all_clips.append(clip_path)
            else:
                print(f"  [Warning] Scene {scene_no} finished but returned no generated video data.")

        except Exception as e:
            print(f"  [ERROR] Failed to generate Scene {scene_no}: {e}")
            log_video_cost(
                model=model_id,
                duration_sec=scene_duration,
                cost_usd=0.0,
                scene_no=scene_no,
                status=f"error: {type(e).__name__}",
            )

    print("\n" + "=" * 65)
    print(f"Veo clip generation completed. Ready clips: {len(all_clips)}/{len(scenes)}")
    print("=" * 65 + "\n")

    return all_clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video clips with Veo 3.1 via Vertex AI")
    parser.add_argument("--storyboard", type=str, help="Path to storyboard.json", default="")
    parser.add_argument("--mode", type=str, choices=["draft", "final"], default="draft", help="draft (720p Lite) or final (1080p)")
    parser.add_argument("--run-id", type=str, default="", help="Run ID for output folder")
    parser.add_argument("--dry-run", action="store_true", help="Use local FFmpeg dummy clips without calling Vertex AI")
    args = parser.parse_args()

    sample_storyboard = {
        "voiceover_text": "Sample voiceover text for testing video clip pipeline generation.",
        "scenes": [
            {"scene_no": 1, "visual_prompt": "Cinematic camera push in on college student with books, no text.", "caption": "Scene 1"},
            {"scene_no": 2, "visual_prompt": "Lateral pan of student scanning notebook with smartphone, no text.", "caption": "Scene 2"},
            {"scene_no": 3, "visual_prompt": "Medium shot of student smiling at interactive flashcards, no text.", "caption": "Scene 3"},
            {"scene_no": 4, "visual_prompt": "Wide shot of student on campus quad holding A plus paper, no text.", "caption": "Scene 4"},
        ],
    }

    if args.storyboard and Path(args.storyboard).exists():
        with open(args.storyboard, "r", encoding="utf-8") as f:
            sample_storyboard = json.load(f)

    generate_clips(
        storyboard=sample_storyboard,
        mode=args.mode,
        run_id=args.run_id or "cli_veo_run",
        dry_run=args.dry_run,
    )
