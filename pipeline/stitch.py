"""Video Stitching Pipeline using FFmpeg.

Concatenates 4 scene video clips, removes original clip audio, integrates the voiceover,
optionally mixes background music at low volume, burns in styled subtitles, enforces
exact 20-second duration, and exports H.264 MP4 at 1920x1080 (16:9) or 1080x1920 (9:16).
"""

from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional, Union

import config
from config import BASE_DIR, PROMO_CONFIG
from pipeline.voice import find_binary, get_audio_duration


def verify_ffmpeg_installed() -> str:
    """Verify that FFmpeg is available and return its executable path.

    Raises:
        RuntimeError: If FFmpeg binary is not found on the system.
    """
    ffmpeg_bin = find_binary("ffmpeg")
    try:
        res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
        if res.returncode == 0:
            return ffmpeg_bin
    except Exception:
        pass

    error_msg = (
        "FFmpeg is not installed or not found in system PATH.\n"
        "To install FFmpeg:\n"
        "  - Windows (PowerShell): winget install Gyan.FFmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "  - Linux (Debian/Ubuntu): sudo apt update && sudo apt install -y ffmpeg\n"
        "After installation, ensure ffmpeg is added to your PATH."
    )
    raise RuntimeError(error_msg)


def get_export_resolution() -> tuple[int, int]:
    """Parse export dimensions from PROMO_CONFIG."""
    res_str = getattr(PROMO_CONFIG, "resolution", "") or "1920x1080"
    if "x" in res_str.lower():
        parts = res_str.lower().split("x")
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass

    # Fallback based on aspect ratio
    if getattr(PROMO_CONFIG, "aspect_ratio", "") == "9:16":
        return 1080, 1920
    return 1920, 1080


def format_subtitles_filter_path(subtitle_path: Path) -> str:
    """Escape Windows and Unix file paths for the FFmpeg subtitles filtergraph."""
    resolved = str(subtitle_path.resolve())
    # Forward slashes and escaped colon for Windows drive letters (e.g. C\:/path)
    escaped = resolved.replace("\\", "/").replace(":", "\\:")
    return escaped


def find_scene_clips(clips_dir: Path) -> list[Path]:
    """Find and sort scene_*.mp4 clips in clips_dir."""
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory does not exist: {clips_dir}")

    # Look for scene_1.mp4, scene_2.mp4, ...
    found = []
    for idx in range(1, PROMO_CONFIG.total_scenes + 1):
        target = clips_dir / f"scene_{idx}.mp4"
        if target.exists() and target.stat().st_size > 0:
            found.append(target)

    # Fallback to any sorted mp4 files if scene_N.mp4 naming is not exact
    if not found:
        all_mp4s = sorted(clips_dir.glob("*.mp4"))
        found = all_mp4s[:PROMO_CONFIG.total_scenes]

    if not found:
        raise FileNotFoundError(f"No scene video clips found in: {clips_dir}")

    return found


def make_promo(
    clips_dir: Union[str, Path],
    voiceover_path: Union[str, Path],
    subtitle_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    music_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Stitch 4 scene video clips, replace audio with voiceover (+ BGM), and burn subtitles.

    Args:
        clips_dir: Directory containing the 4 video clips.
        voiceover_path: Path to synthesized voiceover MP3.
        subtitle_path: Path to SRT subtitle file (optional).
        out_path: Destination path for final MP4 video.
        music_path: Optional background music path (defaults to assets/music.mp3).

    Returns:
        Dictionary containing video_path, duration, resolution, and status metadata.
    """
    ffmpeg_bin = verify_ffmpeg_installed()

    clips_path = Path(clips_dir)
    voice_path = Path(voiceover_path)
    if not voice_path.exists():
        raise FileNotFoundError(f"Voiceover audio file not found: {voice_path}")

    # Final output destination
    if out_path:
        final_out = Path(out_path)
    else:
        final_out = clips_path.parent / "final_promo.mp4"
    final_out.parent.mkdir(parents=True, exist_ok=True)

    # Locate the 4 scene clips
    scene_clips = find_scene_clips(clips_path)
    num_clips = len(scene_clips)
    target_duration = float(PROMO_CONFIG.total_duration_seconds)
    clip_duration = target_duration / max(1, num_clips)

    width, height = get_export_resolution()
    print(f"\nStitching {num_clips} clips into {width}x{height} promo video (Duration: {target_duration}s)...")

    # Optional background music: check provided path or assets/music.mp3
    bgm_file = None
    if music_path and Path(music_path).exists():
        bgm_file = Path(music_path)
    else:
        default_asset_bgm = BASE_DIR / "assets" / "music.mp3"
        if default_asset_bgm.exists():
            bgm_file = default_asset_bgm

    has_bgm = bgm_file is not None and bgm_file.stat().st_size > 0
    if has_bgm:
        print(f"Mixing background music from: {bgm_file} (Volume: 15%)")
    else:
        print("No background music file detected in assets/music.mp3. Proceeding with voiceover only.")

    # Check subtitles file
    sub_file = Path(subtitle_path) if subtitle_path else None
    has_subtitles = sub_file and sub_file.exists() and sub_file.stat().st_size > 10
    if has_subtitles:
        print(f"Burning subtitles from: {sub_file}")

    # Build FFmpeg command inputs
    cmd = [ffmpeg_bin, "-y"]

    # Inputs 0 .. num_clips-1: Video clips
    for clip in scene_clips:
        cmd.extend(["-i", str(clip)])

    # Input num_clips: Voiceover audio
    voice_input_idx = num_clips
    cmd.extend(["-i", str(voice_path)])

    # Input num_clips + 1: Background music (if present)
    bgm_input_idx = num_clips + 1 if has_bgm else None
    if has_bgm:
        cmd.extend(["-i", str(bgm_file)])

    # Construct Filtergraph
    filter_parts = []

    # 1. Video processing: trim, pad, and scale each clip to uniform dimensions
    concat_inputs = []
    for i in range(num_clips):
        v_tag = f"v{i}"
        filter_parts.append(
            f"[{i}:v]trim=duration={clip_duration:.3f},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{v_tag}]"
        )
        concat_inputs.append(f"[{v_tag}]")

    # Concat all video clips in order (dropping all clip audio)
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={num_clips}:v=1:a=0[vcat]")

    # 2. Subtitles burn-in
    if has_subtitles:
        escaped_sub_path = format_subtitles_filter_path(sub_file)
        # Font size adapted for portrait vs landscape
        font_size = 28 if width > height else 22
        style = (
            f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BackColour=&H80000000,BorderStyle=3,Outline=2,Shadow=1,Alignment=2,MarginV=40"
        )
        filter_parts.append(f"[vcat]subtitles='{escaped_sub_path}':force_style='{style}'[vout]")
        final_video_tag = "[vout]"
    else:
        final_video_tag = "[vcat]"

    # 3. Audio mixing
    if has_bgm:
        filter_parts.append(
            f"[{voice_input_idx}:a]volume=1.0[voice];"
            f"[{bgm_input_idx}:a]aloop=loop=-1:size=2e+09,volume=0.15[bgm];"
            f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        final_audio_tag = "[aout]"
    else:
        final_audio_tag = f"{voice_input_idx}:a"

    # Assemble complete command
    cmd.extend([
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        final_video_tag,
        "-map",
        final_audio_tag,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{target_duration:.2f}",
        str(final_out),
    ])

    # Execute FFmpeg stitching
    print("Executing FFmpeg rendering...")
    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode != 0:
        error_tail = process.stderr[-800:] if process.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg rendering failed with return code {process.returncode}:\n{error_tail}")

    actual_duration = get_audio_duration(final_out)
    size_mb = final_out.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("PROMO VIDEO STITCHING SUCCESSFUL!")
    print(f"  Output Path: {final_out}")
    print(f"  Resolution:  {width}x{height}")
    print(f"  Duration:    {actual_duration:.2f}s (Target: {target_duration}s)")
    print(f"  File Size:   {size_mb:.2f} MB")
    print(f"  Subtitles:   {'Burned in' if has_subtitles else 'None'}")
    print(f"  Audio Track: {'Voiceover + BGM' if has_bgm else 'Voiceover'}")
    print("=" * 65 + "\n")

    return {
        "video_path": str(final_out),
        "duration": actual_duration,
        "resolution": f"{width}x{height}",
        "file_size_mb": round(size_mb, 2),
        "has_subtitles": has_subtitles,
        "has_bgm": has_bgm,
    }
