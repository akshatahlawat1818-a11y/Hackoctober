"""Voice Generation Pipeline using edge-tts, ffprobe, and ffmpeg.

Synthesizes voiceovers with tone-appropriate neural voices, extracts word/sentence
level timing to produce synchronized SRT subtitles, and automatically enforces a
strict 20-second duration limit via intelligent tempo adjustment.
"""

import asyncio
from datetime import timedelta
import functools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional, Union

import edge_tts

import config
from config import PROMO_CONFIG


@functools.lru_cache(maxsize=32)
def find_binary(name: str) -> str:
    """Locate executable binary in PATH or Windows WinGet packages directory."""
    found = shutil.which(name)
    if found:
        return found

    # Check winget packages directory on Windows
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_dir.exists():
            matches = list(winget_dir.glob(f"**/{name}.exe"))
            if matches:
                return str(matches[0])

    return name


def get_voice_for_tone(tone: str) -> str:
    """Select appropriate voice from config based on tone."""
    t = (tone or "").lower()
    if any(k in t for k in ["humor", "funny", "energetic", "fun", "casual", "playful"]):
        return PROMO_CONFIG.voice_humorous
    return PROMO_CONFIG.voice_serious


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """Measure exact audio duration in seconds using ffprobe."""
    ffprobe_bin = find_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"Warning: ffprobe failed to measure duration for {file_path}: {e}")
        # Rough fallback estimation: ~16KB/sec for 128kbps MP3
        size = Path(file_path).stat().st_size
        return round(size / 16000.0, 2)


def scale_srt_timestamps(srt_content: str, speed_factor: float) -> str:
    """Scale subtitle timestamps when audio is sped up by speed_factor."""
    if speed_factor == 1.0 or not srt_content:
        return srt_content

    time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

    def _rescale_time(match: re.Match) -> str:
        """Rescale an SRT timestamp regex match by the speed factor."""
        h = int(match.group(1))
        m = int(match.group(2))
        s = int(match.group(3))
        ms = int(match.group(4))
        total_ms = (h * 3600 + m * 60 + s) * 1000 + ms
        new_ms = int(total_ms / speed_factor)

        new_h = new_ms // 3600000
        new_ms %= 3600000
        new_m = new_ms // 60000
        new_ms %= 60000
        new_s = new_ms // 1000
        new_ms %= 1000
        return f"{new_h:02d}:{new_m:02d}:{new_s:02d},{new_ms:03d}"

    return time_pattern.sub(_rescale_time, srt_content)


def adjust_audio_speed(
    mp3_path: Path,
    srt_path: Optional[Path] = None,
    max_duration: float = 20.0,
    max_speedup: float = 1.15,
) -> tuple[float, float, bool]:
    """Check duration and speed up audio with ffmpeg if longer than max_duration.

    Returns:
        (final_duration, applied_speed_factor, was_sped_up)
    """
    duration = get_audio_duration(mp3_path)
    if duration <= max_duration:
        return duration, 1.0, False

    # Calculate required speedup factor
    needed_factor = duration / max_duration
    applied_factor = min(needed_factor, max_speedup)

    ffmpeg_bin = find_binary("ffmpeg")
    temp_spedup = mp3_path.with_name(f"temp_speed_{mp3_path.name}")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(mp3_path),
        "-filter:a",
        f"atempo={applied_factor:.4f}",
        "-vn",
        str(temp_spedup),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        # Replace original file with sped up version
        if temp_spedup.exists():
            temp_spedup.replace(mp3_path)
    except Exception as e:
        print(f"Warning: ffmpeg speedup failed: {e}")
        if temp_spedup.exists():
            temp_spedup.unlink()
        return duration, 1.0, False

    new_duration = get_audio_duration(mp3_path)

    # Adjust SRT subtitles to match new tempo
    if srt_path and srt_path.exists():
        original_srt = srt_path.read_text(encoding="utf-8")
        scaled_srt = scale_srt_timestamps(original_srt, applied_factor)
        srt_path.write_text(scaled_srt, encoding="utf-8")

    # Check if still exceeding 20 seconds
    if new_duration > max_duration:
        print(
            f"\n[WARNING] Voiceover audio is {new_duration:.2f}s long! "
            f"Even after maximum {max_speedup}x speedup, it exceeds the 20-second target. "
            f"Consider shortening the voiceover text by {int((new_duration - max_duration) * 2.5)} words.\n"
        )
    else:
        print(f"Adjusted audio tempo: sped up by {applied_factor:.2f}x to fit 20s target ({new_duration:.2f}s).")

    return new_duration, applied_factor, True


async def _synthesize_edge_tts(
    text: str,
    voice: str,
    out_mp3: Path,
    out_srt: Path,
) -> list[dict[str, Any]]:
    """Internal async generator for edge-tts streaming and SRT extraction."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    submaker = edge_tts.SubMaker()
    timing_events: list[dict[str, Any]] = []

    with open(out_mp3, "wb") as audio_file:
        async for chunk in communicate.stream():
            chunk_type = chunk.get("type")
            if chunk_type == "audio":
                audio_file.write(chunk["data"])
            elif chunk_type in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
                timing_events.append({
                    "type": chunk_type,
                    "offset_ms": chunk.get("offset", 0) / 10000,
                    "duration_ms": chunk.get("duration", 0) / 10000,
                    "text": chunk.get("text", ""),
                })

    # Save SRT file
    srt_content = submaker.get_srt()
    if not srt_content.strip() and timing_events:
        # Fallback SRT builder if get_srt() was blank
        lines = []
        for idx, event in enumerate(timing_events, start=1):
            start_s = event["offset_ms"] / 1000.0
            end_s = (event["offset_ms"] + event["duration_ms"]) / 1000.0
            lines.append(f"{idx}\n{timedelta(seconds=start_s)} --> {timedelta(seconds=end_s)}\n{event['text']}\n")
        srt_content = "\n".join(lines)

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return timing_events


def make_voiceover(
    text: str,
    tone: str = "serious",
    out_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Generate voiceover audio with edge-tts, subtitles, and duration enforcement.

    Args:
        text: Voiceover script text.
        tone: 'serious' (calm voice) or 'humorous' (energetic voice).
        out_path: Destination path for mp3 file.

    Returns:
        Dictionary containing audio_path, srt_path, duration, voice, and speedup info.
    """
    if out_path:
        mp3_path = Path(out_path)
    else:
        mp3_path = config.OUTPUTS_DIR / "voiceover.mp3"

    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path = mp3_path.with_suffix(".srt")
    timing_path = mp3_path.with_name(f"{mp3_path.stem}_timings.json")

    # Select voice based on tone from config.py
    voice = get_voice_for_tone(tone)
    print(f"Synthesizing voiceover with voice: '{voice}' (Tone: {tone})...")

    # Execute async synthesis synchronously
    try:
        timings = asyncio.run(_synthesize_edge_tts(text, voice, mp3_path, srt_path))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            timings = executor.submit(asyncio.run, _synthesize_edge_tts(text, voice, mp3_path, srt_path)).result()

    # Save raw timing metadata
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)

    # Measure duration and enforce 20s limit with ffmpeg
    final_duration, speed_factor, was_sped_up = adjust_audio_speed(
        mp3_path=mp3_path,
        srt_path=srt_path,
        max_duration=PROMO_CONFIG.total_duration_seconds,
        max_speedup=1.15,
    )

    print(f"Voiceover generated successfully: {mp3_path} ({final_duration:.2f}s)")
    print(f"Subtitles generated successfully: {srt_path}")

    return {
        "audio_path": str(mp3_path),
        "srt_path": str(srt_path),
        "timing_path": str(timing_path),
        "voice": voice,
        "duration": final_duration,
        "speed_factor": speed_factor,
        "was_sped_up": was_sped_up,
    }
