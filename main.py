"""Main Command-Line Runner for the Multi-Agent AI Workflow.

Executes the full end-to-end flow:
  1. Idea Input
  2. Planner Agent (Topic generation)
  3. Human Approval (Interactive CLI review)
  4. Researcher Agent (DuckDuckGo + Wikipedia with verified sources)
  5. Script Revision Loop (Writer + Critic in LangGraph with max 3 retries)
  6. Tools Agent (Identifies APIs, tools, costs, setup)
  7. Promo Agent (Generates 20-second 4-scene storyboard + voiceover script)
  8. Voiceover Pipeline (edge-tts + auto duration adjustment with FFmpeg)
  9. Video Generation Pipeline (Vertex AI Veo 3.1 or FFmpeg --dry-run dummy clips)
 10. Video Stitching Pipeline (FFmpeg concat, audio replace + BGM, burned subtitles)

Supports --resume <run_id>, --no-video, and --dry-run.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import BASE_DIR, OUTPUTS_DIR, get_run_dir
from agents import (
    build_storyboard,
    build_tools_list,
    plan_topics,
    research_all,
)
from pipeline import (
    approve_topics,
    generate_clips,
    make_promo,
    make_voiceover,
    run_script_loop,
)
from utils.run_manager import RunManager


def print_banner() -> None:
    """Print the ASCII art banner and pipeline overview."""
    print(r"""
===============================================================================
       MULTI-AGENT AI SCRIPT & VIDEO PROMO GENERATION PIPELINE
===============================================================================
  Agents: Planner -> Researcher -> Writer & Critic Loop -> Tools -> Promo
  Video:  Vertex AI Veo 3.1 | Voice: edge-tts | Assembly: FFmpeg
===============================================================================
""")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the multi-agent workflow."""
    parser = argparse.ArgumentParser(
        description="Run the end-to-end multi-agent script and promo video generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--idea",
        type=str,
        default=None,
        help="The core project/video concept (prompts interactively if not provided).",
    )
    parser.add_argument(
        "--tone",
        type=str,
        choices=["serious", "humorous"],
        default="serious",
        help="Tone of voice for the script and promotional video.",
    )
    parser.add_argument(
        "--length",
        type=str,
        default="60 seconds",
        help="Target length of the main video script (e.g., '60 seconds', '90 seconds', '2 minutes').",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Resume an existing run by ID, skipping any steps whose outputs already exist on disk.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Stop workflow after the storyboard and tools list (skips voiceover, video gen, and stitch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate dummy video clips with FFmpeg instead of calling Vertex AI Veo (avoids API costs).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["draft", "final"],
        default="draft",
        help="Veo generation mode: 'draft' (Lite 720p at $0.05/s) or 'final' (Standard 1080p at $0.20/s).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        "--auto-approve",
        action="store_true",
        dest="auto_approve",
        help="Automatically approve planner research topics without interactive CLI prompt.",
    )
    return parser.parse_args()


def run_pipeline(
    idea: Optional[str] = None,
    tone: str = "serious",
    length: str = "60 seconds",
    resume_run_id: Optional[str] = None,
    no_video: bool = False,
    dry_run: bool = False,
    mode: str = "draft",
    auto_approve: bool = False,
    progress_callback: Optional[Callable[[str, str, float, Optional[dict[str, Any]]], None]] = None,
) -> dict[str, Any]:
    """Execute the multi-agent workflow in sequential order with step caching and resumption."""
    # 1. Initialize Run Manager
    run_manager = RunManager(run_id=resume_run_id)
    run_id = run_manager.run_id
    run_dir = run_manager.run_dir

    print(f"\n[RUN INITIALIZED] Run ID: {run_id}")
    print(f"[OUTPUT DIRECTORY] {run_dir}")

    # Check for existing user input if resuming
    input_file = run_dir / "00_user_input.json"
    if resume_run_id and input_file.exists():
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                saved_input = json.load(f)
                idea = idea or saved_input.get("idea")
                tone = saved_input.get("tone", tone)
                length = saved_input.get("length", length)
                print(f"[Resume] Restored parameters from run: Idea='{idea[:40]}...', Tone='{tone}', Length='{length}'")
        except Exception as e:
            print(f"[Warning] Could not load previous input: {e}")

    # Prompt interactively if idea is still empty
    if not idea or not idea.strip():
        try:
            print("\nPlease provide the initial idea for your project or video:")
            user_input_idea = input("Idea: ").strip()
        except EOFError:
            user_input_idea = ""
        idea = user_input_idea or "AI app that turns messy college notes into viral study quizzes"
        print(f"Using idea: {idea}")

    # Save user input for traceability
    run_manager.save_user_input(idea=idea, tone=tone, length=length, mode=mode, dry_run=dry_run, no_video=no_video)

    # -------------------------------------------------------------------------
    # STEP 1 & 2: PLANNER AGENT & CLI APPROVAL
    # -------------------------------------------------------------------------
    topics_file = run_dir / "topics.json"
    if resume_run_id and topics_file.exists():
        print(f"\n[Step 1 & 2: Planner & Approval] Skipped (found {topics_file.name})")
        with open(topics_file, "r", encoding="utf-8") as f:
            topics = json.load(f)
        if progress_callback:
            progress_callback("Planner", f"Loaded {len(topics)} approved research topics.", 0.15, {"topics": topics})
    else:
        print("\n" + "=" * 65)
        print("STEP 1: PLANNER AGENT (Gemini)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Planner", "Analyzing concept and generating 5-8 research angles...", 0.08, None)
        raw_topics = plan_topics(idea=idea, tone=tone, length=length)

        print("\n" + "=" * 65)
        print("STEP 2: CLI HUMAN APPROVAL")
        print("=" * 65)
        topics = approve_topics(
            topics=raw_topics,
            idea=idea,
            tone=tone,
            length=length,
            run_id=run_id,
            auto_approve=auto_approve,
        )
        if progress_callback:
            progress_callback("Planner", f"Approved {len(topics)} research topics.", 0.15, {"topics": topics})

    # -------------------------------------------------------------------------
    # STEP 3: RESEARCHER AGENT (DuckDuckGo + Wikipedia)
    # -------------------------------------------------------------------------
    research_file = run_dir / "research.json"
    if resume_run_id and research_file.exists():
        print(f"\n[Step 3: Researcher Agent] Skipped (found {research_file.name})")
        with open(research_file, "r", encoding="utf-8") as f:
            research_data = json.load(f)
        if progress_callback:
            progress_callback("Researcher", f"Loaded cached research ({len(research_data)} topics).", 0.35, {"research": research_data})
    else:
        print("\n" + "=" * 65)
        print("STEP 3: RESEARCHER AGENT (DDG & Wikipedia with Source Validation)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Researcher", "Querying DuckDuckGo & Wikipedia with verified sources...", 0.20, None)
        research_data = research_all(topics=topics, run_id=run_id, delay_seconds=2.0)
        with open(research_file, "w", encoding="utf-8") as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        if progress_callback:
            progress_callback("Researcher", f"Research complete across {len(research_data)} topics.", 0.35, {"research": research_data})

    # -------------------------------------------------------------------------
    # STEP 4: SCRIPT REVISION LOOP (Writer + Critic via LangGraph)
    # -------------------------------------------------------------------------
    script_file = run_dir / "final_script.md"
    if resume_run_id and script_file.exists():
        print(f"\n[Step 4: Script Loop] Skipped (found {script_file.name})")
        script_text = script_file.read_text(encoding="utf-8")
        script_result = {"final_script": script_text, "status": "CACHED"}
        if progress_callback:
            progress_callback("Writer", "Script already finalized (Cached).", 0.50, None)
            progress_callback("Critic", "Critic evaluations loaded (Cached).", 0.60, {"script_result": script_result})
    else:
        print("\n" + "=" * 65)
        print("STEP 4: SCRIPT REVISION LOOP (Writer: Gemini | Critic: Groq Rubric)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Writer", "Drafting script and iteratively revising with Critic Agent...", 0.40, None)
        script_result = run_script_loop(
            idea=idea,
            tone=tone,
            length=length,
            research=research_data,
            run_id=run_id,
        )
        script_text = script_result.get("final_script", "")
        if progress_callback:
            progress_callback("Writer", "Script draft completed.", 0.50, {"script": script_text})
            avg_sc = script_result.get("average", 0.0)
            st_sc = script_result.get("status", "completed")
            progress_callback("Critic", f"Critique complete (Status: {st_sc}, Avg: {avg_sc:.2f}/10).", 0.60, {"script_result": script_result})

    # -------------------------------------------------------------------------
    # STEP 5: TOOLS AGENT (APIs, Setup, and Pricing)
    # -------------------------------------------------------------------------
    tools_file = run_dir / "tools.json"
    if resume_run_id and tools_file.exists():
        print(f"\n[Step 5: Tools Agent] Skipped (found {tools_file.name})")
        with open(tools_file, "r", encoding="utf-8") as f:
            tools_list = json.load(f)
        if progress_callback:
            progress_callback("Tools", f"Loaded {len(tools_list)} tools and stack items (Cached).", 0.72, {"tools": tools_list})
    else:
        print("\n" + "=" * 65)
        print("STEP 5: TOOLS AGENT (APIs, Stack & Cost Estimation)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Tools", "Identifying APIs, tech stack, and cost breakdown...", 0.65, None)
        tools_list = build_tools_list(
            idea=idea,
            script=script_text,
            research=research_data,
            run_id=run_id,
        )
        if progress_callback:
            progress_callback("Tools", f"Tools list constructed ({len(tools_list)} items).", 0.72, {"tools": tools_list})

    # -------------------------------------------------------------------------
    # STEP 6: PROMO AGENT (20-Second Storyboard & Voiceover Script)
    # -------------------------------------------------------------------------
    storyboard_file = run_dir / "storyboard.json"
    if resume_run_id and storyboard_file.exists():
        print(f"\n[Step 6: Promo Agent] Skipped (found {storyboard_file.name})")
        with open(storyboard_file, "r", encoding="utf-8") as f:
            storyboard = json.load(f)
        if progress_callback:
            progress_callback("Promo", f"Storyboard ready ({len(storyboard.get('scenes', []))} scenes, Cached).", 0.78, {"storyboard": storyboard})
    else:
        print("\n" + "=" * 65)
        print("STEP 6: PROMO AGENT (4-Scene 20s Storyboard)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Promo", "Crafting 20-second storyboard and voiceover script...", 0.75, None)
        storyboard = build_storyboard(
            idea=idea,
            script=script_text,
            tone=tone,
            run_id=run_id,
        )
        if progress_callback:
            progress_callback("Promo", f"Storyboard ready ({len(storyboard.get('scenes', []))} scenes).", 0.78, {"storyboard": storyboard})

    # Early exit if --no-video was specified
    if no_video:
        print("\n" + "=" * 65)
        print("[NOTICE] --no-video option set: Halting workflow before video rendering.")
        print("=" * 65)
        _print_final_summary(
            run_id=run_id,
            idea=idea,
            tone=tone,
            length=length,
            script_file=script_file,
            tools_file=tools_file,
            promo_file=None,
            storyboard_file=storyboard_file,
        )
        if progress_callback:
            progress_callback("Video", "Video generation skipped via --no-video.", 1.0, None)
        return {
            "run_id": run_id,
            "script_file": str(script_file),
            "tools_file": str(tools_file),
            "storyboard_file": str(storyboard_file),
            "promo_file": None,
        }

    # -------------------------------------------------------------------------
    # STEP 7: VOICEOVER SYNTHESIS (edge-tts + duration control)
    # -------------------------------------------------------------------------
    voiceover_file = run_dir / "voiceover.mp3"
    srt_file = run_dir / "voiceover.srt"
    if resume_run_id and voiceover_file.exists() and srt_file.exists():
        print(f"\n[Step 7: Voiceover Pipeline] Skipped (found {voiceover_file.name} & {srt_file.name})")
        if progress_callback:
            progress_callback("Voiceover", "Voiceover and SRT subtitles ready (Cached).", 0.88, None)
    else:
        print("\n" + "=" * 65)
        print("STEP 7: VOICEOVER SYNTHESIS (edge-tts + FFmpeg timing)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Voiceover", "Synthesizing voiceover with edge-tts and timing SRT...", 0.82, None)
        voiceover_text = storyboard.get("voiceover_text", "")
        make_voiceover(
            text=voiceover_text,
            tone=tone,
            out_path=voiceover_file,
        )
        if progress_callback:
            progress_callback("Voiceover", "Voiceover and SRT subtitles successfully generated.", 0.88, None)

    # -------------------------------------------------------------------------
    # STEP 8: VIDEO GENERATION (Vertex AI Veo 3.1 / Dry Run)
    # -------------------------------------------------------------------------
    clips_dir = run_dir / "clips"
    print("\n" + "=" * 65)
    print(f"STEP 8: VIDEO GENERATION ({'DRY-RUN DUMMY CLIPS' if dry_run else f'VERTEX AI VEO 3.1 ({mode.upper()})'})")
    print("=" * 65)
    if progress_callback:
        mode_desc = "Generating FFmpeg dummy clips (Dry-run)" if dry_run else f"Calling Vertex AI Veo 3.1 ({mode})"
        progress_callback("Video", f"{mode_desc}...", 0.90, None)
    generated_clips = generate_clips(
        storyboard=storyboard,
        mode=mode,
        run_id=run_id,
        dry_run=dry_run,
    )
    if progress_callback:
        progress_callback("Video", f"{len(generated_clips)} scene clips prepared.", 0.95, {"clips": generated_clips})

    # -------------------------------------------------------------------------
    # STEP 9: VIDEO STITCHING & POST-PROCESSING (FFmpeg)
    # -------------------------------------------------------------------------
    final_promo_file = run_dir / "final_promo.mp4"
    if resume_run_id and final_promo_file.exists():
        print(f"\n[Step 9: Stitching Pipeline] Skipped (found {final_promo_file.name})")
        if progress_callback:
            progress_callback("Video", "Final stitched promo video ready (Cached).", 1.0, {"promo_file": str(final_promo_file)})
    else:
        print("\n" + "=" * 65)
        print("STEP 9: VIDEO STITCHING & MASTERING (FFmpeg)")
        print("=" * 65)
        if progress_callback:
            progress_callback("Video", "Concatenating clips, mixing audio & burning subtitles...", 0.97, None)
        make_promo(
            clips_dir=clips_dir,
            voiceover_path=voiceover_file,
            subtitle_path=srt_file,
            out_path=final_promo_file,
        )
        if progress_callback:
            progress_callback("Video", "Final stitched promo video ready!", 1.0, {"promo_file": str(final_promo_file)})

    # Final summary display
    _print_final_summary(
        run_id=run_id,
        idea=idea,
        tone=tone,
        length=length,
        script_file=script_file,
        tools_file=tools_file,
        promo_file=final_promo_file,
        storyboard_file=storyboard_file,
    )

    return {
        "run_id": run_id,
        "script_file": str(script_file),
        "tools_file": str(tools_file),
        "storyboard_file": str(storyboard_file),
        "promo_file": str(final_promo_file),
        "clips": [str(c) for c in generated_clips],
    }


def _print_final_summary(
    run_id: str,
    idea: str,
    tone: str,
    length: str,
    script_file: Path,
    tools_file: Path,
    promo_file: Optional[Path],
    storyboard_file: Path,
) -> None:
    """Print the final executive summary with artifact locations."""
    print("\n" + "=" * 75)
    print("                 MULTI-AGENT WORKFLOW COMPLETED!")
    print("=" * 75)
    print(f"  Run ID:        {run_id}")
    print(f"  Concept:       {idea}")
    print(f"  Tone / Length: {tone.capitalize()} | {length}")
    print("-" * 75)
    print("  DELIVERABLE ARTIFACT LOCATIONS:")
    print(f"  [1] Script:       {script_file.resolve()}")
    print(f"  [2] Tools List:   {tools_file.resolve()}")
    print(f"  [3] Storyboard:   {storyboard_file.resolve()}")
    if promo_file and promo_file.exists():
        size_mb = promo_file.stat().st_size / (1024 * 1024)
        print(f"  [4] Promo Video:  {promo_file.resolve()} ({size_mb:.2f} MB)")
    else:
        print("  [4] Promo Video:  [SKIPPED via --no-video]")
    print("-" * 75)
    print(f"  Token Cost Log:  {(OUTPUTS_DIR / 'cost_log.csv').resolve()}")
    print("=" * 75 + "\n")


def main() -> None:
    """CLI entrypoint: executes banner, argument parsing, and workflow execution."""
    print_banner()
    args = parse_arguments()
    try:
        run_pipeline(
            idea=args.idea,
            tone=args.tone,
            length=args.length,
            resume_run_id=args.resume,
            no_video=args.no_video,
            dry_run=args.dry_run,
            mode=args.mode,
            auto_approve=args.auto_approve,
        )
    except KeyboardInterrupt:
        print("\n\n[ABORTED] Workflow interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
