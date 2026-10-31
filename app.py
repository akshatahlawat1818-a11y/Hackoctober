"""Streamlit Web Application for Multi-Agent AI Script & Promo Generator.

Features:
  - Sidebar: Project idea, tone, video length, mode, dry-run toggle, credentials check.
  - Step 1: Interactive Topic Approval cards with Add, Delete, and Regenerate.
  - Step 2: Real-time non-blocking agent progress with polling, status rows, and critic round expanders.
  - Step 3: Comprehensive deliverables dashboard with final script, tools list, critic chart,
            video player, download buttons, and cost estimation.
  - Past Runs: Inspect past run outputs and resume incomplete workflows.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Optional
import uuid

import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import API_CONFIG, OUTPUTS_DIR, get_run_dir
from agents import plan_topics
from main import run_pipeline
from pipeline.voice import find_binary
from utils.cost import calculate_run_cost
from utils.jobs import ACTIVE_JOBS, get_job, has_active_thread, set_job
from utils.run_manager import RunManager

AGENT_ORDER = [
    "Planner",
    "Researcher",
    "Writer",
    "Critic",
    "Tools",
    "Voiceover",
    "Video",
]

AGENT_ICONS = {
    "Planner": "🧠",
    "Researcher": "🌐",
    "Writer": "✍️",
    "Critic": "⚖️",
    "Tools": "🛠️",
    "Voiceover": "🎙️",
    "Video": "🎬",
}

# -----------------------------------------------------------------------------
# Page Configuration & Clean Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Script & Promo Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E88E5 0%, #7E57C2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .main-subtitle {
        color: #555;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .topic-card {
        border-radius: 8px;
        padding: 14px 18px;
        background-color: #f8f9fb;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .agent-row {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 6px;
        background-color: #f8fafc;
        border: 1px solid #edf2f7;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def format_friendly_error(exc: Exception) -> str:
    """Format exceptions into friendly, actionable error messages."""
    msg = str(exc)
    lower = msg.lower()

    if any(k in lower for k in ["429", "quota", "resourceexhausted", "rate limit", "ratelimit"]):
        return (
            "⏳ **API Rate Limit Exceeded**: The AI provider is temporarily throttling requests. "
            "Please wait a minute and click **Resume** to continue without losing progress."
        )
    if any(k in lower for k in ["api_key", "api key", "unauthenticated", "invalid api key", "credential"]):
        return (
            "🔑 **API Key Authentication Error**: One of your API keys is missing or invalid. "
            "Please check `GEMINI_API_KEY` and `GROQ_API_KEY` in your `.env` file."
        )
    if any(k in lower for k in ["ffmpeg", "ffprobe", "executable file not found"]):
        return (
            "🎥 **FFmpeg Not Available**: FFmpeg is required for assembling video and audio. "
            "Install FFmpeg via `winget install Gyan.FFmpeg` or install from https://ffmpeg.org."
        )
    return f"⚠️ **Execution Error**: {msg}"


def load_run_artifacts(run_id: str) -> dict[str, Any]:
    """Load all artifact deliverables from disk for a given run ID."""
    run_dir = get_run_dir(run_id)
    artifacts: dict[str, Any] = {"run_id": run_id}

    # User input
    input_file = run_dir / "00_user_input.json"
    if input_file.exists():
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                artifacts["user_input"] = json.load(f)
        except Exception:
            artifacts["user_input"] = {}

    # Topics
    topics_file = run_dir / "topics.json"
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as f:
                artifacts["topics"] = json.load(f)
        except Exception:
            artifacts["topics"] = []

    # Research
    research_file = run_dir / "research.json"
    if research_file.exists():
        try:
            with open(research_file, "r", encoding="utf-8") as f:
                artifacts["research"] = json.load(f)
        except Exception:
            artifacts["research"] = []

    # Script
    script_file = run_dir / "final_script.md"
    if script_file.exists():
        try:
            artifacts["script"] = script_file.read_text(encoding="utf-8")
        except Exception:
            artifacts["script"] = ""

    # Script result metadata
    script_res_file = run_dir / "script_result.json"
    if script_res_file.exists():
        try:
            with open(script_res_file, "r", encoding="utf-8") as f:
                artifacts["script_result"] = json.load(f)
        except Exception:
            artifacts["script_result"] = {}

    # Tools
    tools_file = run_dir / "tools.json"
    if tools_file.exists():
        try:
            with open(tools_file, "r", encoding="utf-8") as f:
                artifacts["tools"] = json.load(f)
        except Exception:
            artifacts["tools"] = []

    # Storyboard
    storyboard_file = run_dir / "storyboard.json"
    if storyboard_file.exists():
        try:
            with open(storyboard_file, "r", encoding="utf-8") as f:
                artifacts["storyboard"] = json.load(f)
        except Exception:
            artifacts["storyboard"] = {}

    # Media paths
    promo_file = run_dir / "final_promo.mp4"
    artifacts["promo_video"] = str(promo_file) if promo_file.exists() else None

    voice_file = run_dir / "voiceover.mp3"
    artifacts["voiceover"] = str(voice_file) if voice_file.exists() else None

    # Critic draft evaluations
    drafts_dir = run_dir / "drafts"
    critiques = []
    if drafts_dir.exists():
        for cf in sorted(drafts_dir.glob("critique_*.json")):
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    r_num = cf.stem.replace("critique_", "")
                    critiques.append({"round": r_num, "data": cdata})
            except Exception:
                pass
    artifacts["critiques"] = critiques

    # Cost calculation
    since_ts = artifacts.get("user_input", {}).get("recorded_at")
    artifacts["cost"] = calculate_run_cost(since_timestamp=since_ts)

    return artifacts


def start_pipeline_job(
    run_id: str,
    idea: str,
    tone: str,
    length: str,
    dry_run: bool,
    mode: str,
    resume: bool = False,
) -> None:
    """Launch the multi-agent pipeline in a non-blocking background thread."""
    if has_active_thread(run_id):
        return

    # Initialize job tracking structure
    job_state: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "progress": 0.15,
        "current_agent": "Researcher" if resume else "Planner",
        "status_message": "Initializing multi-agent pipeline...",
        "agent_states": {
            agent: {
                "status": "pending",
                "message": "Waiting...",
            }
            for agent in AGENT_ORDER
        },
        "error": None,
        "results": None,
        "started_at": datetime.now().isoformat(),
    }

    # If topics already exist on disk, mark Planner as done
    run_dir = get_run_dir(run_id)
    if (run_dir / "topics.json").exists():
        job_state["agent_states"]["Planner"] = {
            "status": "done",
            "message": "Research topics approved.",
        }

    ACTIVE_JOBS[run_id] = job_state

    def progress_callback(
        agent_name: str,
        message: str,
        progress_val: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Invoked by pipeline steps to update live progress state."""
        if run_id not in ACTIVE_JOBS:
            return

        state = ACTIVE_JOBS[run_id]
        state["progress"] = min(1.0, max(0.0, progress_val))
        state["current_agent"] = agent_name
        state["status_message"] = message

        if agent_name in state["agent_states"]:
            # If completing or running
            is_done = progress_val >= 1.0 or "ready" in message.lower() or "complete" in message.lower() or "constructed" in message.lower()
            state["agent_states"][agent_name]["status"] = "done" if is_done else "running"
            state["agent_states"][agent_name]["message"] = message

    def worker_thread() -> None:
        """Background thread target function."""
        try:
            results = run_pipeline(
                idea=idea,
                tone=tone,
                length=length,
                resume_run_id=run_id,
                dry_run=dry_run,
                mode=mode,
                auto_approve=True,
                progress_callback=progress_callback,
            )
            # Mark all agents as done
            if run_id in ACTIVE_JOBS:
                for a in AGENT_ORDER:
                    ACTIVE_JOBS[run_id]["agent_states"][a]["status"] = "done"
                ACTIVE_JOBS[run_id]["progress"] = 1.0
                ACTIVE_JOBS[run_id]["status"] = "completed"
                ACTIVE_JOBS[run_id]["results"] = results
                ACTIVE_JOBS[run_id]["status_message"] = "Multi-agent workflow completed successfully!"
        except Exception as exc:
            if run_id in ACTIVE_JOBS:
                ACTIVE_JOBS[run_id]["status"] = "error"
                ACTIVE_JOBS[run_id]["error"] = format_friendly_error(exc)
                ACTIVE_JOBS[run_id]["status_message"] = f"Failed: {exc}"

    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()
    ACTIVE_JOBS[run_id]["thread"] = t


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # Stages: 'input' -> 'approval' -> 'running' -> 'results'

if "run_id" not in st.session_state:
    st.session_state.run_id = None

if "idea" not in st.session_state:
    st.session_state.idea = "AI app that turns messy college notes into viral study quizzes"

if "tone" not in st.session_state:
    st.session_state.tone = "humorous"

if "length" not in st.session_state:
    st.session_state.length = "60s"

if "mode" not in st.session_state:
    st.session_state.mode = "Dry run"

if "dry_run" not in st.session_state:
    st.session_state.dry_run = True

if "topics" not in st.session_state:
    st.session_state.topics = []

if "results" not in st.session_state:
    st.session_state.results = {}

if "error_message" not in st.session_state:
    st.session_state.error_message = None


def reset_workflow() -> None:
    """Reset the current session to input stage for a new project."""
    st.session_state.stage = "input"
    st.session_state.run_id = None
    st.session_state.topics = []
    st.session_state.results = {}
    st.session_state.error_message = None
    st.rerun()


# -----------------------------------------------------------------------------
# SIDEBAR: Configuration & Environment Setup
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    # Project Idea Text Area
    sidebar_idea = st.text_area(
        "Project / Video Idea:",
        value=st.session_state.idea,
        height=100,
        placeholder="e.g. AI tool that summarizes YouTube tutorials into interactive code playgrounds",
        help="The core project concept explored by the Planner Agent.",
    )
    st.session_state.idea = sidebar_idea

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        # Tone Selector
        tone_options = ["humorous", "serious", "inspirational"]
        tone_idx = tone_options.index(st.session_state.tone) if st.session_state.tone in tone_options else 0
        sidebar_tone = st.selectbox(
            "Tone:",
            options=tone_options,
            index=tone_idx,
            help="Writing & voiceover narrative pacing.",
        )
        st.session_state.tone = sidebar_tone

    with col_s2:
        # Video Length
        length_options = ["30s", "60s"]
        length_idx = length_options.index(st.session_state.length) if st.session_state.length in length_options else 1
        sidebar_length = st.selectbox(
            "Length:",
            options=length_options,
            index=length_idx,
            help="Target video script duration.",
        )
        st.session_state.length = sidebar_length

    # Mode Selector
    mode_options = ["Dry run", "Draft", "Final"]
    mode_idx = mode_options.index(st.session_state.mode) if st.session_state.mode in mode_options else 0
    sidebar_mode = st.selectbox(
        "Mode:",
        options=mode_options,
        index=mode_idx,
        help="Draft: Veo 3.1 Lite (720p). Final: Veo 3.1 Standard (1080p).",
    )
    st.session_state.mode = sidebar_mode

    # Dry-Run Toggle (ON by default)
    sidebar_dry_run = st.toggle(
        "⚡ Dry run (Free FFmpeg dummy clips)",
        value=st.session_state.dry_run,
        help="When enabled, produces video clips locally with FFmpeg at zero API cost.",
    )
    st.session_state.dry_run = sidebar_dry_run

    if sidebar_mode in ["Draft", "Final"] and not sidebar_dry_run:
        st.info("ℹ️ **Note**: Draft & Final modes require Google Cloud Vertex AI (Veo 3.1) credentials.")

    st.markdown("---")
    st.markdown("### 🔑 Credentials & Environment")

    has_gemini = bool(API_CONFIG.gemini_api_key)
    has_groq = bool(API_CONFIG.groq_api_key)
    ffmpeg_detected = bool(find_binary("ffmpeg") != "ffmpeg" or shutil.which("ffmpeg"))

    # Status indicators (Never show or log the actual API key values)
    st.markdown(f"- **Gemini API:** {'🟢 Configured' if has_gemini else '🔴 Missing (.env)'}")
    st.markdown(f"- **Groq API:** {'🟢 Configured' if has_groq else '🔴 Missing (.env)'}")
    st.markdown(f"- **FFmpeg Binary:** {'🟢 Detected' if ffmpeg_detected else '🔴 Not found'}")

    if not has_gemini or not has_groq:
        st.warning(
            "⚠️ **Missing API Keys**\n\n"
            "Live LLM calls require API keys in your `.env` file:\n"
            "1. **Gemini**: Get a key at [Google AI Studio](https://aistudio.google.com)\n"
            "2. **Groq**: Get a key at [Groq Console](https://console.groq.com)\n"
            "3. Add to `.env`:\n"
            "```bash\n"
            "GEMINI_API_KEY=your_key_here\n"
            "GROQ_API_KEY=your_key_here\n"
            "```\n"
            "*(If missing, automated test simulation responses will be used.)*"
        )

    if not ffmpeg_detected:
        st.warning(
            "⚠️ **FFmpeg Not Found**\n\n"
            "FFmpeg is needed for voiceover timing and video stitching:\n"
            "- Windows: `winget install Gyan.FFmpeg`\n"
            "- Or download from https://ffmpeg.org"
        )

    st.markdown("---")
    if st.button("🔄 Reset Project", use_container_width=True):
        reset_workflow()


# -----------------------------------------------------------------------------
# MAIN APP HEADER & TOP NAVIGATION TABS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎬 AI Multi-Agent Script & Promo Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Autonomous multi-agent pipeline: Planning ➔ Research ➔ Writer & Critic Loop ➔ Tech Stack ➔ Promo Video</div>',
    unsafe_allow_html=True,
)

tab_workflow, tab_past_runs = st.tabs(["🚀 Generator Workflow", "📁 Past Runs & Resumption"])

# =============================================================================
# TAB 1: GENERATOR WORKFLOW (STEPS 1, 2, AND 3)
# =============================================================================
with tab_workflow:
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    # -------------------------------------------------------------------------
    # STEP 1: TOPIC APPROVAL
    # -------------------------------------------------------------------------
    if st.session_state.stage in ["input", "approval"]:
        st.markdown("### 📋 Step 1: Research Topics Planning & Approval")

        # Initial generate button if no topics yet
        if not st.session_state.topics:
            st.info("Click **Generate topics** to have the **Planner Agent (Gemini)** analyze your project idea and propose research angles.")
            if st.button("✨ Generate topics", type="primary", use_container_width=True):
                if not st.session_state.idea.strip():
                    st.warning("Please provide a project idea in the sidebar before generating topics.")
                else:
                    with st.spinner("Planner Agent is breaking down your concept into 5-8 research angles..."):
                        try:
                            # Map 30s/60s to script length text
                            length_mapped = "30 seconds" if st.session_state.length == "30s" else "60 seconds"
                            topics = plan_topics(
                                idea=st.session_state.idea,
                                tone=st.session_state.tone,
                                length=length_mapped,
                            )
                            # Ensure unique ID for stable UI card editing
                            for idx, t in enumerate(topics):
                                if "id" not in t:
                                    t["id"] = f"topic_{idx}_{uuid.uuid4().hex[:4]}"

                            # Initialize run ID and save initial input
                            run_mgr = RunManager()
                            st.session_state.run_id = run_mgr.run_id
                            run_mgr.save_user_input(
                                idea=st.session_state.idea,
                                tone=st.session_state.tone,
                                length=length_mapped,
                                mode=st.session_state.mode,
                                dry_run=st.session_state.dry_run,
                            )

                            st.session_state.topics = topics
                            st.session_state.stage = "approval"
                            st.session_state.error_message = None
                            st.rerun()
                        except Exception as exc:
                            st.error(format_friendly_error(exc))

        else:
            # Topic Approval Interface with editable cards
            st.write(
                f"**Review & customize research angles** ({len(st.session_state.topics)} topics). "
                "Edit any field, remove topics with the delete button, or add your own angles."
            )

            to_delete_idx = None
            current_topics = list(st.session_state.topics)

            for idx, topic in enumerate(current_topics):
                t_id = topic.get("id", f"top_{idx}")
                with st.expander(f"Topic {idx + 1}: {topic.get('title', 'Untitled')}", expanded=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        new_title = st.text_input("Title:", value=topic.get("title", ""), key=f"title_{t_id}")
                        new_why = st.text_area("Why it matters:", value=topic.get("why_it_matters", ""), key=f"why_{t_id}", height=65)
                        queries_str = ", ".join(topic.get("search_queries", []))
                        new_queries = st.text_input("Search Queries (comma-separated):", value=queries_str, key=f"q_{t_id}")

                        # Update data directly in current topic dictionary
                        topic["title"] = new_title.strip()
                        topic["why_it_matters"] = new_why.strip()
                        topic["search_queries"] = [q.strip() for q in new_queries.split(",") if q.strip()]

                    with c2:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        if st.button("🗑️ Delete", key=f"del_{t_id}"):
                            to_delete_idx = idx

            # Process pending deletion
            if to_delete_idx is not None:
                current_topics.pop(to_delete_idx)
                st.session_state.topics = current_topics
                st.rerun()

            # Add Topic Input
            with st.expander("➕ Add a Custom Topic", expanded=False):
                col_a1, col_a2 = st.columns([1, 1])
                with col_a1:
                    add_title = st.text_input("Topic Title:", key="add_topic_title_input")
                with col_a2:
                    add_queries = st.text_input("Search Queries (comma-separated):", key="add_topic_queries_input")
                add_why = st.text_area("Why it matters:", key="add_topic_why_input", height=60)

                if st.button("Add Topic"):
                    if add_title.strip():
                        new_topic_item = {
                            "id": f"custom_{uuid.uuid4().hex[:6]}",
                            "title": add_title.strip(),
                            "why_it_matters": add_why.strip() or "Custom research angle added by user.",
                            "search_queries": [q.strip() for q in add_queries.split(",") if q.strip()] or [f"{add_title.strip()} overview"],
                        }
                        st.session_state.topics.append(new_topic_item)
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([3, 2, 2])

            with col_b1:
                # "Approve and continue" shown ONLY when at least 1 topic remains
                has_topics = len(st.session_state.topics) >= 1
                if has_topics:
                    if st.button("✅ Approve and continue", type="primary", use_container_width=True):
                        # Ensure run_id exists
                        if not st.session_state.run_id:
                            run_mgr = RunManager()
                            st.session_state.run_id = run_mgr.run_id

                        run_dir = get_run_dir(st.session_state.run_id)
                        # Persist approved topics to run directory
                        with open(run_dir / "topics.json", "w", encoding="utf-8") as f:
                            json.dump(st.session_state.topics, f, indent=2, ensure_ascii=False)

                        # Launch background thread execution
                        start_pipeline_job(
                            run_id=st.session_state.run_id,
                            idea=st.session_state.idea,
                            tone=st.session_state.tone,
                            length="30 seconds" if st.session_state.length == "30s" else "60 seconds",
                            dry_run=st.session_state.dry_run,
                            mode="draft" if st.session_state.mode == "Draft" else ("final" if st.session_state.mode == "Final" else "draft"),
                            resume=False,
                        )
                        st.session_state.stage = "running"
                        st.rerun()
                else:
                    st.warning("⚠️ At least one topic must remain to continue. Click 'Add a Custom Topic' or 'Regenerate'.")

            with col_b2:
                if st.button("🔄 Regenerate", use_container_width=True):
                    with st.spinner("Regenerating research topics with Planner Agent..."):
                        try:
                            length_mapped = "30 seconds" if st.session_state.length == "30s" else "60 seconds"
                            topics = plan_topics(
                                idea=st.session_state.idea,
                                tone=st.session_state.tone,
                                length=length_mapped,
                            )
                            for idx, t in enumerate(topics):
                                t["id"] = f"topic_{idx}_{uuid.uuid4().hex[:4]}"
                            st.session_state.topics = topics
                            st.rerun()
                        except Exception as exc:
                            st.error(format_friendly_error(exc))

            with col_b3:
                if st.button("⬅️ Change Idea", use_container_width=True):
                    st.session_state.stage = "input"
                    st.session_state.topics = []
                    st.rerun()

    # -------------------------------------------------------------------------
    # STEP 2: LIVE NON-FREEZING PROGRESS
    # -------------------------------------------------------------------------
    elif st.session_state.stage == "running":
        run_id = st.session_state.run_id or "default_run"
        st.markdown(f"### ⚙️ Step 2: Multi-Agent Workflow In Progress")
        st.caption(f"Active Run ID: `{run_id}` • Execution running in background thread")

        job = get_job(run_id)

        if not job and not has_active_thread(run_id):
            # If for some reason job not found in ACTIVE_JOBS, attempt to start it
            start_pipeline_job(
                run_id=run_id,
                idea=st.session_state.idea,
                tone=st.session_state.tone,
                length="30 seconds" if st.session_state.length == "30s" else "60 seconds",
                dry_run=st.session_state.dry_run,
                mode="draft" if st.session_state.mode == "Draft" else ("final" if st.session_state.mode == "Final" else "draft"),
                resume=True,
            )
            job = get_job(run_id)

        # Progress bar
        curr_progress = job.get("progress", 0.15) if job else 0.15
        st.progress(curr_progress)

        status_text = job.get("status_message", "Executing multi-agent pipeline...") if job else "Running..."
        st.info(f"**Current Activity:** {status_text}")

        # Status Line per Agent
        st.markdown("#### Agent Status Pipeline")
        agent_states = job.get("agent_states", {}) if job else {}

        cols_agent = st.columns(len(AGENT_ORDER))
        for i, agent_name in enumerate(AGENT_ORDER):
            state_info = agent_states.get(agent_name, {"status": "pending", "message": "Waiting..."})
            a_status = state_info.get("status", "pending")
            icon = AGENT_ICONS.get(agent_name, "🤖")

            with cols_agent[i]:
                if a_status == "done":
                    st.markdown(f"**{icon} {agent_name}**  \n`✅ Complete`")
                elif a_status == "running":
                    st.markdown(f"**{icon} {agent_name}**  \n`⏳ In Progress`")
                else:
                    st.markdown(f"**{icon} {agent_name}**  \n`⚪ Pending`")

        # Real-time Critic Rounds Inspection
        st.markdown("---")
        st.markdown("#### Critic Agent Scoring & Feedback History")

        # Scan drafts folder for critic rounds
        run_dir = get_run_dir(run_id)
        drafts_dir = run_dir / "drafts"
        critic_files = sorted(drafts_dir.glob("critique_*.json")) if drafts_dir.exists() else []

        if critic_files:
            for cf in critic_files:
                try:
                    with open(cf, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                    round_num = cf.stem.replace("critique_", "")
                    avg_score = c_data.get("average", 0.0)
                    scores = c_data.get("scores", {})
                    feedback = c_data.get("feedback", [])

                    with st.expander(f"⚖️ Critic Round {round_num} — Average Score: {avg_score:.2f} / 10", expanded=True):
                        score_cols = st.columns(len(scores) if scores else 1)
                        for idx, (crit_key, crit_val) in enumerate(scores.items()):
                            if idx < len(score_cols):
                                with score_cols[idx]:
                                    st.metric(crit_key.replace("_", " ").title(), f"{crit_val}/10")

                        st.markdown("**Critic Feedback & Revisions:**")
                        if isinstance(feedback, list):
                            for item in feedback:
                                st.markdown(f"- {item}")
                        else:
                            st.write(feedback)
                except Exception:
                    pass
        else:
            st.caption("Critic evaluations will appear here once the Writer & Critic loop executes.")

        # Polling / Status Check
        if job and job.get("status") == "completed":
            st.success("🎉 Pipeline execution completed! Redirecting to Results...")
            # Load deliverables into session state
            st.session_state.results = load_run_artifacts(run_id)
            st.session_state.stage = "results"
            time.sleep(1)
            st.rerun()

        elif job and job.get("status") == "error":
            err_msg = job.get("error", "An unknown error occurred.")
            st.error(err_msg)
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if st.button("🔄 Retry / Resume Workflow", type="primary", use_container_width=True):
                    start_pipeline_job(
                        run_id=run_id,
                        idea=st.session_state.idea,
                        tone=st.session_state.tone,
                        length="30 seconds" if st.session_state.length == "30s" else "60 seconds",
                        dry_run=st.session_state.dry_run,
                        mode="draft" if st.session_state.mode == "Draft" else "draft",
                        resume=True,
                    )
                    st.rerun()
            with col_e2:
                if st.button("⬅️ Back to Topic Approval", use_container_width=True):
                    st.session_state.stage = "approval"
                    st.rerun()

        else:
            # Poll every 1 second without freezing UI
            time.sleep(1.0)
            st.rerun()

    # -------------------------------------------------------------------------
    # STEP 3: RESULTS DASHBOARD & DELIVERABLES
    # -------------------------------------------------------------------------
    elif st.session_state.stage == "results":
        run_id = st.session_state.run_id or "default_run"
        results = st.session_state.results or load_run_artifacts(run_id)

        st.markdown(f"### 🎉 Step 3: Generation Results & Deliverables")

        # Top Metric Bar: Run ID & Cost Estimate from cost_log.csv
        cost_info = results.get("cost", calculate_run_cost())
        total_cost = cost_info.get("total_cost_usd", 0.0)
        total_tokens = cost_info.get("total_tokens", 0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Run ID", run_id)
        with m2:
            st.metric("Estimated Cost (USD)", f"${total_cost:.4f}", help="Calculated from outputs/cost_log.csv")
        with m3:
            st.metric("Total Tokens Logged", f"{total_tokens:,} tok")
        with m4:
            script_status = results.get("script_result", {}).get("status", "PASSED").upper()
            st.metric("Critic Status", script_status)

        # Tabs for Deliverables
        res_tab_script, res_tab_tools, res_tab_critic, res_tab_video = st.tabs([
            "📜 Final Script",
            "🛠️ Tools & Stack",
            "⚖️ Critic Score History",
            "🎬 Promo Video",
        ])

        # Tab: Final Script
        with res_tab_script:
            st.subheader("Final Script")
            script_text = results.get("script", "")
            if script_text:
                st.markdown(script_text)
                st.download_button(
                    label="📥 Download Script (.md)",
                    data=script_text,
                    file_name=f"{run_id}_final_script.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("No finalized script found for this run.")

        # Tab: Tools List with Source Links
        with res_tab_tools:
            st.subheader("Required APIs, Tools & Source Links")
            tools_data = results.get("tools", [])

            if tools_data:
                # Format into a clean display table with markdown source links
                tools_display = []
                for t in tools_data:
                    link_url = t.get("link", "")
                    link_markdown = f"[{link_url}]({link_url})" if link_url else "—"
                    setup_str = ", ".join(t.get("setup_steps", [])) if isinstance(t.get("setup_steps"), list) else str(t.get("setup_steps", ""))
                    tools_display.append({
                        "Tool / API": t.get("name", "Tool"),
                        "Purpose": t.get("purpose", ""),
                        "Cost": t.get("cost", "Free"),
                        "Source Link": link_markdown,
                        "Setup Steps": setup_str,
                    })

                df_tools = pd.DataFrame(tools_display)
                st.markdown(df_tools.to_markdown(index=False), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Tools List (.json)",
                    data=json.dumps(tools_data, indent=2),
                    file_name=f"{run_id}_tools.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("No tools list generated for this run.")

        # Tab: Critic Score History (Small Chart & Expander)
        with res_tab_critic:
            st.subheader("Critic Score History Across Rounds")
            critiques = results.get("critiques", [])

            if critiques:
                # Construct data table for the small chart
                chart_rows = []
                for c in critiques:
                    r_num = c.get("round", "1")
                    cdata = c.get("data", {})
                    scores = cdata.get("scores", {})
                    row = {
                        "Round": f"Round {r_num}",
                        "Average": cdata.get("average", 0.0),
                    }
                    for k, v in scores.items():
                        row[k.replace("_", " ").title()] = v
                    chart_rows.append(row)

                df_chart = pd.DataFrame(chart_rows)

                # Show small line chart of scores
                st.markdown("**Rubric Metric Progression:**")
                chart_cols = [c for c in df_chart.columns if c != "Round"]
                st.line_chart(df_chart.set_index("Round")[chart_cols], height=260)

                # Expanders for each round's feedback
                st.markdown("#### Detailed Round Breakdowns")
                for c in critiques:
                    r_num = c.get("round", "1")
                    cdata = c.get("data", {})
                    scores = cdata.get("scores", {})
                    feedback = cdata.get("feedback", [])

                    with st.expander(f"Round {r_num} Critic Rubric (Average: {cdata.get('average', 0.0):.2f}/10)", expanded=True):
                        st.json(scores)
                        st.markdown("**Revision Feedback:**")
                        if isinstance(feedback, list):
                            for item in feedback:
                                st.markdown(f"- {item}")
                        else:
                            st.write(feedback)
            else:
                st.info("No critic rounds recorded for this run.")

        # Tab: Final Video with st.video
        with res_tab_video:
            st.subheader("Final Promotional Video")
            promo_path = results.get("promo_video")

            if promo_path and Path(promo_path).exists():
                st.video(promo_path)

                with open(promo_path, "rb") as vf:
                    video_bytes = vf.read()

                st.download_button(
                    label="📥 Download Video (.mp4)",
                    data=video_bytes,
                    file_name=f"{run_id}_promo.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                )
            else:
                st.info("Video generation output not found or video was skipped for this run.")

            # Optional Audio Player if voiceover exists
            voice_path = results.get("voiceover")
            if voice_path and Path(voice_path).exists():
                st.markdown("#### Synthesized Voiceover Track")
                st.audio(voice_path)

        st.markdown("<br><hr>", unsafe_allow_html=True)
        if st.button("🚀 Start a New Project", type="primary"):
            reset_workflow()


# =============================================================================
# TAB 2: PAST RUNS & RESUMPTION
# =============================================================================
with tab_past_runs:
    st.markdown("### 📁 Past Workflow Runs")
    st.caption("Inspect artifacts from previous pipeline runs or resume interrupted workflows.")

    # List all run subdirectories in OUTPUTS_DIR
    run_folders = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    run_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not run_folders:
        st.info("No past runs found in `outputs/` directory.")
    else:
        for rf in run_folders:
            folder_name = rf.name
            user_input_file = rf / "00_user_input.json"
            script_file = rf / "final_script.md"
            promo_file = rf / "final_promo.mp4"
            topics_file = rf / "topics.json"

            # Determine run metadata and completion status
            run_idea = "Unknown Concept"
            run_tone = "serious"
            run_len = "60s"
            run_ts = datetime.fromtimestamp(rf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

            if user_input_file.exists():
                try:
                    with open(user_input_file, "r", encoding="utf-8") as f:
                        ui_data = json.load(f)
                        run_idea = ui_data.get("idea", run_idea)
                        run_tone = ui_data.get("tone", run_tone)
                        run_len = ui_data.get("length", run_len)
                        if "recorded_at" in ui_data:
                            run_ts = ui_data["recorded_at"][:16].replace("T", " ")
                except Exception:
                    pass

            is_complete = promo_file.exists()
            has_script = script_file.exists()
            is_unfinished = not is_complete

            status_badge = "🟢 Completed" if is_complete else ("🟡 Script Finalized" if has_script else "⚪ Incomplete")

            with st.container():
                st.markdown(
                    f"""
                    <div class="topic-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 1.1rem;">Run: <code>{folder_name}</code></strong>
                            <span style="font-weight: 600;">{status_badge}</span>
                        </div>
                        <div style="color: #444; margin-top: 5px;"><strong>Concept:</strong> {run_idea}</div>
                        <div style="font-size: 0.85rem; color: #777; margin-top: 4px;">
                            Tone: <em>{run_tone}</em> • Target: <em>{run_len}</em> • Created: <em>{run_ts}</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if st.button("👁️ Open Results", key=f"open_{folder_name}", use_container_width=True):
                        st.session_state.run_id = folder_name
                        st.session_state.results = load_run_artifacts(folder_name)
                        st.session_state.stage = "results"
                        st.rerun()

                with btn_col2:
                    if is_unfinished:
                        if st.button("▶️ Resume Run (--resume)", key=f"resume_{folder_name}", type="primary", use_container_width=True):
                            st.session_state.run_id = folder_name
                            st.session_state.idea = run_idea
                            st.session_state.tone = run_tone
                            st.session_state.length = run_len
                            # Start pipeline in resume mode
                            start_pipeline_job(
                                run_id=folder_name,
                                idea=run_idea,
                                tone=run_tone,
                                length=run_len,
                                dry_run=True,
                                mode="draft",
                                resume=True,
                            )
                            st.session_state.stage = "running"
                            st.rerun()
                    else:
                        st.button("✅ Run Finished", key=f"fin_{folder_name}", disabled=True, use_container_width=True)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
