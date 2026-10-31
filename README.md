# Multi-Agent AI Script & Video Promo Workflow

An end-to-end multi-agent system built with Python, LangGraph, Google Gemini, Groq, Vertex AI Veo 3.1, edge-tts, and FFmpeg. Given a rough idea, tone, and length, this workflow researches topics, drafts and critiques a full script, compiles technical implementation tools, and produces a stitched 20-second commercial promo video.

---

## Workflow Overview

```
User Input (Idea, Tone, Length)
       │
       ▼
1. Planner Agent (Gemini 3.8 Flash) ──► Suggests 5–8 research topics
       │
       ▼
2. Human Approval (CLI / Web UI)    ──► User inspects, edits, adds, or approves topics
       │
       ▼
3. Research Agent                   ──► DuckDuckGo & Wikipedia search with real source URLs
       │
       ▼
4. Writer Agent (Gemini 2.5 Pro)    ──► Drafts full script in specified tone
       │
       ▼
5. Critic Agent (Groq / GPT-OSS 120B) ► Strict 6-criteria rubric scoring (1-10)
       │                                 (Pass: avg >= 9.0 and each criterion >= 8.0)
       │                                 Loops back to Writer with feedback (max 3 retries)
       ├─────────────────────────────────┐
       ▼                                 ▼
6. Tools Agent (Gemini 3.8 Flash)   7. Promo Agent (Gemini 3.8 Flash)
   Compiles APIs, costs, & setup       Creates 4-scene storyboard (4 x 5s) & voiceover text
                                         │
                                         ▼
                                    8. Voiceover Pipeline (edge-tts)
                                       Synthesizes audio & auto-scales tempo with FFmpeg
                                         │
                                         ▼
                                    9. Video Generation (Vertex AI Veo 3.1 / Dry-Run)
                                       Generates 4 clips (Draft Lite 720p or Final 1080p)
                                         │
                                         ▼
                                   10. Video Stitching Pipeline (FFmpeg)
                                       Concatenates clips, replaces audio, mixes BGM,
                                       burns subtitles, and exports 1080p / vertical MP4
                                         │
                                         ▼
                             Outputs in outputs/<run_id>/
                             • final_script.md
                             • tools.json
                             • storyboard.json
                             • final_promo.mp4
```

---

## Directory Structure

```
.
├── agents/                    # Specialized AI agents
│   ├── planner.py             # Topic generation with Gemini
│   ├── researcher.py          # Sequential web research with source verification
│   ├── writer.py              # Long-form script generation
│   ├── critic.py              # Strict 6-criteria rubric evaluation with Groq
│   ├── tools_agent.py         # Required APIs, pricing, and setup steps
│   └── promo_agent.py         # 20-second storyboard (4 scenes x 5s) & voiceover text
├── pipeline/                  # Pipeline execution & media mastering
│   ├── approval.py            # CLI interactive topic approval
│   ├── script_loop.py         # LangGraph state machine with revision loop
│   ├── voice.py               # edge-tts voiceover & FFmpeg duration scaling
│   ├── video_gen.py           # Vertex AI Veo 3.1 generation & FFmpeg dry-run
│   └── stitch.py              # FFmpeg video concat, audio replace + BGM, burned subtitles
├── utils/                     # Shared utilities
│   ├── llm.py                 # Gemini & Groq SDK wrappers with exponential backoff
│   └── run_manager.py         # Step persistence in outputs/<run_id>/
├── outputs/                   # Run directories, artifacts, and cost logs
│   └── cost_log.csv           # Token usage and API cost tracking
├── assets/                    # Static assets (e.g. music.mp3 for background music)
├── tests/                     # 38 unit & integration tests
├── main.py                    # Complete Command-Line Runner
├── app.py                     # Streamlit Interactive Web Application
├── config.py                  # Model IDs, scoring rubrics, pricing, and paths
├── requirements.txt           # Python dependencies
└── README.md                  # System documentation
```

---

## Prerequisites

1. **Python 3.11+** installed on your system.
2. **FFmpeg** installed and accessible in your system `PATH` (needed for video stitching and audio processing).
3. **API Keys & Credentials**:
   - **Google Gemini API Key** (for Planner, Writer, Tools, and Promo agents) from [Google AI Studio](https://aistudio.google.com/).
   - **Groq API Key** (for Critic agent using Llama models) from [Groq Console](https://console.groq.com/keys).
   - **Google Cloud Project** (optional for Vertex AI Veo video generation; dry-run mode works without GCP).

---

## Installation & Setup

### Step 1: Clone and navigate to the project directory
```bash
cd Hackoctober
```

### Step 2: Create and activate a virtual environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure environment variables
Create `.env` from `.env.example`:
- **Windows**: `copy .env.example .env`
- **macOS / Linux**: `cp .env.example .env`

Edit `.env` with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

### Step 5: Install FFmpeg (if not already installed)
- **Windows**: `winget install Gyan.FFmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y ffmpeg`

Verify installation:
```bash
ffmpeg -version
```

---

## How to Run

### 1. Interactive Web Application (`app.py`)

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

The web UI provides:
- **Concept Input**: Text area for your idea, tone selector (`serious` or `humorous`), and script length selector.
- **Topic Editing & Approval**: Review topics suggested by the Planner Agent. Edit titles, adjust search queries, delete unwanted topics, or add custom topics before proceeding.
- **Live Progress Tracker**: Visual progress bar and status updates for research, script revision rounds, tools estimation, storyboard, voiceover, and video rendering.
- **Results Dashboard**:
  - **Script**: Formatted Markdown display with a download button.
  - **Tools & Stack**: Searchable table of required APIs, pricing, setup steps, and download button.
  - **Critic Rubric Scores**: Comparison table of scores across revision rounds (Hook, Clarity, Tone Match, Accuracy, Length Fit, Originality) and revision feedback.
  - **20s Promo Video Player**: Embedded video player (`st.video`), voiceover player (`st.audio`), storyboard scenes, and MP4 download button.
  - **Verified Research**: Inspect key facts with source citations.

---

### 2. Command-Line Runner (`main.py`)

Run the full end-to-end pipeline directly from the command line:

```bash
python main.py
```

#### CLI Options & Flags:
```text
usage: main.py [-h] [--idea IDEA] [--tone {serious,humorous}]
               [--length LENGTH] [--resume RUN_ID] [--no-video] [--dry-run]
               [--mode {draft,final}]

options:
  --idea IDEA           The core project/video concept (prompts interactively if omitted)
  --tone {serious,humorous}
                        Tone of voice for script and promo video (default: serious)
  --length LENGTH       Target script length (default: '60 seconds')
  --resume RUN_ID       Resume an existing run by ID, skipping any steps whose outputs exist on disk
  --no-video            Stop workflow after storyboard & tools list (skips voiceover, video gen, stitch)
  --dry-run             Generate dummy video clips with FFmpeg instead of calling Vertex AI Veo (free)
  --mode {draft,final}  Veo model mode: 'draft' (Lite 720p @ $0.05/s) or 'final' (Standard 1080p @ $0.20/s)
```

#### Example Commands:

1. **Run with local dummy clips (fastest & 100% free test)**:
   ```bash
   python main.py --dry-run
   ```

2. **Run with a specific idea and tone**:
   ```bash
   python main.py --idea "AI app that turns messy college notes into viral study quizzes" --tone humorous --length "60 seconds" --dry-run
   ```

3. **Stop after storyboard and tools list (no video generation)**:
   ```bash
   python main.py --no-video
   ```

4. **Resume an existing run without re-executing completed steps**:
   ```bash
   python main.py --resume run_20261031_183920_abc123 --dry-run
   ```

5. **Full production run with Vertex AI Veo 3.1 video generation**:
   ```bash
   python main.py --mode final
   ```

---

## Output Artifacts

Every run generates an isolated output directory in `outputs/<run_id>/`:

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **User Input** | `00_user_input.json` | Recorded idea, tone, and duration settings |
| **Approved Topics** | `topics.json` | 5–8 approved research topics and search queries |
| **Research Findings** | `research.json` | Verified facts with source URLs and tools mentioned |
| **Final Script** | `final_script.md` | Approved script adhering to rubric criteria |
| **Draft Revisions** | `drafts/draft_*.md` | Intermediate script drafts and critic evaluations |
| **Tools List** | `tools.json` | APIs, estimated costs, setup steps, and links |
| **Storyboard** | `storyboard.json` | 4-scene video prompts and 20s voiceover text |
| **Voiceover Audio** | `voiceover.mp3` | Synthesized voiceover scaled to 20 seconds |
| **Subtitles** | `voiceover.srt` | Word/sentence-level synchronized subtitles |
| **Scene Clips** | `clips/scene_*.mp4` | 4 individual 5-second scene video clips |
| **Final Promo Video** | `final_promo.mp4` | Mastered 20-second promo with voiceover, BGM, and burned-in subtitles |
| **Token Cost Log** | `outputs/cost_log.csv` | Cumulative token counts and estimated costs |

---

## Central Configuration (`config.py`)

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Global Gemini model override via `.env` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Global Groq model override via `.env` |
| `MODELS.planner` | `gemini-3.8-flash` | Gemini model for planning topics |
| `MODELS.research` | `gemini-3.8-flash` | Gemini model for research extraction |
| `MODELS.writer` | `gemini-2.5-pro` | Gemini model for script writing |
| `MODELS.critic` | `openai/gpt-oss-120b` | Groq model for rubric critique |
| `MODELS.tools` | `gemini-3.8-flash` | Gemini model for tools identification |
| `MODELS.promo` | `gemini-3.8-flash` | Gemini model for storyboard generation |
| `VEO_CONFIG.draft_model` | `veo-3.1-lite-generate-001` | Veo 3.1 Lite for 720p draft clips ($0.05/sec) |
| `VEO_CONFIG.final_model` | `veo-3.1-generate-001` | Veo 3.1 Standard for 1080p final clips ($0.20/sec) |
| `CRITIC_CONFIG.pass_score` | `9.0` | Minimum average rubric score to pass |
| `CRITIC_CONFIG.min_criterion_score` | `8.0` | Minimum score for every individual criterion |
| `CRITIC_CONFIG.max_retries` | `3` | Maximum rewrite attempts before selecting highest score |
| `PROMO_CONFIG.total_duration_seconds` | `20` | Promo video duration in seconds |
| `PROMO_CONFIG.resolution` | `1920x1080` | Export resolution (`1920x1080` landscape or `1080x1920` portrait) |

---

## Running the Test Suite

Run all 41 unit and integration tests across the codebase:
```bash
python -m unittest discover tests
```
Or run individual test modules:
```bash
python tests/test_planner.py
python tests/test_researcher.py
python tests/test_script_loop.py
python tests/test_tools_and_promo.py
python tests/test_voice.py
python tests/test_video_gen.py
python tests/test_stitch.py
python tests/test_main.py
```
