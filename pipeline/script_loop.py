"""Script Generation and Critique Revision Loop using LangGraph.

Coordinates the iterative cycle between Writer Agent (Gemini) and Critic Agent (Groq)
with rubric scoring, retry thresholds, best-effort fallback, and draft artifact caching.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.critic import critique_script
from agents.writer import write_script
import config
from config import CRITIC_CONFIG, OUTPUTS_DIR, get_run_dir


class ScriptLoopState(TypedDict):
    """LangGraph state representation for the script writing and critique loop."""

    idea: str
    tone: str
    length: str
    research: list[dict[str, Any]]
    run_id: str
    round: int
    script: str
    critique: dict[str, Any]
    scores: dict[str, float]
    average: float
    best_script: str
    best_scores: dict[str, float]
    best_average: float
    best_round: int
    status: str  # "in_progress", "passed", "best_effort"


def get_drafts_dir(run_id: str) -> Path:
    """Return and create outputs/<run_id>/drafts/ folder."""
    run_dir = get_run_dir(run_id) if run_id else OUTPUTS_DIR
    drafts_dir = run_dir / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir


# ==============================================================================
# LangGraph Nodes
# ==============================================================================


def write_node(state: ScriptLoopState) -> dict[str, Any]:
    """Writer node: generates initial draft or rewrites based on critic feedback."""
    current_round = state.get("round", 0) + 1
    run_id = state.get("run_id", "default_run")
    drafts_dir = get_drafts_dir(run_id)

    feedback = None
    if current_round > 1:
        feedback = state.get("critique", {}).get("feedback", [])
        print(f"\n[Round {current_round}] Revising script to address critic feedback...")
    else:
        print(f"\n[Round {current_round}] Drafting initial script for idea: '{state['idea']}'...")

    script = write_script(
        idea=state["idea"],
        tone=state.get("tone", "serious"),
        length=state.get("length", "medium"),
        research=state.get("research", []),
        feedback=feedback,
    )

    # Save draft file: outputs/<run_id>/drafts/draft_<round>.md
    draft_file = drafts_dir / f"draft_{current_round}.md"
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Saved draft to: {draft_file}")

    return {
        "script": script,
        "round": current_round,
        "status": "in_progress",
    }


def critique_node(state: ScriptLoopState) -> dict[str, Any]:
    """Critic node: scores the script with Groq Llama and calculates metric averages."""
    current_round = state["round"]
    run_id = state.get("run_id", "default_run")
    drafts_dir = get_drafts_dir(run_id)

    print(f"[Round {current_round}] Evaluating script against rubric with Critic Agent (Groq)...")
    critique = critique_script(
        script=state["script"],
        tone=state.get("tone", "serious"),
        research=state.get("research", []),
    )

    scores = critique.get("scores", {})
    average = critique.get("average", 0.0)

    # Save critique file: outputs/<run_id>/drafts/critique_<round>.json
    critique_file = drafts_dir / f"critique_{current_round}.json"
    with open(critique_file, "w", encoding="utf-8") as f:
        json.dump(critique, f, indent=2, ensure_ascii=False)

    # Print round score breakdown
    criteria_str = ", ".join(f"{k}: {v}" for k, v in scores.items())
    print(f"  [Round {current_round} Score] Average: {average:.2f}/10")
    print(f"  Breakdown: {criteria_str}")

    # Track highest-scoring version
    best_script = state.get("best_script", "")
    best_average = state.get("best_average", -1.0)
    best_scores = state.get("best_scores", {})
    best_round = state.get("best_round", 1)

    if average >= best_average or not best_script:
        best_script = state["script"]
        best_average = average
        best_scores = scores
        best_round = current_round

    return {
        "critique": critique,
        "scores": scores,
        "average": average,
        "best_script": best_script,
        "best_average": best_average,
        "best_scores": best_scores,
        "best_round": best_round,
    }


def decide_routing(state: ScriptLoopState) -> str:
    """Evaluate whether the script meets passing thresholds or needs revision."""
    average = state.get("average", 0.0)
    scores = state.get("scores", {})
    current_round = state.get("round", 1)

    pass_score = CRITIC_CONFIG.pass_score
    min_criterion = CRITIC_CONFIG.min_criterion_score
    max_retries = CRITIC_CONFIG.max_retries

    # Check passing conditions
    no_low_scores = all(val >= min_criterion for val in scores.values()) if scores else False
    passed = (average >= pass_score) and no_low_scores

    if passed:
        print(f"\n>> Script PASSED rubric on Round {current_round}! (Avg: {average:.2f} >= {pass_score}, min score >= {min_criterion})")
        return "finish"

    # Retries check: Round 1 is initial, followed by up to max_retries revisions
    retries_used = current_round - 1
    if retries_used >= max_retries:
        print(f"\n>> Maximum retries ({max_retries}) reached. Exiting with highest-scoring draft.")
        return "finish"

    print(f"\n>> Script did not meet pass threshold (Avg: {average:.2f}/{pass_score}, Min: {min_criterion}). Initiating Retry {retries_used + 1}/{max_retries}...")
    return "retry"


def finalize_node(state: ScriptLoopState) -> dict[str, Any]:
    """Finalize script selection and save outputs/<run_id>/final_script.md."""
    run_id = state.get("run_id", "default_run")
    run_dir = get_run_dir(run_id) if run_id else OUTPUTS_DIR

    average = state.get("average", 0.0)
    scores = state.get("scores", {})
    pass_score = CRITIC_CONFIG.pass_score
    min_criterion = CRITIC_CONFIG.min_criterion_score

    all_met = all(v >= min_criterion for v in scores.values()) if scores else False
    has_passed = (average >= pass_score) and all_met

    if has_passed:
        final_script = state["script"]
        final_scores = state["scores"]
        final_average = state["average"]
        final_status = "passed"
        selected_round = state["round"]
    else:
        final_script = state.get("best_script", state["script"])
        final_scores = state.get("best_scores", state["scores"])
        final_average = state.get("best_average", state["average"])
        final_status = "best_effort"
        selected_round = state.get("best_round", state["round"])

    # Save final script to outputs/<run_id>/final_script.md
    final_script_path = run_dir / "final_script.md"
    with open(final_script_path, "w", encoding="utf-8") as f:
        f.write(final_script)

    # Save summary metadata to outputs/<run_id>/script_result.json
    result_metadata = {
        "run_id": run_id,
        "status": final_status,
        "selected_round": selected_round,
        "total_rounds": state["round"],
        "average_score": final_average,
        "scores": final_scores,
        "completed_at": datetime.now().isoformat(),
    }
    with open(run_dir / "script_result.json", "w", encoding="utf-8") as f:
        json.dump(result_metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"SCRIPT WORKFLOW COMPLETE: Status = [{final_status.upper()}]")
    print(f"  Selected Version: Round {selected_round} (Score: {final_average:.2f}/10)")
    print(f"  Saved script to: {final_script_path}")
    print("=" * 65 + "\n")

    return {
        "script": final_script,
        "scores": final_scores,
        "average": final_average,
        "status": final_status,
    }


# ==============================================================================
# Graph Construction & Orchestration
# ==============================================================================


def build_script_graph() -> Any:
    """Build and compile the LangGraph workflow for script generation and critique."""
    workflow = StateGraph(ScriptLoopState)

    # Add core nodes
    workflow.add_node("write", write_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("finalize", finalize_node)

    # Set flow: Start -> write -> critique -> decide -> (retry: write | finish: finalize) -> END
    workflow.set_entry_point("write")
    workflow.add_edge("write", "critique")
    workflow.add_conditional_edges(
        "critique",
        decide_routing,
        {
            "retry": "write",
            "finish": "finalize",
        },
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_script_loop(
    idea: str,
    tone: str = "serious",
    length: str = "medium",
    research: Optional[list[dict[str, Any]] | dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Execute the iterative script writing and critique revision loop.

    Args:
        idea: Project idea description.
        tone: 'serious' or 'humorous'.
        length: 'short', 'medium', or 'long'.
        research: Research findings and sources.
        run_id: Optional run ID for output folder paths.

    Returns:
        Final state dictionary containing script, scores, average, round, and status.
    """
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"

    # Prepare inputs
    initial_research = research if isinstance(research, list) else ([research] if research else [])

    initial_state: ScriptLoopState = {
        "idea": idea,
        "tone": tone,
        "length": length,
        "research": initial_research,
        "run_id": run_id,
        "round": 0,
        "script": "",
        "critique": {},
        "scores": {},
        "average": 0.0,
        "best_script": "",
        "best_scores": {},
        "best_average": -1.0,
        "best_round": 1,
        "status": "in_progress",
    }

    graph = build_script_graph()
    final_output = graph.invoke(initial_state)
    return final_output
