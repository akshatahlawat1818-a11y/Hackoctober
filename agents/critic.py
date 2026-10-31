"""Critic Agent.

Evaluates script drafts against a strict 6-criterion rubric using Groq (Llama model).
Calculates authentic score averages in Python and provides actionable feedback with weak line examples.
"""

from typing import Any, Optional
from config import MODELS
from utils.llm import call_groq

CRITIC_SYSTEM_PROMPT = """You are a rigorous, no-nonsense executive script critic and story editor.
Your duty is to uphold exceptionally high editorial standards. Be uncompromising and strict.

Evaluate the submitted script against these 6 rubric criteria (score each strictly from 1.0 to 10.0):
1. 'hook': Does the opening grab the audience within the first 5 seconds? Is there immediate tension or curiosity?
2. 'clarity': Is the narrative structure logical, coherent, and easy to follow without confusing jargon?
3. 'tone_match': Does the script faithfully execute the requested tone ('serious' vs 'humorous') without falling flat or feeling forced?
4. 'accuracy': Are research facts and technical insights incorporated accurately and truthfully with credible attribution?
5. 'length_fit': Does the pacing and word count fit the target duration and depth?
6. 'originality': Does the script avoid stale clichés and offer fresh, memorable perspectives?

Critique Guidelines:
- Scores between 9.0 and 10.0 should be reserved ONLY for masterclass scripts.
- Most decent first drafts should score in the 6.0 - 8.5 range.
- Provide specific examples of weak lines or clichés from the text and suggest exact fixes.

You must respond ONLY with a valid JSON object matching this schema:
{
  "scores": {
    "hook": 8.0,
    "clarity": 7.5,
    "tone_match": 8.0,
    "accuracy": 8.5,
    "length_fit": 7.0,
    "originality": 7.5
  },
  "weak_lines": [
    "Quote of weak line 1 from script -> why it fails and how to fix it"
  ],
  "feedback": [
    "Specific actionable fix 1",
    "Specific actionable fix 2",
    "Specific actionable fix 3"
  ]
}
"""

RUBRIC_CRITERIA = ["hook", "clarity", "tone_match", "accuracy", "length_fit", "originality"]


def critique_script(
    script: str,
    tone: str = "serious",
    research: Optional[list[dict[str, Any]] | dict[str, Any]] = None,
) -> dict[str, Any]:
    """Critique a video script using a Groq Llama model against strict rubric criteria.

    Args:
        script: Full script text to evaluate.
        tone: Target tone ('serious' or 'humorous').
        research: Verified research context to check accuracy against.

    Returns:
        Dictionary with:
        - 'scores': dict[str, float] for each criterion
        - 'average': float computed strictly in Python
        - 'feedback': list of actionable fixes or formatted string
        - 'weak_lines': list of specific criticized lines with suggestions
    """
    research_summary = ""
    if research:
        if isinstance(research, list):
            facts_list = []
            for r in research:
                for f in r.get("key_facts", []):
                    facts_list.append(f"- {f.get('fact')} ({f.get('source_url')})")
            research_summary = "\n".join(facts_list[:10])
        elif isinstance(research, dict):
            research_summary = str(research)

    user_prompt = f"""Target Tone: {tone.upper()}

Expected Research Facts for Accuracy Verification:
{research_summary or '(No specific external research provided - verify internal logic)'}

Script Draft to Evaluate:
\"\"\"
{script}
\"\"\"

Analyze the script ruthlessly. Score all 6 rubric criteria (1-10), quote weak lines, and list specific fixes.
Return ONLY valid JSON.
"""

    response = call_groq(
        prompt=user_prompt,
        system=CRITIC_SYSTEM_PROMPT,
        json_mode=True,
        model=MODELS.critic,
    )

    # Extract raw scores
    raw_scores = response.get("scores", {}) if isinstance(response, dict) else {}
    validated_scores: dict[str, float] = {}

    for criterion in RUBRIC_CRITERIA:
        raw_val = raw_scores.get(criterion, 7.0)
        try:
            score_float = float(raw_val)
            # Clamp between 1.0 and 10.0
            score_float = max(1.0, min(10.0, score_float))
        except (ValueError, TypeError):
            score_float = 7.0
        validated_scores[criterion] = round(score_float, 2)

    # Compute average strictly in Python (do not trust LLM's own arithmetic)
    computed_average = round(sum(validated_scores.values()) / len(validated_scores), 2)

    # Format feedback
    feedback_raw = response.get("feedback", []) if isinstance(response, dict) else []
    if isinstance(feedback_raw, list):
        feedback_list = [str(item).strip() for item in feedback_raw if str(item).strip()]
    elif isinstance(feedback_raw, str):
        feedback_list = [line.strip() for line in feedback_raw.split("\n") if line.strip()]
    else:
        feedback_list = ["Enhance pacing and punchiness in dialogue.", "Tighten connection to core thesis."]

    # Weak lines
    weak_lines_raw = response.get("weak_lines", []) if isinstance(response, dict) else []
    weak_lines = [str(w).strip() for w in weak_lines_raw if str(w).strip()]

    # Combine feedback and weak lines into a comprehensive feedback summary
    feedback_combined = list(feedback_list)
    if weak_lines:
        feedback_combined.append("Weak lines to rewrite:")
        for wl in weak_lines:
            feedback_combined.append(f"  • {wl}")

    return {
        "scores": validated_scores,
        "average": computed_average,
        "feedback": feedback_combined,
        "weak_lines": weak_lines,
    }
