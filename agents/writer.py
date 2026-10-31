"""Writer Agent.

Generates full script drafts based on user idea, tone, target length, and research facts.
Incorporates natural source citations and systematically addresses critique feedback on revisions.
"""

from typing import Any, Optional
from config import MODELS
from utils.llm import call_gemini

WRITER_SYSTEM_PROMPT = """You are an award-winning scriptwriter and content strategist.
Your task is to write a compelling, informative, and structurally engaging full video script based on the provided project idea and verified research facts.

Tone Requirements:
- If tone is 'humorous': Use clever wit, relatable analogies, sharp pacing, and tasteful comedic commentary while maintaining factual rigor.
- If tone is 'serious': Use authoritative, professional, compelling storytelling with deep technical precision.

Research Integration:
- Naturally incorporate the provided verified research facts.
- Mention or attribute sources smoothly in the dialogue or narration (e.g. 'As researchers at X demonstrated...', 'According to findings documented in Y...').

Script Structure:
- Write in clean markdown format.
- Include scene headings (e.g. ## Scene 1: The Hook), visual/action directions, and spoken narration/dialogue.
- Ensure the script matches the requested target length.
"""


def format_research_context(research: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Format research findings and sources into clean text for prompt context."""
    if isinstance(research, dict):
        items = [research]
    elif isinstance(research, list):
        items = research
    else:
        return "(No research context provided.)"

    lines = []
    for idx, item in enumerate(items, start=1):
        topic = item.get("topic", f"Topic {idx}")
        lines.append(f"\n### Topic {idx}: {topic}")

        facts = item.get("key_facts", [])
        if facts:
            lines.append("Verified Facts & Sources:")
            for f in facts:
                fact_text = f.get("fact", "")
                source_url = f.get("source_url", "")
                lines.append(f"- {fact_text} (Source: {source_url})")

        tools = item.get("tools_or_tech_mentioned", [])
        if tools:
            lines.append(f"Tools & Tech Mentioned: {', '.join(tools)}")

    return "\n".join(lines)


def write_script(
    idea: str,
    tone: str = "serious",
    length: str = "medium",
    research: Optional[list[dict[str, Any]] | dict[str, Any]] = None,
    feedback: Optional[str | list[str]] = None,
) -> str:
    """Generate or revise a full video script based on idea, tone, and research facts.

    Args:
        idea: User project idea.
        tone: 'serious' or 'humorous'.
        length: 'short', 'medium', or 'long'.
        research: Research findings containing verified facts and sources.
        feedback: Editorial feedback from critic to address during revision.

    Returns:
        Full script text in markdown format.
    """
    research_text = format_research_context(research or [])

    revision_instructions = ""
    if feedback:
        if isinstance(feedback, list):
            feedback_str = "\n".join(f"- {item}" for item in feedback)
        else:
            feedback_str = str(feedback)

        revision_instructions = f"""
======================================================================
CRITICAL EDITORIAL REVISION REQUIRED:
Your previous draft was reviewed by the critic and received the following feedback:

{feedback_str}

REVISION RULES:
1. Address and fix EVERY SINGLE POINT in the feedback above.
2. If the critic pointed out weak lines or pacing issues, rewrite them completely.
3. Keep the strengths of the previous version while elevating the quality to pass strict rubric criteria.
======================================================================
"""

    user_prompt = f"""Project Idea: {idea}
Target Tone: {tone.upper()}
Target Length: {length}

Verified Research Findings:
{research_text}
{revision_instructions}
Write the complete script now. Structure it with scene numbers, visual cues, and narration/dialogue.
Seamlessly weave in the research facts and cite sources naturally.
"""

    response = call_gemini(
        prompt=user_prompt,
        system=WRITER_SYSTEM_PROMPT,
        json_mode=False,
        model=MODELS.writer,
    )

    return str(response).strip()
