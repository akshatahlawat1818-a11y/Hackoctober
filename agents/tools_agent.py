"""Tools Agent.

Analyzes the project idea, written script, and research findings to compile an actionable
list of tools, APIs, libraries, and setup steps needed to build the idea.
"""

import json
from pathlib import Path
from typing import Any, Optional

from config import MODELS, OUTPUTS_DIR, get_run_dir
from utils.llm import call_gemini

TOOLS_SYSTEM_PROMPT = """You are a senior software architect and technical lead.
Your task is to identify the precise tools, SDKs, APIs, and libraries required to build the project based on the idea, script, and research findings.

Requirements:
1. Include ONLY tools that were explicitly mentioned in the research context or are standard, well-established industry technologies (e.g. PyMuPDF, FastAPI, LangChain, Tesseract, Hugging Face transformers, OpenAI, PostgreSQL).
2. 'cost' MUST be strictly one of: 'free', 'paid', or 'freemium'.
3. 'setup_steps' MUST be a list of 2 to 4 concise, practical terminal commands or setup steps.
4. 'link' MUST be a real, official documentation or GitHub URL. NEVER invent or fabricate broken URLs.

You must respond ONLY with a valid JSON object matching this schema:
{
  "tools": [
    {
      "name": "Tool Name",
      "purpose": "Specific role this tool plays in building the application",
      "cost": "free",
      "setup_steps": [
        "pip install package-name",
        "Configure API key in environment"
      ],
      "link": "https://official-doc-or-github-url"
    }
  ]
}
"""


def build_tools_list(
    idea: str,
    script: str,
    research: Optional[list[dict[str, Any]] | dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Build a comprehensive list of tools and setup steps required to execute the idea.

    Args:
        idea: Project idea description.
        script: Approved final script.
        research: Research findings containing tools and verified sources.
        run_id: Optional run ID to save output in outputs/<run_id>/tools.json.

    Returns:
        List of tool dictionaries.
    """
    # Compile research context
    research_summary = ""
    known_tools: list[str] = []
    if research:
        if isinstance(research, list):
            for item in research:
                tools = item.get("tools_or_tech_mentioned", [])
                known_tools.extend(tools)
                topic = item.get("topic", "")
                research_summary += f"\nTopic: {topic}\nTools found: {', '.join(tools)}\n"
        elif isinstance(research, dict):
            research_summary = str(research)

    user_prompt = f"""Project Idea: {idea}

Script Context:
\"\"\"
{script[:1500]}
\"\"\"

Research Technologies Identified:
{research_summary or '(Use foundational industry tools for this domain)'}

Compile a clean, actionable list of 4-7 core tools/APIs needed to build this system from scratch.
Enforce accurate costs ('free', 'paid', 'freemium') and verified official documentation links.
"""

    response = call_gemini(
        prompt=user_prompt,
        system=TOOLS_SYSTEM_PROMPT,
        json_mode=True,
        model=MODELS.tools,
    )

    # Extract tools list
    raw_tools = []
    if isinstance(response, dict):
        if "tools" in response and isinstance(response["tools"], list):
            raw_tools = response["tools"]
        elif "items" in response and isinstance(response["items"], list):
            raw_tools = response["items"]
        else:
            for val in response.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    raw_tools = val
                    break
    elif isinstance(response, list):
        raw_tools = response

    validated_tools: list[dict[str, Any]] = []
    valid_costs = {"free", "paid", "freemium"}

    for item in raw_tools:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "Tool")).strip()
        purpose = str(item.get("purpose", "")).strip()

        # Normalize cost
        cost_raw = str(item.get("cost", "free")).strip().lower()
        cost = cost_raw if cost_raw in valid_costs else ("freemium" if "free" in cost_raw else "paid")

        # Setup steps
        steps_raw = item.get("setup_steps", [])
        if isinstance(steps_raw, list):
            steps = [str(s).strip() for s in steps_raw if str(s).strip()]
        elif isinstance(steps_raw, str):
            steps = [s.strip() for s in steps_raw.split("\n") if s.strip()]
        else:
            steps = [f"Install and configure {name}"]

        # Link verification and fallback
        link = str(item.get("link", "")).strip()
        if not link.startswith("http://") and not link.startswith("https://"):
            link = f"https://www.google.com/search?q={name.replace(' ', '+')}+documentation"

        validated_tools.append({
            "name": name,
            "purpose": purpose,
            "cost": cost,
            "setup_steps": steps,
            "link": link,
        })

    # Save output if run_id is provided
    if run_id:
        target_dir = get_run_dir(run_id)
    else:
        target_dir = OUTPUTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

    tools_file = target_dir / "tools.json"
    with open(tools_file, "w", encoding="utf-8") as f:
        json.dump(validated_tools, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(validated_tools)} tools to: {tools_file}")

    return validated_tools
