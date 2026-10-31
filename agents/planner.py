"""Planner Agent.

Generates 5 to 8 focused research topics with search queries for a given project idea,
tone, and target script length.
"""

from typing import Any
from config import MODELS
from utils.llm import call_gemini

PLANNER_SYSTEM_PROMPT = """You are a strategic AI research planner.
Your goal is to break down a project idea into exactly 5 to 8 focused research topics needed to write an exceptional, well-researched script and product roadmap.

For each topic, provide:
1. 'title': A concise, descriptive topic name.
2. 'why_it_matters': 1-2 clear sentences explaining why this topic is essential for the script and build.
3. 'search_queries': An array of 2 to 3 precise web search queries (ready for DuckDuckGo/Wikipedia).

You must respond ONLY with a valid JSON object matching this schema:
{
  "topics": [
    {
      "title": "Topic title",
      "why_it_matters": "Why this is critical to research",
      "search_queries": [
        "search query 1",
        "search query 2"
      ]
    }
  ]
}
Ensure there are between 5 and 8 topics in the 'topics' array.
"""


def plan_topics(idea: str, tone: str = "serious", length: str = "medium") -> list[dict[str, Any]]:
    """Plan 5 to 8 research topics for an idea with search queries.

    Args:
        idea: User's rough project idea.
        tone: Desired script tone (e.g. 'serious', 'humorous').
        length: Desired script length (e.g. 'short', 'medium', 'long').

    Returns:
        List of 5 to 8 topic dictionaries, each containing:
        - 'title': str
        - 'why_it_matters': str
        - 'search_queries': list of 2-3 strings
    """
    user_prompt = f"""Project Idea: {idea}
Target Tone: {tone}
Script Length: {length}

Identify 5 to 8 key research topics to explore before writing the script and planning technical tools.
Cover market context, core AI technologies/algorithms, user pain points, implementation pitfalls, and practical execution steps.
Return ONLY valid JSON with a 'topics' list.
"""

    response = call_gemini(
        prompt=user_prompt,
        system=PLANNER_SYSTEM_PROMPT,
        json_mode=True,
        model=MODELS.planner,
    )

    # Extract topics list from response dict
    raw_topics: list[Any] = []
    if isinstance(response, dict):
        if "topics" in response and isinstance(response["topics"], list):
            raw_topics = response["topics"]
        elif "items" in response and isinstance(response["items"], list):
            raw_topics = response["items"]
        else:
            # Check for any list value inside dictionary
            for val in response.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    raw_topics = val
                    break
    elif isinstance(response, list):
        raw_topics = response

    # Normalize and validate topics
    cleaned_topics: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_topics, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", f"Research Topic {idx}")).strip()
        why = str(item.get("why_it_matters", "Critical background context for project execution.")).strip()
        queries = item.get("search_queries", [])

        if not isinstance(queries, list) or not queries:
            queries = [f"{title} technical overview", f"{title} best practices"]
        else:
            # Ensure 2-3 string queries
            queries = [str(q).strip() for q in queries if str(q).strip()][:3]
            while len(queries) < 2:
                queries.append(f"{title} implementation")

        cleaned_topics.append({
            "title": title,
            "why_it_matters": why,
            "search_queries": queries,
        })

    # Ensure bounds: minimum 5, maximum 8 topics
    if len(cleaned_topics) > 8:
        cleaned_topics = cleaned_topics[:8]

    return cleaned_topics
