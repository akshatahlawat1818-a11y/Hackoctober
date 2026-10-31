"""Researcher Agent.

Performs web search via DuckDuckGo and Wikipedia for research topics, passes the
gathered context to Gemini for synthesis, strictly enforces URL source verification,
and persists results sequentially with rate limit pacing.
"""

import json
from pathlib import Path
import time
from typing import Any, Optional

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore

import wikipedia

from config import MODELS, OUTPUTS_DIR, get_run_dir
from utils.llm import call_gemini

# Set respectful User-Agent for Wikipedia API requests
wikipedia.set_user_agent("MultiAgentWorkflow/1.0 (https://github.com/example/workflow; researcher@example.com)")

RESEARCHER_SYSTEM_PROMPT = """You are an expert research analyst.
Your job is to synthesize raw search results into accurate, highly informative key facts and identify mentioned tools/technologies.

CRITICAL SOURCE URL RULE:
1. Every fact in 'key_facts' MUST cite an exact 'source_url' taken directly from the provided search sources.
2. Never invent, truncate, fabricate, or guess a URL.
3. If an insight cannot be directly attributed to a provided source URL, DO NOT include it.

Output ONLY a valid JSON object matching this schema:
{
  "topic": "Topic Name",
  "key_facts": [
    {
      "fact": "A concise, verified factual insight",
      "source_url": "https://exact.url.from.sources.com/page"
    }
  ],
  "tools_or_tech_mentioned": ["Tool1", "Tool2", "Library3"]
}
"""


def fetch_duckduckgo_results(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Run a DuckDuckGo search query and return results with title, href, and body."""
    results: list[dict[str, str]] = []
    try:
        ddg = DDGS()
        raw_items = ddg.text(query, max_results=max_results)
        for item in raw_items:
            href = item.get("href") or item.get("link") or ""
            title = item.get("title") or ""
            body = item.get("body") or item.get("snippet") or ""
            if href and (title or body):
                results.append({"title": title, "href": href, "body": body})
    except Exception as e:
        print(f"Warning: DuckDuckGo search error for '{query}': {e}")
    return results


def fetch_wikipedia_summary(query: str) -> Optional[dict[str, str]]:
    """Search and fetch a Wikipedia summary and canonical page URL when relevant."""
    try:
        # Search for closest Wikipedia page
        search_hits = wikipedia.search(query, results=1)
        if not search_hits:
            return None

        page_title = search_hits[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        summary = wikipedia.summary(page_title, sentences=3, auto_suggest=False)

        if page.url and summary:
            return {
                "title": f"Wikipedia: {page.title}",
                "href": page.url,
                "body": summary,
            }
    except Exception:
        # Ignore Wikipedia disambiguation, missing page, or network errors
        return None
    return None


def research_topic(topic: dict[str, Any] | str) -> dict[str, Any]:
    """Research a single topic using DuckDuckGo and Wikipedia, then synthesize with Gemini.

    Enforces that every fact cites a verified URL present in the real search results.

    Args:
        topic: Topic dictionary with 'title' and 'search_queries', or topic title string.

    Returns:
        Dictionary containing:
        - 'topic': str
        - 'key_facts': list of dicts with 'fact' and 'source_url'
        - 'tools_or_tech_mentioned': list of strings
    """
    if isinstance(topic, str):
        topic_title = topic
        queries = [topic, f"{topic} tools and technologies"]
    else:
        topic_title = topic.get("title", "Research Topic")
        queries = topic.get("search_queries", [topic_title])

    # 1. Gather search results from DuckDuckGo and Wikipedia
    valid_sources: dict[str, dict[str, str]] = {}

    # DuckDuckGo queries
    for query in queries:
        ddg_hits = fetch_duckduckgo_results(query, max_results=3)
        for hit in ddg_hits:
            href = hit["href"].strip()
            if href:
                valid_sources[href] = hit

    # Wikipedia search on topic title
    wiki_hit = fetch_wikipedia_summary(topic_title)
    if wiki_hit:
        valid_sources[wiki_hit["href"].strip()] = wiki_hit

    # Build prompt context from verified sources
    sources_text = ""
    for idx, (url, data) in enumerate(valid_sources.items(), start=1):
        sources_text += f"\nSource {idx}:\nURL: {url}\nTitle: {data['title']}\nSnippet: {data['body']}\n"

    if not valid_sources:
        sources_text = "(No web search results could be retrieved for this topic.)"

    user_prompt = f"""Topic to Research: {topic_title}

Search Results and Sources:
{sources_text}

Synthesize 3-6 key facts about this topic and identify tools/technologies mentioned.
MANDATORY RULE:
- Every item in 'key_facts' MUST have 'source_url' exactly matching one of the URLs listed above.
- Do NOT fabricate or modify URLs.
"""

    # 2. Call Gemini for synthesis in JSON mode
    response = call_gemini(
        prompt=user_prompt,
        system=RESEARCHER_SYSTEM_PROMPT,
        json_mode=True,
        model=MODELS.research,
    )

    # 3. Strictly validate facts: Drop any fact whose source_url was not in real search results
    raw_facts = response.get("key_facts", []) if isinstance(response, dict) else []
    tools_mentioned = response.get("tools_or_tech_mentioned", []) if isinstance(response, dict) else []

    verified_facts: list[dict[str, str]] = []
    real_urls = set(valid_sources.keys())

    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        fact_text = str(item.get("fact", "")).strip()
        source_url = str(item.get("source_url", "")).strip()

        if not fact_text or not source_url:
            continue

        # Check exact or normalized match against real URLs
        matched_url = None
        for real_url in real_urls:
            if source_url == real_url or source_url.rstrip("/") == real_url.rstrip("/"):
                matched_url = real_url
                break

        if matched_url:
            verified_facts.append({
                "fact": fact_text,
                "source_url": matched_url,
            })
        else:
            # Drop fact with invented / unverified URL
            print(f"  [Security/Fact-Check] Dropped unverified fact: URL '{source_url}' not found in search results.")

    # Clean tools list
    cleaned_tools = [str(t).strip() for t in tools_mentioned if str(t).strip()]

    return {
        "topic": topic_title,
        "key_facts": verified_facts,
        "tools_or_tech_mentioned": cleaned_tools,
    }


def research_all(
    topics: list[dict[str, Any] | str],
    run_id: Optional[str] = None,
    pause_seconds: float = 3.0,
    delay_seconds: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Research all topics sequentially with a pause between each to respect rate limits.

    Saves each result to outputs/<run_id>/research_<n>.json.

    Args:
        topics: List of topic dicts or title strings.
        run_id: Optional run ID for output folder.
        pause_seconds: Delay between research queries (default 3 seconds).
        delay_seconds: Alias for pause_seconds.

    Returns:
        List of all research summary dicts.
    """
    if delay_seconds is not None:
        pause_seconds = delay_seconds
    total = len(topics)
    all_results: list[dict[str, Any]] = []

    if run_id:
        out_dir = get_run_dir(run_id)
    else:
        out_dir = OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"Starting Sequential Research for {total} Topics (Rate limit pause: {pause_seconds}s)")
    print("=" * 65)

    for idx, topic in enumerate(topics, start=1):
        title = topic if isinstance(topic, str) else topic.get("title", f"Topic {idx}")
        print(f"\n[{idx}/{total}] Researching: {title}...")

        summary = research_topic(topic)
        all_results.append(summary)

        # Save to outputs/<run_id>/research_<n>.json
        out_file = out_dir / f"research_{idx}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        facts_count = len(summary.get("key_facts", []))
        tools_count = len(summary.get("tools_or_tech_mentioned", []))
        print(f"[{idx}/{total}] Completed: {facts_count} verified facts, {tools_count} tools found.")
        print(f"  Saved to: {out_file}")

        # Pause between topics to respect rate limits (except after last topic)
        if idx < total and pause_seconds > 0:
            print(f"  Pacing delay: waiting {pause_seconds}s...")
            time.sleep(pause_seconds)

    print("\n" + "=" * 65)
    print(f"Research phase complete for all {total} topics.")
    print("=" * 65 + "\n")

    return all_results
