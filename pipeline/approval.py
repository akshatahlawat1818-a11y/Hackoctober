"""Human-in-the-loop approval step for research topics.

Allows the user to review, edit, add, remove, or regenerate research topics
before proceeding with research and writing.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional

from agents.planner import plan_topics
from config import OUTPUTS_DIR, get_run_dir


def display_topics(topics: list[dict[str, Any]], idea: Optional[str] = None) -> None:
    """Print research topics in a clear, numbered list format."""
    print("\n" + "=" * 65)
    print("               PROPOSED RESEARCH TOPICS")
    print("=" * 65)
    if idea:
        print(f"Project Idea: {idea}")
        print("-" * 65)

    if not topics:
        print("  (No topics currently in list)")
    else:
        for idx, topic in enumerate(topics, start=1):
            title = topic.get("title", f"Topic {idx}")
            why = topic.get("why_it_matters", "No rationale specified.")
            queries = topic.get("search_queries", [])
            print(f"{idx}. {title}")
            print(f"   Why it matters: {why}")
            if queries:
                print("   Search Queries:")
                for q in queries:
                    print(f"     - {q}")
            print()

    print("-" * 65)
    print("Available Commands:")
    print("  ok           - Approve topics and proceed")
    print("  remove <num> - Delete a topic by number (e.g. 'remove 3')")
    print("  add <text>   - Add a new topic (e.g. 'add Vector databases')")
    print("  redo         - Regenerate topics from scratch")
    print("=" * 65 + "\n")


def save_topics_to_disk(topics: list[dict[str, Any]], run_id: Optional[str] = None) -> Path:
    """Save approved topics to outputs/<run_id>/topics.json or outputs/topics.json."""
    if run_id:
        target_dir = get_run_dir(run_id)
    else:
        target_dir = OUTPUTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / "topics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(topics)} approved topics to {output_path}")
    return output_path


def approve_topics(
    topics: list[dict[str, Any]],
    idea: str = "",
    tone: str = "serious",
    length: str = "medium",
    run_id: Optional[str] = None,
    input_func: Callable[[str], str] = input,
    auto_approve: bool = False,
) -> list[dict[str, Any]]:
    """Interactive command-line approval step for research topics.

    Supported commands:
    - 'ok': accept and proceed
    - 'remove <n>': delete topic at index n (1-indexed)
    - 'add <text>': add a new topic
    - 'redo': regenerate topics with the planner agent

    Args:
        topics: Initial list of topic dicts from planner.
        idea: Project idea description.
        tone: Target tone for regeneration.
        length: Target length for regeneration.
        run_id: Optional run ID for output folder outputs/<run_id>/topics.json.
        input_func: Function to read input (defaults to built-in input).
        auto_approve: If True, bypasses prompt and approves topics immediately.

    Returns:
        Final list of approved topic dictionaries.
    """
    import sys
    current_topics = list(topics)
    display_topics(current_topics, idea=idea)

    if auto_approve or (input_func is input and not sys.stdin.isatty()):
        print("[Auto-Approval] Proceeding with suggested topics.")
        save_topics_to_disk(current_topics, run_id=run_id)
        return current_topics

    while True:
        try:
            raw_input = input_func("Enter command (ok, remove <n>, add <text>, redo): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting approval loop.")
            break

        if not raw_input:
            continue

        cmd = raw_input.lower()

        # 1. OK - Accept
        if cmd == "ok":
            print(f"\nApproved {len(current_topics)} research topics.")
            break

        # 2. REDO - Regenerate
        elif cmd == "redo":
            print(f"\nRegenerating research topics for: '{idea}'...")
            current_topics = plan_topics(idea=idea, tone=tone, length=length)
            display_topics(current_topics, idea=idea)

        # 3. REMOVE <num> - Delete topic
        elif cmd.startswith("remove"):
            parts = raw_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip().isdigit():
                print("Usage: remove <number> (e.g. 'remove 3')")
                continue

            idx_to_remove = int(parts[1].strip())
            if 1 <= idx_to_remove <= len(current_topics):
                removed = current_topics.pop(idx_to_remove - 1)
                print(f"Removed: '{removed.get('title', 'Unknown')}'")
                display_topics(current_topics, idea=idea)
            else:
                print(f"Invalid index {idx_to_remove}. Choose between 1 and {len(current_topics)}.")

        # 4. ADD <text> - Add custom topic
        elif cmd.startswith("add"):
            parts = raw_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("Usage: add <topic title> (e.g. 'add Vector databases')")
                continue

            title = parts[1].strip()
            new_topic = {
                "title": title,
                "why_it_matters": f"User-requested topic investigating {title}.",
                "search_queries": [
                    f"{title} technical overview",
                    f"{title} implementation guide",
                ],
            }
            current_topics.append(new_topic)
            print(f"Added: '{title}'")
            display_topics(current_topics, idea=idea)

        # Unknown command
        else:
            print(f"Unknown command: '{raw_input}'")
            print("Commands: 'ok' to accept, 'remove <n>' to delete, 'add <text>' to add, 'redo' to regenerate.")

    # Save approved topics to outputs/<run_id>/topics.json
    save_topics_to_disk(current_topics, run_id=run_id)
    return current_topics
