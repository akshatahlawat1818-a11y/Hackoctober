"""Cost calculation and token usage analyzer for multi-agent runs.

Reads outputs/cost_log.csv and computes total token usage, per-provider
breakdowns, and estimated USD costs for Gemini, Groq, and Vertex Veo video generation.
"""

import csv
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Optional

from config import OUTPUTS_DIR

COST_LOG_PATH = OUTPUTS_DIR / "cost_log.csv"

# Cost rates per 1,000,000 tokens in USD
TOKEN_RATES = {
    # Gemini models
    "gemini-3.8-flash": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.5-flash": {"prompt": 0.10, "completion": 0.40},
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
    "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "gemini-2.5-pro": {"prompt": 1.25, "completion": 5.00},
    "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    # Groq models
    "openai/gpt-oss-120b": {"prompt": 0.59, "completion": 0.59},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},
}


def calculate_run_cost(
    since_timestamp: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Parse cost_log.csv and calculate total tokens and estimated USD cost.

    Args:
        since_timestamp: Optional ISO timestamp to only include calls made after this time.
        csv_path: Optional custom path to cost_log.csv.

    Returns:
        Dictionary with:
          - 'total_cost_usd': float
          - 'total_tokens': int
          - 'prompt_tokens': int
          - 'completion_tokens': int
          - 'provider_breakdown': dict
          - 'entries_count': int
    """
    log_file = csv_path or COST_LOG_PATH
    summary = {
        "total_cost_usd": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "provider_breakdown": {},
        "entries_count": 0,
    }

    if not log_file.exists():
        return summary

    since_dt = None
    if since_timestamp:
        try:
            since_dt = datetime.fromisoformat(since_timestamp)
        except Exception:
            since_dt = None

    cost_regex = re.compile(r"cost:\s*\$([0-9\.]+)\s*USD", re.IGNORECASE)

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp", "")
                if since_dt and ts_str:
                    try:
                        row_dt = datetime.fromisoformat(ts_str)
                        if row_dt < since_dt:
                            continue
                    except Exception:
                        pass

                provider = (row.get("provider") or "unknown").strip().lower()
                model = (row.get("model") or "").strip().lower()
                p_tok = int(row.get("prompt_tokens") or 0)
                c_tok = int(row.get("completion_tokens") or 0)
                t_tok = int(row.get("total_tokens") or (p_tok + c_tok))
                status = row.get("status") or ""

                summary["entries_count"] += 1
                summary["prompt_tokens"] += p_tok
                summary["completion_tokens"] += c_tok
                summary["total_tokens"] += t_tok

                row_cost = 0.0

                # 1. Check for explicit cost in status string (e.g. from Veo or dry-run)
                match = cost_regex.search(status)
                if match:
                    try:
                        row_cost = float(match.group(1))
                    except ValueError:
                        row_cost = 0.0
                elif "simulated" in provider or "dry_run" in status:
                    row_cost = 0.0
                elif provider == "vertex_veo":
                    # Veo fallback if status didn't log cost string
                    if "lite" in model or "draft" in model:
                        row_cost = 0.05 * max(1, p_tok)
                    elif "dummy" in model:
                        row_cost = 0.0
                    else:
                        row_cost = 0.20 * max(1, p_tok)
                else:
                    # Token-based estimation for LLMs
                    rate = TOKEN_RATES.get(model)
                    if not rate:
                        for k, v in TOKEN_RATES.items():
                            if k in model:
                                rate = v
                                break
                    if not rate:
                        rate = {"prompt": 0.10, "completion": 0.40}

                    row_cost = (p_tok * rate["prompt"] + c_tok * rate["completion"]) / 1_000_000.0

                summary["total_cost_usd"] += row_cost

                # Provider breakdown
                if provider not in summary["provider_breakdown"]:
                    summary["provider_breakdown"][provider] = {
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "calls": 0,
                    }
                summary["provider_breakdown"][provider]["tokens"] += t_tok
                summary["provider_breakdown"][provider]["cost_usd"] += row_cost
                summary["provider_breakdown"][provider]["calls"] += 1

    except Exception:
        pass

    return summary
