"""Central background job registry for multi-agent pipeline runs.

Preserves thread instances, execution states, progress percentages, and results
across Streamlit script reruns without being re-initialized.
"""

from typing import Any

# Global registry: run_id -> job_state dictionary
ACTIVE_JOBS: dict[str, dict[str, Any]] = {}


def get_job(run_id: str) -> dict[str, Any]:
    """Retrieve job state dict for a given run ID."""
    return ACTIVE_JOBS.get(run_id, {})


def set_job(run_id: str, state: dict[str, Any]) -> None:
    """Store or update job state dict for a given run ID."""
    ACTIVE_JOBS[run_id] = state


def has_active_thread(run_id: str) -> bool:
    """Check if a background thread is currently alive for run_id."""
    job = ACTIVE_JOBS.get(run_id)
    if not job:
        return False
    thread = job.get("thread")
    return bool(thread and thread.is_alive())
