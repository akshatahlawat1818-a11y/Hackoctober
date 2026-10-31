"""Run manager and step persistence for the multi-agent workflow.

Persists every step's output to disk under outputs/<run_id>/ so that if any later
step fails (e.g. video rendering or API rate limiting), the workflow can be resumed
without re-executing earlier steps.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Optional
import uuid

from config import OUTPUTS_DIR


class RunManager:
    """Manages workflow run folders and caches step outputs for fault tolerance and resume."""

    def __init__(self, run_id: Optional[str] = None, base_dir: Optional[Path] = None):
        """Initialize the RunManager.

        Args:
            run_id: Optional existing run ID to resume. If None, a new run ID
                    is generated using timestamp and unique hash.
            base_dir: Optional root output directory. Defaults to OUTPUTS_DIR from config.
        """
        self.base_dir = Path(base_dir) if base_dir else OUTPUTS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_id:
            self.run_id = run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            self.run_id = f"run_{timestamp}_{short_id}"

        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.run_dir / "manifest.json"
        self._init_manifest()

    def _init_manifest(self) -> None:
        """Initialize manifest file if it doesn't already exist."""
        if not self.manifest_file.exists():
            initial_data = {
                "run_id": self.run_id,
                "created_at": datetime.now().isoformat(),
                "completed_steps": {},
            }
            self._write_json(self.manifest_file, initial_data)

    def _read_manifest(self) -> dict[str, Any]:
        """Read manifest data."""
        if self.manifest_file.exists():
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"run_id": self.run_id, "completed_steps": {}}

    def _update_manifest(self, step_name: str, filename: str, metadata: Optional[dict] = None) -> None:
        """Update manifest with completed step details."""
        manifest = self._read_manifest()
        manifest.setdefault("completed_steps", {})[step_name] = {
            "file": filename,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._write_json(self.manifest_file, manifest)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write formatted JSON to path atomically."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_user_input(self, idea: str, tone: str, length: str, **extra: Any) -> Path:
        """Persist the initial user request parameters."""
        data = {
            "idea": idea,
            "tone": tone,
            "length": length,
            "extra": extra,
            "recorded_at": datetime.now().isoformat(),
        }
        file_path = self.run_dir / "00_user_input.json"
        self._write_json(file_path, data)
        self._update_manifest("user_input", file_path.name)
        return file_path

    def get_step_file(self, step_name: str, default_ext: str = "json") -> Path:
        """Get the file path for a step."""
        manifest = self._read_manifest()
        completed = manifest.get("completed_steps", {})
        if step_name in completed:
            return self.run_dir / completed[step_name]["file"]
        return self.run_dir / f"{step_name}.{default_ext}"

    def has_step(self, step_name: str) -> bool:
        """Check if a step's output has already been saved and exists on disk."""
        manifest = self._read_manifest()
        completed = manifest.get("completed_steps", {})
        if step_name in completed:
            target = self.run_dir / completed[step_name]["file"]
            return target.exists()
        return False

    def save_step(
        self,
        step_name: str,
        data: Any,
        filename: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save step output data (dict, list, str, or bytes) to file in the run directory.

        Args:
            step_name: Logical step name (e.g. '01_topics', '02_research', '03_script').
            data: Step data to save.
            filename: Custom filename (optional). If None, generated from step_name.
            metadata: Additional metadata dictionary to store in the manifest.

        Returns:
            Path to the saved file.
        """
        if filename is None:
            if isinstance(data, (dict, list)):
                filename = f"{step_name}.json"
            elif isinstance(data, str):
                filename = f"{step_name}.txt"
            elif isinstance(data, bytes):
                filename = f"{step_name}.bin"
            else:
                filename = f"{step_name}.json"

        file_path = self.run_dir / filename

        if isinstance(data, (dict, list)):
            self._write_json(file_path, data)
        elif isinstance(data, str):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(file_path, "wb") as f:
                f.write(data)
        else:
            self._write_json(file_path, data)

        self._update_manifest(step_name, filename, metadata=metadata)
        return file_path

    def load_step(self, step_name: str, as_type: str = "auto") -> Any:
        """Load cached step output if it exists.

        Args:
            step_name: Logical step name to load.
            as_type: 'auto', 'json', 'text', or 'bytes'.

        Returns:
            The loaded data, or None if the step has not been saved.
        """
        if not self.has_step(step_name):
            return None

        file_path = self.get_step_file(step_name)
        if not file_path.exists():
            return None

        if as_type == "json" or (as_type == "auto" and file_path.suffix.lower() == ".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif as_type == "bytes" or (as_type == "auto" and file_path.suffix.lower() in [".bin", ".mp4", ".mp3", ".wav"]):
            with open(file_path, "rb") as f:
                return f.read()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def list_completed_steps(self) -> list[str]:
        """Return list of step names completed in this run."""
        manifest = self._read_manifest()
        return list(manifest.get("completed_steps", {}).keys())
