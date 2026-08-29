from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tools.visualization.models import StemVisualizationResult


_ARTIFACT_ID = re.compile(r"^[a-f0-9]{64}$")


def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def visualization_cache_key(
    question: str, *, subject: str, model: str, version: str
) -> str:
    payload = "\n".join((normalize_question(question), subject, model, version))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VisualizationArtifactStore:
    """Persistent, ownership-aware run ledger and content-addressed cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _artifact_dir(self, artifact_id: str) -> Path:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ValueError("Invalid visualization artifact id")
        candidate = (self.root / artifact_id).resolve()
        if self.root not in candidate.parents:
            raise ValueError("Visualization path escaped artifact root")
        return candidate

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def initialize(
        self,
        artifact_id: str,
        *,
        user_id: str,
        question: str,
        subject: str,
        model: str,
        version: str,
    ) -> Path:
        run_dir = self._artifact_dir(artifact_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "artifact_id": artifact_id,
                "question": question,
                "subject": subject,
                "model": model,
                "visualization_version": version,
                "owners": [user_id],
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "attempts": [],
            },
        )
        return run_dir

    def read_manifest(self, artifact_id: str) -> dict[str, Any] | None:
        path = self._artifact_dir(artifact_id) / "manifest.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def update_manifest(self, artifact_id: str, **changes: Any) -> dict[str, Any]:
        run_dir = self._artifact_dir(artifact_id)
        manifest = self.read_manifest(artifact_id) or {"artifact_id": artifact_id}
        manifest.update(changes)
        self._write_json_atomic(run_dir / "manifest.json", manifest)
        return manifest

    def add_owner(self, artifact_id: str, user_id: str) -> None:
        manifest = self.read_manifest(artifact_id)
        if not manifest:
            return
        owners = list(dict.fromkeys([*manifest.get("owners", []), user_id]))
        self.update_manifest(artifact_id, owners=owners)

    def cached_result(
        self, artifact_id: str, *, user_id: str
    ) -> StemVisualizationResult | None:
        manifest = self.read_manifest(artifact_id)
        if not manifest or manifest.get("status") != "completed":
            return None
        result_data = manifest.get("result")
        if not isinstance(result_data, dict):
            return None
        result = StemVisualizationResult.model_validate(result_data)
        if not result.success:
            return None
        video = self.resolve_owned_file(
            artifact_id, user_id=None, kind="video", require_owner=False
        )
        if video is None or video.stat().st_size < 1024:
            return None
        self.add_owner(artifact_id, user_id)
        return result.model_copy(update={"cached": True})

    def write_scene(self, artifact_id: str, code: str) -> Path:
        path = self._artifact_dir(artifact_id) / "stem_scene.py"
        path.write_text(code, encoding="utf-8")
        return path

    def write_stage(self, artifact_id: str, name: str, payload: dict[str, Any]) -> Path:
        if not re.fullmatch(r"[a-z0-9_.-]+", name):
            raise ValueError("Invalid visualization stage name")
        path = self._artifact_dir(artifact_id) / name
        self._write_json_atomic(path, payload)
        return path

    def resolve_owned_file(
        self,
        artifact_id: str,
        user_id: str | None,
        *,
        kind: Literal["video", "scene"],
        require_owner: bool = True,
    ) -> Path | None:
        run_dir = self._artifact_dir(artifact_id)
        manifest = self.read_manifest(artifact_id)
        if not manifest:
            return None
        if require_owner and (not user_id or user_id not in manifest.get("owners", [])):
            return None
        if kind == "scene":
            path = run_dir / "stem_scene.py"
        else:
            relative = (manifest.get("render") or {}).get("video_relative_path")
            if not isinstance(relative, str):
                return None
            path = (run_dir / relative).resolve()
            if run_dir not in path.parents:
                return None
        return path if path.is_file() else None

