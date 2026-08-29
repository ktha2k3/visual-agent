from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.visualization.models import RenderResult

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


_SCENE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUALITY = re.compile(r"^[lmhpk]$")


class RenderError(RuntimeError):
    pass


class RenderEnvironmentError(RenderError):
    """The host cannot render; changing generated code cannot fix it."""


def resolve_inside(root: Path, candidate: Path | str) -> Path:
    base = root.resolve()
    path = Path(candidate)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if resolved != base and base not in resolved.parents:
        raise RenderError(f"Path escapes visualization directory: {candidate}")
    return resolved


def _resource_limits() -> None:
    if resource is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    except (ValueError, OSError):
        pass


class ManimRenderer:
    def __init__(self, *, quality: str = "l", timeout: float = 300):
        if not _QUALITY.fullmatch(quality):
            raise ValueError(f"Invalid Manim quality: {quality!r}")
        self.quality = quality
        self.timeout = timeout

    def preflight(self) -> None:
        if importlib.util.find_spec("manim") is None and not shutil.which("manim"):
            raise RenderEnvironmentError(
                "Missing visualization runtime: Manim CE 0.19+"
            )

    @property
    def supports_latex(self) -> bool:
        return bool(shutil.which("latex"))

    @property
    def preferred_text_font(self) -> str | None:
        try:
            from manimpango import list_fonts

            available = set(list_fonts())
        except Exception:
            return None
        for candidate in (
            "Noto Sans",
            "Arial",
            "Segoe UI",
            "DejaVu Sans",
            "Liberation Sans",
        ):
            if candidate in available:
                return candidate
        return None

    def build_command(self, run_dir: Path, scene_name: str) -> list[str]:
        if not _SCENE_NAME.fullmatch(scene_name):
            raise RenderError(f"Invalid Manim scene name: {scene_name!r}")
        root = run_dir.resolve()
        scene_file = resolve_inside(root, "stem_scene.py")
        media_dir = resolve_inside(root, "media")
        if importlib.util.find_spec("manim") is not None:
            executable = [sys.executable, "-m", "manim"]
        else:
            executable = [shutil.which("manim") or "manim"]
        return executable + [
            f"-q{self.quality}",
            "--media_dir",
            str(media_dir),
            "--progress_bar",
            "none",
            str(scene_file.name),
            scene_name,
        ]

    @staticmethod
    def find_video(run_dir: Path) -> Path | None:
        candidates = [
            path
            for path in run_dir.glob("**/*.mp4")
            if "partial_movie_files" not in {part.casefold() for part in path.parts}
        ]
        return max(candidates, key=lambda item: item.stat().st_mtime_ns) if candidates else None

    def _render_sync(self, run_dir: Path, scene_name: str) -> RenderResult:
        self.preflight()
        run_dir = run_dir.resolve()
        (run_dir / "media").mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "render.stdout.log"
        stderr_path = run_dir / "render.stderr.log"
        started = time.monotonic()
        try:
            completed = subprocess.run(
                self.build_command(run_dir, scene_name),
                cwd=run_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
                env=dict(os.environ),
                preexec_fn=_resource_limits if resource is not None else None,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            raise RenderError(f"Manim render timed out after {self.timeout:.0f}s") from exc
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        duration = time.monotonic() - started
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()[-6000:]
            raise RenderError(
                f"Manim exited with code {completed.returncode}: {detail}"
            )
        video = self.find_video(run_dir)
        if video is None or video.stat().st_size < 1024:
            raise RenderError("Manim completed without a valid MP4 artifact")
        return RenderResult(
            success=True,
            video_relative_path=video.relative_to(run_dir).as_posix(),
            stdout_relative_path=stdout_path.name,
            stderr_relative_path=stderr_path.name,
            duration_seconds=round(duration, 3),
        )

    async def render(self, run_dir: Path, scene_name: str) -> RenderResult:
        return await asyncio.to_thread(self._render_sync, run_dir, scene_name)
