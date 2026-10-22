from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from tools.visualization.artifacts import (
    VisualizationArtifactStore,
    visualization_cache_key,
)
from tools.visualization.models import (
    ConceptAnalysis,
    StemVisualizationResult,
    VisualizationPlan,
)
from tools.visualization.prompts import (
    ANALYZER_SYSTEM,
    CODEGEN_SYSTEM,
    MANIM_COMPATIBILITY_RULES,
    PLANNER_SYSTEM,
    REPAIR_SYSTEM,
)
from tools.visualization.provider import LLMProvider, OllamaProvider
from tools.visualization.renderer import (
    ManimRenderer,
    RenderEnvironmentError,
    RenderError,
)
from tools.visualization.validation import extract_python_code, validate_manim_code
from utils.config import Settings, get_settings


ProgressCallback = Callable[[str], None]


class StemVisualizationService:
    """Question-to-Manim pipeline with bounded repair and persistent cache."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        renderer: ManimRenderer | None = None,
        store: VisualizationArtifactStore | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or OllamaProvider()
        self.renderer = renderer or ManimRenderer(
            quality=self.settings.visualization_quality,
            timeout=self.settings.visualization_render_timeout,
            renderer=self.settings.visualization_renderer,
        )
        self.store = store or VisualizationArtifactStore(
            self.settings.visualization_artifact_dir
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self._generation_slots = asyncio.Semaphore(
            self.settings.visualization_max_concurrency
        )

    @staticmethod
    def _notify(callback: ProgressCallback | None, stage: str) -> None:
        if callback:
            callback(stage)

    async def visualize(
        self,
        *,
        question: str,
        subject: str,
        user_id: str,
        level: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> StemVisualizationResult:
        if subject not in {"math", "physics"}:
            raise ValueError("STEM visualization currently supports math and physics")
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        question = question[:10_000]
        artifact_id = visualization_cache_key(
            question,
            subject=subject,
            model=self.settings.visualization_model,
            version=self.settings.visualization_version,
        )
        lock = self._locks.setdefault(artifact_id, asyncio.Lock())
        async with lock:
            cached = self.store.cached_result(artifact_id, user_id=user_id)
            if cached:
                self._notify(progress, "cache_hit")
                return cached
            async with self._generation_slots:
                # A different request may have completed while this one waited.
                cached = self.store.cached_result(artifact_id, user_id=user_id)
                if cached:
                    self._notify(progress, "cache_hit")
                    return cached
                return await self._generate(
                    artifact_id=artifact_id,
                    question=question,
                    subject=subject,
                    user_id=user_id,
                    level=level,
                    progress=progress,
                )

    async def _generate(
        self,
        *,
        artifact_id: str,
        question: str,
        subject: str,
        user_id: str,
        level: str | None,
        progress: ProgressCallback | None,
    ) -> StemVisualizationResult:
        run_dir = self.store.initialize(
            artifact_id,
            user_id=user_id,
            question=question,
            subject=subject,
            model=self.settings.visualization_model,
            version=self.settings.visualization_version,
        )
        analysis: ConceptAnalysis | None = None
        plan: VisualizationPlan | None = None
        try:
            self._notify(progress, "concept_analysis")
            analysis = await self.provider.structured(
                ConceptAnalysis,
                system=ANALYZER_SYSTEM,
                prompt=(
                    f"Môn bắt buộc: {subject}. Trình độ: {level or 'THPT lớp 12'}.\n"
                    f"Câu hỏi học sinh: {question}"
                ),
            )
            analysis.subject = subject  # the caller owns subject routing
            self.store.write_stage(
                artifact_id, "01_concept.json", analysis.model_dump(mode="json")
            )

            self._notify(progress, "visualization_planning")
            plan = await self.provider.structured(
                VisualizationPlan,
                system=PLANNER_SYSTEM,
                prompt=(
                    "Thiết kế visual explanation từ concept packet sau. Mỗi timeline "
                    "beat phải chỉ rõ điều người học quan sát được.\n"
                    + analysis.model_dump_json(indent=2)
                ),
            )
            self.store.write_stage(
                artifact_id, "02_visual_plan.json", plan.model_dump(mode="json")
            )

            self._notify(progress, "code_generation")
            code = extract_python_code(
                await self.provider.text(
                system=MANIM_COMPATIBILITY_RULES + CODEGEN_SYSTEM,
                    prompt=self._code_prompt(
                        analysis,
                        plan,
                        latex_available=bool(
                            getattr(self.renderer, "supports_latex", True)
                        ),
                        text_font=getattr(
                            self.renderer, "preferred_text_font", None
                        ),
                        renderer=getattr(self.renderer, "renderer", "cairo"),
                    ),
                )
            )
        except Exception as exc:
            return self._failed(
                artifact_id,
                subject=subject,
                concept=analysis.concept if analysis else question[:160],
                explanation=(
                    analysis.explanation
                    if analysis
                    else "Không thể hoàn tất bước phân tích trực quan."
                ),
                formulas=analysis.formulas if analysis else [],
                key_takeaway=analysis.key_takeaway if analysis else "Hãy dùng phần giải thích văn bản.",
                plan=plan,
                attempts=0,
                public_error="Không thể tạo kế hoạch trực quan; trợ giảng sẽ tiếp tục bằng văn bản.",
                internal_error=f"{type(exc).__name__}: {exc}",
            )

        attempts_log: list[dict] = []
        max_retries = self.settings.visualization_max_render_retries
        for attempt_index in range(max_retries + 1):
            attempt_number = attempt_index + 1
            self.store.write_scene(artifact_id, code)
            validation = validate_manim_code(
                code,
                required_text_font=getattr(
                    self.renderer, "preferred_text_font", None
                ),
            )
            self.store.write_stage(
                artifact_id,
                f"validation-{attempt_number}.json",
                validation.model_dump(mode="json"),
            )
            attempt_record: dict = {
                "attempt": attempt_number,
                "validation": validation.model_dump(mode="json"),
            }
            attempts_log.append(attempt_record)
            self.store.update_manifest(artifact_id, attempts=attempts_log)

            failure: str | None = None
            if validation.valid and validation.scene_name:
                self._notify(progress, "rendering")
                try:
                    render = await self.renderer.render(run_dir, validation.scene_name)
                    attempt_record["render"] = render.model_dump(mode="json")
                    result = StemVisualizationResult(
                        success=True,
                        subject=subject,
                        concept=analysis.concept,
                        formulas=analysis.formulas,
                        explanation=analysis.explanation,
                        key_takeaway=analysis.key_takeaway,
                        visual_plan=plan,
                        artifact_id=artifact_id,
                        video_url=f"/api/visualizations/{artifact_id}/video",
                        scene_url=f"/api/visualizations/{artifact_id}/scene",
                        attempts=attempt_number,
                    )
                    self.store.update_manifest(
                        artifact_id,
                        status="completed",
                        completed_at=self._utc_now(),
                        attempts=attempts_log,
                        render=render.model_dump(mode="json"),
                        result=result.model_dump(mode="json"),
                    )
                    self._notify(progress, "completed")
                    return result
                except RenderEnvironmentError as exc:
                    attempt_record["environment_error"] = str(exc)
                    return self._failed(
                        artifact_id,
                        subject=subject,
                        concept=analysis.concept,
                        explanation=analysis.explanation,
                        formulas=analysis.formulas,
                        key_takeaway=analysis.key_takeaway,
                        plan=plan,
                        attempts=attempt_number,
                        public_error=(
                            "Máy chủ chưa cài đủ Manim/FFmpeg/LaTeX; trợ giảng sẽ "
                            "tiếp tục bằng văn bản."
                        ),
                        internal_error=str(exc),
                        attempts_log=attempts_log,
                    )
                except RenderError as exc:
                    failure = str(exc)
                    attempt_record["render_error"] = failure
            else:
                failure = "\n".join(validation.errors)

            if attempt_index >= max_retries:
                return self._failed(
                    artifact_id,
                    subject=subject,
                    concept=analysis.concept,
                    explanation=analysis.explanation,
                    formulas=analysis.formulas,
                    key_takeaway=analysis.key_takeaway,
                    plan=plan,
                    attempts=attempt_number,
                    public_error=(
                        "Animation chưa render được sau các lần tự sửa; trợ giảng "
                        "sẽ tiếp tục bằng văn bản."
                    ),
                    internal_error=failure or "Unknown render failure",
                    attempts_log=attempts_log,
                )

            self._notify(progress, "repairing")
            try:
                code = extract_python_code(
                    await self.provider.text(
                        system=MANIM_COMPATIBILITY_RULES + REPAIR_SYSTEM,
                        prompt=(
                            f"Runtime renderer: {getattr(self.renderer, 'renderer', 'cairo')}. "
                            "Use APIs compatible with this renderer.\n"
                            f"LỖI CẦN SỬA:\n{failure}\n\n"
                            f"SCENE CONTRACT:\n{plan.model_dump_json(indent=2)}\n\n"
                            f"MÃ HIỆN TẠI:\n```python\n{code}\n```"
                        ),
                    )
                )
            except Exception as exc:
                return self._failed(
                    artifact_id,
                    subject=subject,
                    concept=analysis.concept,
                    explanation=analysis.explanation,
                    formulas=analysis.formulas,
                    key_takeaway=analysis.key_takeaway,
                    plan=plan,
                    attempts=attempt_number,
                    public_error="Không thể tự sửa animation; trợ giảng sẽ tiếp tục bằng văn bản.",
                    internal_error=f"Repair {type(exc).__name__}: {exc}",
                    attempts_log=attempts_log,
                )

        raise AssertionError("Visualization retry loop terminated unexpectedly")

    @staticmethod
    def _code_prompt(
        analysis: ConceptAnalysis,
        plan: VisualizationPlan,
        *,
        latex_available: bool,
        text_font: str | None,
        renderer: str = "cairo",
    ) -> str:
        latex_note = (
            "LaTeX có sẵn: dùng MathTex cho công thức."
            if latex_available
            else "LaTeX không có trên worker này: render công thức bằng Text với ký hiệu Unicode/ASCII; không dùng MathTex/Tex."
        )
        font_note = (
            f"Mọi Text tiếng Việt bắt buộc đặt font={text_font!r}."
            if text_font
            else "Worker không xác định được font tiếng Việt; ưu tiên nhãn tiếng Anh ngắn để tránh mất glyph."
        )
        return (
            "Tạo một animation ngắn 20-45 giây từ hai artifact sau. Công thức và "
            "quan hệ phải nhìn thấy được qua chuyển động, không chỉ là slide chữ.\n"
            f"RUNTIME: Manim renderer={renderer}; use compatible APIs. {latex_note} {font_note}\n\n"
            f"CONCEPT PACKET:\n{analysis.model_dump_json(indent=2)}\n\n"
            f"SCENE CONTRACT:\n{plan.model_dump_json(indent=2)}"
        )

    @staticmethod
    def _utc_now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _failed(
        self,
        artifact_id: str,
        *,
        subject: str,
        concept: str,
        explanation: str,
        formulas: list[str],
        key_takeaway: str,
        plan: VisualizationPlan | None,
        attempts: int,
        public_error: str,
        internal_error: str,
        attempts_log: list[dict] | None = None,
    ) -> StemVisualizationResult:
        result = StemVisualizationResult(
            success=False,
            subject=subject,
            concept=concept,
            formulas=formulas,
            explanation=explanation,
            key_takeaway=key_takeaway,
            visual_plan=plan,
            artifact_id=artifact_id,
            attempts=attempts,
            error=public_error,
        )
        self.store.update_manifest(
            artifact_id,
            status="failed",
            completed_at=self._utc_now(),
            attempts=attempts_log or [],
            error=internal_error[-8000:],
            result=result.model_dump(mode="json"),
        )
        return result
