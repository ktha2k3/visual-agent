from __future__ import annotations

import asyncio
from typing import Literal

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from tools.visualization.models import StemVisualizationResult
from tools.visualization.service import StemVisualizationService


class VisualizeStemConceptInput(BaseModel):
    question: str = Field(min_length=2, max_length=10_000)
    subject: Literal["math", "physics"]
    level: str | None = Field(default=None, max_length=100)


async def visualize_stem_concept(
    question: str,
    subject: Literal["math", "physics"],
    level: str | None = None,
    *,
    user_id: str = "tool-user",
    service: StemVisualizationService | None = None,
) -> StemVisualizationResult:
    """Create a validated and rendered visual explanation for Math or Physics."""
    pipeline = service or StemVisualizationService()
    return await pipeline.visualize(
        question=question, subject=subject, level=level, user_id=user_id
    )


class VisualizeStemConceptTool(BaseTool):
    name: str = "visualize_stem_concept"
    description: str = (
        "Create a Manim animation for a Math/Physics concept, formula, law, "
        "rule, or phenomenon. Prefer this tool over text-only explanation."
    )
    args_schema: type[BaseModel] = VisualizeStemConceptInput
    return_direct: bool = False
    _service: StemVisualizationService = PrivateAttr()
    _user_id: str = PrivateAttr()

    def __init__(self, *, user_id: str, service: StemVisualizationService | None = None):
        super().__init__()
        self._user_id = user_id
        self._service = service or StemVisualizationService()

    async def _arun(
        self,
        question: str,
        subject: Literal["math", "physics"],
        level: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        result = await visualize_stem_concept(
            question,
            subject,
            level,
            user_id=self._user_id,
            service=self._service,
        )
        return result.model_dump(mode="json")

    def _run(
        self,
        question: str,
        subject: Literal["math", "physics"],
        level: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        return asyncio.run(
            self._arun(question=question, subject=subject, level=level)
        )


def build_visualization_tool(
    user_id: str, service: StemVisualizationService | None = None
) -> VisualizeStemConceptTool:
    return VisualizeStemConceptTool(user_id=user_id, service=service)

