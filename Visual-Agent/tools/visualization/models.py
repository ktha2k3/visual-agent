from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


VisualizationSubject = Literal["math", "physics"]


class ConceptAnalysis(BaseModel):
    """Student-facing facts selected before any visual design is attempted."""

    subject: VisualizationSubject
    concept: str = Field(min_length=2, max_length=160)
    explanation: str = Field(min_length=20, max_length=4000)
    formulas: list[str] = Field(default_factory=list, max_length=8)
    prerequisites: list[str] = Field(default_factory=list, max_length=8)
    key_takeaway: str = Field(min_length=10, max_length=1000)

    @field_validator("formulas", "prerequisites")
    @classmethod
    def remove_empty_items(cls, values: list[str]) -> list[str]:
        return [item.strip() for item in values if item and item.strip()]


class VisualObject(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal[
        "text", "formula", "graph", "vector", "geometry", "object", "particle"
    ]
    purpose: str = Field(min_length=3, max_length=400)


class AnimationBeat(BaseModel):
    order: int = Field(ge=1, le=30)
    duration_seconds: float = Field(default=2.0, ge=0.2, le=20.0)
    action: str = Field(min_length=5, max_length=1000)
    narration: str = Field(min_length=3, max_length=1000)


class VisualizationPlan(BaseModel):
    """Compiler-like scene contract consumed by the Manim code generator."""

    concept: str = Field(min_length=2, max_length=160)
    learning_goal: str = Field(min_length=10, max_length=1000)
    objects: list[VisualObject] = Field(min_length=1, max_length=30)
    relationships: list[str] = Field(default_factory=list, max_length=20)
    variable_changes: list[str] = Field(default_factory=list, max_length=12)
    camera_movements: list[str] = Field(default_factory=list, max_length=12)
    transformations: list[str] = Field(default_factory=list, max_length=16)
    timeline: list[AnimationBeat] = Field(min_length=2, max_length=30)


class ValidationReport(BaseModel):
    valid: bool
    scene_name: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RenderResult(BaseModel):
    success: bool
    video_relative_path: str | None = None
    stdout_relative_path: str | None = None
    stderr_relative_path: str | None = None
    error: str | None = None
    duration_seconds: float | None = None


class StemVisualizationResult(BaseModel):
    success: bool
    subject: VisualizationSubject
    concept: str
    formulas: list[str] = Field(default_factory=list)
    explanation: str
    key_takeaway: str
    visual_plan: VisualizationPlan | None = None
    artifact_id: str | None = None
    video_url: str | None = None
    scene_url: str | None = None
    cached: bool = False
    attempts: int = 0
    error: str | None = None
    status: str | None = None
    scene_data: dict | None = None
    diagnostic_id: str | None = None

