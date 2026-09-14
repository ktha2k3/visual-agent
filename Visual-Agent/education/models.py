from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Subject(str, Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    UNKNOWN = "unknown"


class LearningMode(str, Enum):
    CONCEPT = "concept"
    GUIDED_PROBLEM = "guided_problem"


# Chemistry and Biology are retained in the enum so existing database rows can
# still be read. New product flows deliberately support only these two subjects.
SUPPORTED_SUBJECTS = frozenset({Subject.MATH, Subject.PHYSICS})


def is_supported_subject(subject: Subject | None) -> bool:
    return subject in SUPPORTED_SUBJECTS


class AuthUser(BaseModel):
    id: str
    email: str | None = None
    is_admin: bool = False
    access_token: str | None = Field(default=None, exclude=True)


class AssetRecord(BaseModel):
    id: str
    user_id: str
    storage_path: str
    mime_type: str
    size_bytes: int
    preview_url: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeIngestionJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    subject: Subject
    original_filename: str
    storage_path: str
    rights_note: str
    skill_ids: list[str] = Field(default_factory=list)
    size_bytes: int = Field(ge=1)
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    document_id: str | None = None
    chunks: int | None = Field(default=None, ge=0)
    embedding_status: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeDocumentSummary(BaseModel):
    id: str
    user_id: str
    title: str
    subject: Subject
    original_filename: str | None = None
    size_bytes: int | None = None
    embedding_status: str
    review_status: str = "approved"
    chunks: int = Field(default=0, ge=0)
    created_at: datetime


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    last_message: str
    message_count: int = Field(ge=0)
    subject: Subject | None = None
    learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM
    updated_at: datetime


class ChatSessionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    subject: Subject
    learning_mode: LearningMode = LearningMode.GUIDED_PROBLEM
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> str:
        return str(value)


class ChatMessage(BaseModel):
    id: str | None = None
    user_id: str
    session_id: str
    request_id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("id", "session_id", mode="before")
    @classmethod
    def normalize_uuid_fields(cls, value: Any) -> str | None:
        return None if value is None else str(value)


class ProblemExtraction(BaseModel):
    subject: Subject = Subject.UNKNOWN
    problem_markdown: str = ""
    givens: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=list)
    diagram_description: str | None = None
    student_work: str | None = None
    skill_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Citation(BaseModel):
    title: str
    url: str
    excerpt: str | None = None
    source_type: str = "curriculum"


class TutorResponse(BaseModel):
    message: str
    hint_level: int = Field(default=0, ge=0, le=4)
    feedback_type: str = "guidance"
    answer_correct: bool | None = None
    used_full_solution: bool = False
    misconception: str | None = None
    evidence: float | None = Field(default=None, ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)


class LearningApproach(BaseModel):
    level: Literal["foundation", "intermediate", "advanced"]
    title: str
    focus: str


class LearningPlan(BaseModel):
    subject: Subject
    concept: str
    intent: Literal["solve_problem", "understand_concept"]
    understanding_level: Literal[
        "insufficient_data", "foundation", "developing", "proficient", "advanced"
    ]
    understanding_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    assessment_summary: str
    approaches: list[LearningApproach]
    research_task: str
    needs_research: bool = False


class MasteryRecord(BaseModel):
    user_id: str
    skill_id: str
    subject: Subject
    mastery: float = Field(default=0.5, ge=0.0, le=1.0)
    attempts: int = 0
    recent_errors: int = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MasteryUpdate(BaseModel):
    skill_id: str
    subject: Subject
    old_mastery: float
    new_mastery: float
    evidence: float


class LearningProfileSummary(BaseModel):
    status: str
    level: str
    overall_mastery: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = Field(default=0, ge=0)
    observed_skills: int = Field(default=0, ge=0)
    strengths: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    narrative: str


class Recommendation(BaseModel):
    skill_id: str
    subject: Subject
    title: str
    score: float = Field(ge=0.0, le=1.0)
    mastery_gap: float
    recent_error_rate: float
    reason: str
