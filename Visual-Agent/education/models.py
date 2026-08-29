from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Subject(str, Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    UNKNOWN = "unknown"


class CognitiveLevel(str, Enum):
    RECOGNITION = "recognition"
    UNDERSTANDING = "understanding"
    APPLICATION = "application"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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
    message_count: int = Field(ge=1)
    subject: Subject | None = None
    updated_at: datetime


class ChatMessage(BaseModel):
    id: str | None = None
    user_id: str
    session_id: str
    request_id: str
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=50_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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


class ExamTrend(BaseModel):
    skill_id: str
    subject: Subject
    latest_weight: float = Field(ge=0.0, le=1.0)
    previous_weight: float = Field(ge=0.0, le=1.0)

    @property
    def weighted_prevalence(self) -> float:
        return 0.6 * self.latest_weight + 0.4 * self.previous_weight


class Recommendation(BaseModel):
    skill_id: str
    subject: Subject
    title: str
    score: float = Field(ge=0.0, le=1.0)
    mastery_gap: float
    exam_prevalence: float
    recent_error_rate: float
    reason: str


class ExamSource(BaseModel):
    id: str | None = None
    url: str
    publisher: str
    exam_year: int
    subject: Subject
    exam_code: str | None = None
    checksum: str | None = None
    attribution: str | None = None
    fetched_at: datetime | None = None


class ExamQuestion(BaseModel):
    id: str | None = None
    source_id: str | None = None
    exam_year: int
    subject: Subject
    exam_code: str | None = None
    question_number: str
    stem_markdown: str
    choices: list[str] = Field(default_factory=list)
    correct_answer: str | None = None
    skill_ids: list[str] = Field(default_factory=list)
    cognitive_level: CognitiveLevel = CognitiveLevel.UNDERSTANDING
    difficulty_estimate: int = Field(default=2, ge=1, le=5)
    difficulty_basis: str = "expert_estimate"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.PENDING
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class SkillExamDistribution(BaseModel):
    skill_id: str
    title: str
    question_count: int = Field(ge=0)
    weighted_prevalence: float = Field(ge=0.0, le=1.0)
    year_prevalence: dict[int, float] = Field(default_factory=dict)
    cognitive_counts: dict[str, int] = Field(default_factory=dict)
    average_difficulty: float = Field(ge=1.0, le=5.0)


class ExamResearchReport(BaseModel):
    subject: Subject
    generated_at: datetime
    timezone: str
    latest_completed_year: int
    years: list[int] = Field(default_factory=list, max_length=2)
    status: Literal["ready", "partial", "insufficient_data"]
    approved_question_count: int = Field(ge=0)
    coverage_by_year: dict[int, int] = Field(default_factory=dict)
    cognitive_distribution: dict[str, float] = Field(default_factory=dict)
    difficulty_distribution: dict[int, float] = Field(default_factory=dict)
    skills: list[SkillExamDistribution] = Field(default_factory=list)
    sources: list[ExamSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StudyPlanPriority(BaseModel):
    skill_id: str
    title: str
    priority_score: float = Field(ge=0.0, le=1.0)
    exam_prevalence: float = Field(ge=0.0, le=1.0)
    current_mastery: float | None = Field(default=None, ge=0.0, le=1.0)
    target_mastery: float = Field(ge=0.0, le=1.0)
    reason: str


class ScoreBandStudyPlan(BaseModel):
    band: str
    min_score: float = Field(ge=0.0, le=10.0)
    max_score: float = Field(ge=0.0, le=10.0)
    cognitive_focus: list[str]
    target_mastery: float = Field(ge=0.0, le=1.0)
    priorities: list[StudyPlanPriority]
    steps: list[str]
    personalization_confidence: float = Field(ge=0.0, le=1.0)


class PersonalizedExamPlan(BaseModel):
    research: ExamResearchReport
    score_bands: list[ScoreBandStudyPlan] = Field(default_factory=list)


class ReviewPatch(BaseModel):
    stem_markdown: str | None = None
    choices: list[str] | None = None
    correct_answer: str | None = None
    skill_ids: list[str] | None = None
    cognitive_level: CognitiveLevel | None = None
    difficulty_estimate: int | None = Field(default=None, ge=1, le=5)
    review_status: ReviewStatus | None = None
