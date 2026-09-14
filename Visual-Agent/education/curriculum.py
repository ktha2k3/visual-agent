"""A compact, reviewable CTGDPT 2018-aligned seed taxonomy for grade 12.

These nodes are identifiers and topic labels, not copied textbook content.
They provide enough breadth for the MVP and can be expanded through migrations.
"""

from __future__ import annotations

from dataclasses import dataclass

from education.models import Subject


@dataclass(frozen=True)
class Skill:
    id: str
    subject: Subject
    title: str
    domain: str


SKILLS: tuple[Skill, ...] = (
    Skill("math.functions", Subject.MATH, "Hàm số và đồ thị", "calculus"),
    Skill("math.derivatives", Subject.MATH, "Đạo hàm và ứng dụng", "calculus"),
    Skill("math.integrals", Subject.MATH, "Nguyên hàm và tích phân", "calculus"),
    Skill("math.spatial_geometry", Subject.MATH, "Hình học không gian", "geometry"),
    Skill("math.probability", Subject.MATH, "Xác suất và thống kê", "probability"),
    Skill("physics.thermal", Subject.PHYSICS, "Vật lí nhiệt", "thermal"),
    Skill("physics.gases", Subject.PHYSICS, "Khí lí tưởng", "thermal"),
    Skill("physics.electromagnetism", Subject.PHYSICS, "Điện từ học", "electromagnetism"),
    Skill("physics.ac", Subject.PHYSICS, "Dòng điện xoay chiều", "electricity"),
    Skill("physics.nuclear", Subject.PHYSICS, "Vật lí hạt nhân", "modern_physics"),
)

SKILL_BY_ID = {skill.id: skill for skill in SKILLS}


def skills_for_subject(subject: Subject) -> list[Skill]:
    return [skill for skill in SKILLS if skill.subject == subject]
