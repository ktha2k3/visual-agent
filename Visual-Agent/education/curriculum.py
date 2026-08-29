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
    Skill("chemistry.esters", Subject.CHEMISTRY, "Ester và lipid", "organic"),
    Skill("chemistry.carbohydrates", Subject.CHEMISTRY, "Carbohydrate", "organic"),
    Skill("chemistry.amines", Subject.CHEMISTRY, "Hợp chất chứa nitrogen", "organic"),
    Skill("chemistry.polymers", Subject.CHEMISTRY, "Polymer", "materials"),
    Skill("chemistry.electrochemistry", Subject.CHEMISTRY, "Pin điện và điện phân", "electrochemistry"),
    Skill("biology.genetics", Subject.BIOLOGY, "Di truyền học", "genetics"),
    Skill("biology.evolution", Subject.BIOLOGY, "Tiến hóa", "evolution"),
    Skill("biology.ecology", Subject.BIOLOGY, "Sinh thái học", "ecology"),
    Skill("biology.cells", Subject.BIOLOGY, "Sinh học tế bào", "cell_biology"),
    Skill("biology.biotechnology", Subject.BIOLOGY, "Công nghệ sinh học", "biotechnology"),
)

SKILL_BY_ID = {skill.id: skill for skill in SKILLS}


def skills_for_subject(subject: Subject) -> list[Skill]:
    return [skill for skill in SKILLS if skill.subject == subject]

