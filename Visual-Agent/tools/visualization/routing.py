from __future__ import annotations

from dataclasses import dataclass

from education.models import Subject


@dataclass(frozen=True)
class VisualizationDecision:
    should_visualize: bool
    subject: str | None
    reason: str


def decide_visualization(question: str, subject: Subject | str) -> VisualizationDecision:
    """Every supported tutoring turn requests an independent illustration."""
    value = subject.value if isinstance(subject, Subject) else str(subject)
    if value not in {Subject.MATH.value, Subject.PHYSICS.value}:
        return VisualizationDecision(False, None, "subject_not_supported")
    return VisualizationDecision(True, value, "automatic_illustration")
