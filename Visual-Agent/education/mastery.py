from __future__ import annotations

from education.models import MasteryRecord, MasteryUpdate, Subject


EVIDENCE_BY_OUTCOME = {
    "correct_unassisted": 1.0,
    "correct_hint_1": 0.75,
    "correct_hint_2": 0.5,
    "full_solution": 0.25,
    "incorrect": 0.0,
}


def evidence_for_outcome(
    *, answer_correct: bool | None, hint_level: int, used_full_solution: bool
) -> float | None:
    if answer_correct is None:
        return None
    if not answer_correct:
        return EVIDENCE_BY_OUTCOME["incorrect"]
    if used_full_solution or hint_level >= 3:
        return EVIDENCE_BY_OUTCOME["full_solution"]
    if hint_level == 2:
        return EVIDENCE_BY_OUTCOME["correct_hint_2"]
    if hint_level == 1:
        return EVIDENCE_BY_OUTCOME["correct_hint_1"]
    return EVIDENCE_BY_OUTCOME["correct_unassisted"]


def update_mastery(
    *,
    user_id: str,
    skill_id: str,
    subject: Subject,
    old_mastery: float | None,
    evidence: float,
) -> tuple[MasteryRecord, MasteryUpdate]:
    old = 0.5 if old_mastery is None else max(0.0, min(1.0, old_mastery))
    new = round(max(0.0, min(1.0, 0.7 * old + 0.3 * evidence)), 4)
    record = MasteryRecord(
        user_id=user_id,
        skill_id=skill_id,
        subject=subject,
        mastery=new,
        attempts=1,
        recent_errors=0 if evidence > 0 else 1,
    )
    return record, MasteryUpdate(
        skill_id=skill_id,
        subject=subject,
        old_mastery=old,
        new_mastery=new,
        evidence=evidence,
    )

