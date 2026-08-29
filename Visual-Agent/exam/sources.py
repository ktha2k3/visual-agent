from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

from education.models import ExamSource, Subject
from utils.config import Settings, get_settings


@dataclass(frozen=True)
class SourceDefinition:
    url: str
    publisher: str
    exam_year: int
    subject: Subject
    exam_code: str | None = None
    attribution: str | None = None

    def to_model(self) -> ExamSource:
        return ExamSource(
            url=self.url,
            publisher=self.publisher,
            exam_year=self.exam_year,
            subject=self.subject,
            exam_code=self.exam_code,
            attribution=self.attribution,
        )


_ANSWER_2025 = "https://vqa.moet.gov.vn/vi/news/tin-tuc-su-kien/cong-bo-dap-an-cac-mon-thi-cua-ky-thi-tot-nghiep-thpt-2025-208.html"
_EXAM_2026 = "https://xaydungchinhsach.chinhphu.vn/lich-thi-tot-nghiep-thpt-nam-2026-119260319121525172.htm"

INITIAL_SOURCES: tuple[SourceDefinition, ...] = tuple(
    SourceDefinition(
        url=_ANSWER_2025,
        publisher="Cục Quản lý Chất lượng - Bộ GDĐT",
        exam_year=2025,
        subject=subject,
        exam_code="representative",
        attribution="Bộ Giáo dục và Đào tạo",
    )
    for subject in (Subject.MATH, Subject.PHYSICS, Subject.CHEMISTRY, Subject.BIOLOGY)
) + tuple(
    SourceDefinition(
        url=_EXAM_2026,
        publisher="Cổng Thông tin điện tử Chính phủ",
        exam_year=2026,
        subject=subject,
        exam_code="representative",
        attribution="Báo điện tử Chính phủ / Bộ Giáo dục và Đào tạo",
    )
    for subject in (Subject.MATH, Subject.PHYSICS, Subject.CHEMISTRY, Subject.BIOLOGY)
)


def is_allowed_source(url: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in settings.source_hosts)


def rolling_completed_years(current_year: int, available_years: list[int]) -> list[int]:
    return sorted({year for year in available_years if year <= current_year}, reverse=True)[:2]


def latest_completed_exam_year(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def recent_source_definitions(today: date | None = None) -> tuple[SourceDefinition, ...]:
    """Select the two latest completed exams represented in the source registry."""
    today = today or date.today()
    # The national exam is not considered complete before July.
    latest_completed_year = latest_completed_exam_year(today)
    years = rolling_completed_years(
        latest_completed_year, [item.exam_year for item in INITIAL_SOURCES]
    )
    return tuple(item for item in INITIAL_SOURCES if item.exam_year in years)
