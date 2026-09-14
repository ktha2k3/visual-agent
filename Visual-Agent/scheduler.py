"""Register maintenance jobs with RQ Scheduler."""

from datetime import datetime, timezone

from redis import Redis
from rq_scheduler import Scheduler

from services.maintenance import run_upload_cleanup
from utils.config import get_settings


RETENTION_SCHEDULE_ID = "daily-upload-retention-cleanup"
LEGACY_EXAM_SCHEDULE_ID = "weekly-exam-source-sync"

def register_schedules() -> None:
    settings = get_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL is required for the scheduler")
    scheduler = Scheduler(
        queue_name="knowledge-ingestion", connection=Redis.from_url(settings.redis_url)
    )
    scheduled_jobs = scheduler.get_jobs()
    for job in scheduled_jobs:
        if job.meta.get("schedule_key") == LEGACY_EXAM_SCHEDULE_ID:
            scheduler.cancel(job)
    existing = {
        job.meta.get("schedule_key")
        for job in scheduled_jobs
        if job.meta.get("schedule_key") != LEGACY_EXAM_SCHEDULE_ID
    }
    if RETENTION_SCHEDULE_ID not in existing:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc), func=run_upload_cleanup,
            interval=24 * 60 * 60, repeat=None,
            meta={"schedule_key": RETENTION_SCHEDULE_ID}, result_ttl=86400, timeout=600,
        )


if __name__ == "__main__":
    register_schedules()
