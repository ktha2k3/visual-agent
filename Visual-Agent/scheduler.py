"""Register the idempotent weekly exam-source scan with RQ Scheduler."""

from datetime import datetime, timezone

from redis import Redis
from rq_scheduler import Scheduler

from exam.jobs import run_initial_sync, run_upload_cleanup
from utils.config import get_settings


SCHEDULE_ID = "weekly-exam-source-sync"
RETENTION_SCHEDULE_ID = "daily-upload-retention-cleanup"


def register_weekly_sync() -> None:
    settings = get_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL is required for the scheduler")
    scheduler = Scheduler(
        queue_name="exam-ingestion", connection=Redis.from_url(settings.redis_url)
    )
    existing = {job.meta.get("schedule_key") for job in scheduler.get_jobs()}
    if SCHEDULE_ID not in existing:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc), func=run_initial_sync,
            kwargs={"extract": True}, interval=7 * 24 * 60 * 60, repeat=None,
            meta={"schedule_key": SCHEDULE_ID}, result_ttl=86400, timeout=1800,
        )
    if RETENTION_SCHEDULE_ID not in existing:
        scheduler.schedule(
            scheduled_time=datetime.now(timezone.utc), func=run_upload_cleanup,
            interval=24 * 60 * 60, repeat=None,
            meta={"schedule_key": RETENTION_SCHEDULE_ID}, result_ttl=86400, timeout=600,
        )


if __name__ == "__main__":
    register_weekly_sync()
