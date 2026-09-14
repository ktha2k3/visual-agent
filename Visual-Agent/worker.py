"""RQ worker entrypoint: ``python worker.py``."""

from redis import Redis
from rq import Queue, Worker

from utils.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL is required for the worker")
    connection = Redis.from_url(settings.redis_url)
    Worker(
        [
            Queue("knowledge-ingestion", connection=connection),
        ],
        connection=connection,
    ).work()
