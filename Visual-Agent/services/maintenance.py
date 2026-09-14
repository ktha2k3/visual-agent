"""Importable background maintenance tasks.

RQ persists the fully qualified function name. Keeping jobs outside an
entrypoint module prevents them from being serialized as ``__main__.*`` and
makes them importable by a separate worker process.
"""

import asyncio

from services.repository import get_repository
from services.storage import AssetStorage


def run_upload_cleanup() -> int:
    """Delete expired uploads and return the number of removed assets."""

    repository = get_repository()
    return asyncio.run(AssetStorage(repository).cleanup_expired())
