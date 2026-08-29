from __future__ import annotations

import hashlib
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from exam.sources import is_allowed_source


MAX_SOURCE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class FetchedSource:
    final_url: str
    content: bytes
    content_type: str
    checksum: str
    artifact_urls: list[str]


class _ArtifactParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        values = dict(attrs)
        candidate = values.get("src") if tag in {"img", "source"} else values.get("href")
        if not candidate:
            return
        absolute = urljoin(self.base_url, candidate)
        clean = absolute.split("#", 1)[0]
        path = clean.lower().split("?", 1)[0]
        if path.endswith((".jpg", ".jpeg", ".png", ".webp", ".pdf")) and is_allowed_source(clean):
            self.urls.append(clean)


class SourceFetcher:
    async def fetch(self, url: str) -> FetchedSource:
        if not is_allowed_source(url):
            raise ValueError(f"Source host is not allowed: {url}")
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30, connect=10),
            headers={"User-Agent": "Visual-STEM-Tutor/0.2 (+education research ingestion)"},
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                if not is_allowed_source(str(response.url)):
                    raise ValueError("Source redirected outside the allowlist")
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_SOURCE_BYTES:
                        raise ValueError("Source exceeds 25 MB")
                    chunks.append(chunk)
                content = b"".join(chunks)
                content_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]
                final_url = str(response.url)
        artifacts: list[str] = []
        if content_type == "text/html":
            parser = _ArtifactParser(final_url)
            parser.feed(content.decode("utf-8", errors="replace"))
            artifacts = list(dict.fromkeys(parser.urls))
        return FetchedSource(
            final_url=final_url,
            content=content,
            content_type=content_type,
            checksum=hashlib.sha256(content).hexdigest(),
            artifact_urls=artifacts,
        )

    async def fetch_artifact(self, url: str) -> tuple[bytes, str, str]:
        fetched = await self.fetch(url)
        return fetched.content, fetched.content_type, fetched.checksum

