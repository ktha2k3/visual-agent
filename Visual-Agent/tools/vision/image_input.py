from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 25 * 1024 * 1024
MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
}


def allowed_asset_root() -> Path:
    configured = os.getenv("VISION_ASSET_ROOT")
    return (
        Path(configured).resolve()
        if configured
        else (Path(__file__).resolve().parents[2] / "assets").resolve()
    )


def resolve_valid_image(raw_path: str) -> tuple[Path, str]:
    """Resolve only decoded images contained by the configured asset directory."""
    if "://" in raw_path or raw_path.lower().startswith("file:"):
        raise ValueError("URLs and file URIs are not accepted by vision tools")
    root = allowed_asset_root()
    candidate = Path(raw_path)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Image path is outside the allowed asset directory")
    if not path.is_file():
        raise FileNotFoundError("Image file was not found")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 25 MB limit")
    try:
        with Image.open(path) as image:
            image.verify()
            detected = image.format
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File is not a valid image") from exc
    if detected not in MIME_BY_FORMAT:
        raise ValueError("Unsupported image format")
    return path, MIME_BY_FORMAT[detected]
