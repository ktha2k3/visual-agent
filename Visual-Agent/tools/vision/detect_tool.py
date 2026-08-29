import os
import re
from functools import lru_cache
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.vision.image_input import resolve_valid_image

_YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo11x.pt")


# ── Path extraction (same regex as describe_tool) ─────────────────────────────
_PATH_PATTERNS = [
    re.compile(r"Image:\s*(\S+)", re.IGNORECASE),
    re.compile(
        r"([A-Za-z]:[\\/][\w\-.\\/]+\.(?:jpe?g|png|gif|webp|bmp))", re.IGNORECASE
    ),
    re.compile(r"(/[\w/.\-]+\.(?:jpe?g|png|gif|webp|bmp))", re.IGNORECASE),
]


def _extract_path(text: str) -> str | None:
    for pattern in _PATH_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


# ── YOLO model (loaded once) ───────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_yolo():
    from ultralytics import YOLO

    return YOLO(_YOLO_MODEL_PATH)


# ── Tool ───────────────────────────────────────────────────────────────────────
class _DetectionInput(BaseModel):
    text: str = Field(
        ..., description="Path inside VISION_ASSET_ROOT for optional object detection."
    )


@tool("object_detection_and_counting", args_schema=_DetectionInput)
def detect_and_count_tool(
    text: Annotated[str, "Text containing a path inside VISION_ASSET_ROOT."],
) -> dict | str:
    """Detect and count objects in an image. Returns counts per class and bounding boxes."""
    image_path = _extract_path(text)
    if not image_path:
        return "No image path found in the input."

    try:
        image_path, _ = resolve_valid_image(image_path)
        model = _get_yolo()
        results = model(str(image_path), verbose=False)
    except Exception as exc:
        return f"Object detection failed: {exc}"

    detections, counts = [], {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            detections.append(
                {
                    "class": name,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": list(map(int, box.xyxy[0])),
                }
            )
            counts[name] = counts.get(name, 0) + 1

    return {"counts": counts, "detections": detections}
