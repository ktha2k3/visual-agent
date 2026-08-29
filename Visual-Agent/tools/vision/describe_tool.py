import base64
import asyncio
import re
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import (AsyncCallbackManagerForToolRun,
                                      CallbackManagerForToolRun)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from utils.model_factory import get_chat_model
from tools.vision.image_input import resolve_valid_image

_llm = get_chat_model("vision")


# ── Path extraction (regex — faster and more reliable than an LLM call) ────────
_PATH_PATTERNS = [
    re.compile(
        r"Image:\s*(\S+)", re.IGNORECASE
    ),  # "Image: /path/to/img.jpg" or Windows path
    re.compile(r"([A-Za-z]:[\\/][\w\-.\\/]+\.(?:jpe?g|png|gif|webp|bmp))", re.IGNORECASE),
    re.compile(r"(/[-A-Za-z0-9_./]+\.(?:jpe?g|png|gif|webp|bmp))", re.IGNORECASE),
]


def _extract_path(text: str) -> str | None:
    for pattern in _PATH_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return None


# ── Image encoding ─────────────────────────────────────────────────────────────
def _encode_image(image_path: str) -> tuple[str | None, str | None]:
    """Return (base64_string, mime_type) or (None, None) on failure."""
    try:
        path, mime = resolve_valid_image(image_path)
    except (ValueError, FileNotFoundError):
        return None, None
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode(), mime


# ── Describer chain ────────────────────────────────────────────────────────────
class _ImageDescription(BaseModel):
    image_description: str = Field(
        ..., description="A detailed description of the image content."
    )


_desc_parser = PydanticOutputParser(pydantic_object=_ImageDescription)


def _build_describer_messages(image_path: str) -> list:
    b64, mime = _encode_image(image_path)
    if not b64:
        raise FileNotFoundError(f"Cannot read image at path: {image_path!r}")
    return [
        SystemMessage(
            content=(
                "You are an image analysis assistant. Analyze the image and describe it in detail.\n"
                f"Image MIME type: {mime}\n"
                "Focus on: objects and positions, colors/lighting/textures, actions, context. "
                "Transcribe any visible text accurately.\n"
                f"{_desc_parser.get_format_instructions()}"
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please analyze the image and provide a detailed description.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
        ),
    ]


# ── LangChain tool ─────────────────────────────────────────────────────────────
class _ImageDescriberInput(BaseModel):
    text: str = Field(
        ..., description="Text containing a path inside VISION_ASSET_ROOT."
    )


class ImageDescriberTool(BaseTool):
    name: str = "image_describer"
    description: str = (
        "Analyze and describe the content of an image. "
        "Input must reference an image inside VISION_ASSET_ROOT."
    )
    args_schema: Optional[ArgsSchema] = _ImageDescriberInput
    return_direct: bool = True

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        image_path = _extract_path(text)
        if not image_path:
            return "No image path found in the input."

        # Check existence before running the expensive LLM chain
        try:
            messages = _build_describer_messages(image_path)
        except FileNotFoundError as exc:
            return str(exc)

        try:
            result = _llm.invoke(messages)
            parsed = _desc_parser.parse(result.content)
            return parsed.image_description
        except Exception as exc:
            return f"Failed to analyze the image: {exc}"

    async def _arun(
        self, text: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        image_path = _extract_path(text)
        if not image_path:
            return "No image path found in the input."
        try:
            messages = await asyncio.to_thread(_build_describer_messages, image_path)
            result = await _llm.ainvoke(messages)
            return _desc_parser.parse(result.content).image_description
        except Exception as exc:
            return f"Failed to analyze the image: {exc}"


image_describer_tool = ImageDescriberTool()
