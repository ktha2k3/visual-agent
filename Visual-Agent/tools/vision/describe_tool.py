import base64
import os
import re
from typing import Optional

import magic
import requests
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "ollama")

_llm = ChatOpenAI(model="gemma4:31b-cloud", base_url=_OLLAMA_BASE_URL, api_key=_OLLAMA_API_KEY)


# ── Path extraction (regex — faster and more reliable than an LLM call) ────────
_PATH_PATTERNS = [
    re.compile(r'Image:\s*(\S+)',                    re.IGNORECASE),   # "Image: /path/to/img.jpg"
    re.compile(r'(/[\w/.\-]+\.(?:jpe?g|png|gif|webp|bmp))', re.IGNORECASE),  # bare absolute path
    re.compile(r'(https?://\S+\.(?:jpe?g|png|gif|webp|bmp)\S*)', re.IGNORECASE),  # URL
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
    if image_path.startswith("http"):
        try:
            resp = requests.get(image_path, stream=True, timeout=15)
            if resp.status_code != 200:
                return None, None
            mime = resp.headers.get("Content-Type", "image/jpeg")
            return base64.b64encode(resp.content).decode(), mime
        except requests.RequestException as exc:
            print(f"[describe_tool] download error: {exc}")
            return None, None
    else:
        if not os.path.exists(image_path):
            print(f"[describe_tool] file not found: {image_path!r}")
            return None, None
        mime = magic.Magic(mime=True).from_file(image_path)
        if not mime.startswith("image/"):
            print(f"[describe_tool] not an image MIME ({mime}): {image_path!r}")
            return None, None
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode(), mime


# ── Describer chain ────────────────────────────────────────────────────────────
class _ImageDescription(BaseModel):
    image_description: str = Field(..., description="A detailed description of the image content.")

_desc_parser = PydanticOutputParser(pydantic_object=_ImageDescription)


def _build_describer_messages(image_path: str) -> list:
    b64, mime = _encode_image(image_path)
    if not b64:
        raise FileNotFoundError(f"Cannot read image at path: {image_path!r}")
    return [
        SystemMessage(content=(
            "You are an image analysis assistant. Analyze the image and describe it in detail.\n"
            f"Image MIME type: {mime}\n"
            "Focus on: objects and positions, colors/lighting/textures, actions, context. "
            "Transcribe any visible text accurately.\n"
            f"{_desc_parser.get_format_instructions()}"
        )),
        HumanMessage(content=[
            {"type": "text", "text": "Please analyze the image and provide a detailed description."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]),
    ]


# ── LangChain tool ─────────────────────────────────────────────────────────────
class _ImageDescriberInput(BaseModel):
    text: str = Field(..., description="Text input that may contain an image path or URL.")


class ImageDescriberTool(BaseTool):
    name: str = "image_describer"
    description: str = (
        "Analyze and describe the content of an image. "
        "Input should be text containing an image path or URL."
    )
    args_schema: Optional[ArgsSchema] = _ImageDescriberInput
    return_direct: bool = True

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        image_path = _extract_path(text)
        if not image_path:
            return "No image path or URL found in the input."

        # Check existence before running the expensive LLM chain
        if not image_path.startswith("http") and not os.path.exists(image_path):
            return f"Image file not found: {image_path!r}. Please verify the path and try again."

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

    async def _arun(self, text: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return self._run(text)


image_describer_tool = ImageDescriberTool()
