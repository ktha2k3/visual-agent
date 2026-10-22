from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from utils.model_factory import get_chat_model


TModel = TypeVar("TModel", bound=BaseModel)


class LLMProvider(ABC):
    """Provider boundary used by every reasoning and repair stage."""

    @abstractmethod
    async def structured(
        self, schema: type[TModel], *, system: str, prompt: str
    ) -> TModel: ...

    @abstractmethod
    async def text(self, *, system: str, prompt: str) -> str: ...


class OllamaProvider(LLMProvider):
    """Configured visualization model through its OpenAI-compatible API.

    The legacy class name remains for existing callers; endpoint and credentials
    are selected by model_factory (River for visualization).
    """

    def __init__(self, model=None):
        self.model = model or get_chat_model("visualization", temperature=0.1)

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError("Model response did not contain JSON")
        depth = 0
        quoted = False
        escaped = False
        for index, character in enumerate(text[start:], start=start):
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
                continue
            if character == '"':
                quoted = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise ValueError("Model response contained incomplete JSON")

    async def structured(
        self, schema: type[TModel], *, system: str, prompt: str
    ) -> TModel:
        runnable = self.model.with_structured_output(
            schema, method="function_calling", strict=True
        )
        try:
            result = await runnable.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
            return schema.model_validate(result)
        except Exception:
            # Some OpenAI-compatible deployments do not implement tool calling.
            # JSON-schema prompting keeps the provider functional without changing
            # any downstream pipeline contracts.
            fallback = await self.model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            system
                            + "\nChỉ trả đúng một JSON object theo JSON Schema sau:\n"
                            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw = self._content_text(fallback.content)
            return schema.model_validate_json(self._extract_json_object(raw))

    async def text(self, *, system: str, prompt: str) -> str:
        result = await self.model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        return self._content_text(result.content)
