from __future__ import annotations

import asyncio
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from fastapi import UploadFile


ALLOWED_KNOWLEDGE_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}
MAX_EXTRACTED_CHARACTERS = 1_000_000
MIME_BY_EXTENSION = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class KnowledgeFileError(ValueError):
    def __init__(self, detail: str, status_code: int = 422):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "word/document.xml" not in archive.namelist():
                raise KnowledgeFileError("File is not a valid DOCX document", 415)
            if archive.getinfo("word/document.xml").file_size > 5_000_000:
                raise KnowledgeFileError("DOCX document content is too large", 413)
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise KnowledgeFileError("File is not a valid DOCX document", 415) from exc

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{namespace}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{namespace}tab":
                pieces.append("\t")
            elif node.tag == f"{namespace}br":
                pieces.append("\n")
        value = "".join(pieces).strip()
        if value:
            paragraphs.append(value)
    return "\n\n".join(paragraphs)


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - guarded by project dependencies
        raise KnowledgeFileError("PDF extraction is not installed", 503) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise KnowledgeFileError("Encrypted PDF files are not supported", 422)
        if len(reader.pages) > 500:
            raise KnowledgeFileError("PDF exceeds the 500-page limit", 413)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except KnowledgeFileError:
        raise
    except Exception as exc:
        raise KnowledgeFileError("File is not a readable PDF document", 415) from exc


def extract_document_text(data: bytes, filename: str) -> tuple[str, str]:
    """Validate a supported document by content and extract normalized plain text."""
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_KNOWLEDGE_EXTENSIONS:
        raise KnowledgeFileError(
            "Supported knowledge files are TXT, Markdown, PDF, and DOCX", 415
        )

    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise KnowledgeFileError("File content does not match PDF", 415)
        text, mime_type = _extract_pdf(data), "application/pdf"
    elif extension == ".docx":
        if not data.startswith(b"PK"):
            raise KnowledgeFileError("File content does not match DOCX", 415)
        text = _extract_docx(data)
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        if b"\x00" in data:
            raise KnowledgeFileError("Text document contains binary data", 415)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise KnowledgeFileError("Text documents must use UTF-8 encoding", 415) from exc
        mime_type = "text/markdown" if extension in {".md", ".markdown"} else "text/plain"

    text = _normalize_text(text)
    if len(text) < 20:
        raise KnowledgeFileError("Document has too little extractable text")
    if len(text) > MAX_EXTRACTED_CHARACTERS:
        raise KnowledgeFileError(
            f"Extracted text exceeds {MAX_EXTRACTED_CHARACTERS:,} characters", 413
        )
    return text, mime_type


async def read_knowledge_upload(
    upload: UploadFile, max_bytes: int
) -> tuple[str, str, str, int]:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise KnowledgeFileError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit", 413)
    if not data:
        raise KnowledgeFileError("Uploaded document is empty")
    filename = Path(upload.filename or "document").name
    text, mime_type = await asyncio.to_thread(extract_document_text, data, filename)
    return text, mime_type, filename, len(data)


async def read_knowledge_payload(
    upload: UploadFile, max_bytes: int
) -> tuple[bytes, str, str]:
    """Read a bounded upload for durable staging; extraction happens in the worker."""
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise KnowledgeFileError(
            f"File exceeds the {max_bytes // (1024 * 1024)} MB limit", 413
        )
    if not data:
        raise KnowledgeFileError("Uploaded document is empty")
    filename = Path(upload.filename or "document").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_KNOWLEDGE_EXTENSIONS:
        raise KnowledgeFileError(
            "Supported knowledge files are TXT, Markdown, PDF, and DOCX", 415
        )
    if extension == ".pdf" and not data.startswith(b"%PDF-"):
        raise KnowledgeFileError("File content does not match PDF", 415)
    if extension == ".docx" and not data.startswith(b"PK"):
        raise KnowledgeFileError("File content does not match DOCX", 415)
    if extension in {".txt", ".md", ".markdown"} and b"\x00" in data:
        raise KnowledgeFileError("Text document contains binary data", 415)
    return data, filename, MIME_BY_EXTENSION[extension]
