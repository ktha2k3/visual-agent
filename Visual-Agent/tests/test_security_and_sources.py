import io
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from services.storage import read_validated_image
from tools.vision.image_input import resolve_valid_image


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.asyncio
async def test_upload_checks_decoded_image_not_filename():
    upload = UploadFile(filename="homework.jpg", file=io.BytesIO(b"not an image"))
    with pytest.raises(HTTPException) as exc:
        await read_validated_image(upload, 10 * 1024 * 1024)
    assert exc.value.status_code == 415

    valid = UploadFile(filename="misleading.exe", file=io.BytesIO(_png_bytes()))
    data, mime, extension = await read_validated_image(valid, 10 * 1024 * 1024)
    assert data
    assert (mime, extension) == ("image/png", ".png")


@pytest.mark.asyncio
async def test_upload_rejects_oversize_before_decode():
    upload = UploadFile(filename="large.png", file=io.BytesIO(b"x" * 1025))
    with pytest.raises(HTTPException) as exc:
        await read_validated_image(upload, 1024)
    assert exc.value.status_code == 413


def test_legacy_vision_tool_rejects_url_and_path_escape():
    with pytest.raises(ValueError):
        resolve_valid_image("https://example.com/private.png")
    with pytest.raises(ValueError):
        resolve_valid_image("../../.env")
