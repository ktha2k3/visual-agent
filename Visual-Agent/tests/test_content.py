from services.content import split_content


def test_content_chunking_preserves_paragraphs():
    text = "Đoạn một.\n\nĐoạn hai dài hơn.\n\nĐoạn ba."
    chunks = split_content(text, max_chars=25)
    assert chunks == ["Đoạn một.", "Đoạn hai dài hơn.", "Đoạn ba."]
    assert all(len(item) <= 25 for item in chunks)
