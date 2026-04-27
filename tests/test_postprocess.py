from app.postprocess import clean_text


def test_clean_text_removes_char_repeats():
    result = clean_text("a" * 25 + "b" * 25 + "cccc")
    assert len(result) < 25
    assert "a" in result
    assert "b" in result


def test_clean_text_removes_pattern_repeats():
    text = "你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好"
    result = clean_text(text)
    assert len(result) < len(text)


def test_clean_text_preserves_normal_text():
    text = "今天天气很好，我们开会讨论项目进展。"
    result = clean_text(text)
    assert result == text


def test_clean_text_handles_empty():
    assert clean_text("") == ""


def test_clean_text_handles_short_text():
    assert clean_text("你好") == "你好"


def test_clean_text_removes_long_pattern_repeats():
    text = "在这里的话，我在这里有没有什么危险？没有吧？" * 25
    result = clean_text(text)
    assert len(result) < len(text)
