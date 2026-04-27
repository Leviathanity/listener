import base64
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path
from app.asr import AsrClient


@pytest.fixture
def asr_client():
    return AsrClient(base_url="http://192.168.2.118:8080", model="asr")


def create_test_wav(path, duration=1.0):
    sr = 16000
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.5).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")
    return path


def test_asr_client_init(asr_client):
    assert asr_client.base_url == "http://192.168.2.118:8080"
    assert asr_client.model == "asr"
    assert asr_client._semaphore._value == 4


def test_build_payload(asr_client, tmp_path):
    wav_path = create_test_wav(str(tmp_path / "payload_test.wav"), duration=1.0)
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = asr_client._build_payload(audio_b64)
    assert payload["model"] == "asr"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert len(payload["messages"][0]["content"]) == 2
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert payload["messages"][0]["content"][1]["type"] == "input_audio"
    assert payload["messages"][0]["content"][1]["input_audio"]["format"] == "wav"


def test_parse_response():
    from app.asr import AsrClient
    client = AsrClient()
    text = client._parse_response("language en<asr_text>Hello world</asr_text>")
    assert text == "Hello world"

    text2 = client._parse_response("language zh<asr_text>你好世界</asr_text>")
    assert text2 == "你好世界"

    text3 = client._parse_response("language None<asr_text></asr_text>")
    assert text3 == ""


def test_parse_response_strips_special_tokens():
    client = AsrClient()
    result = client._parse_response("language zh<asr_text>Hello <|audio_eos|> world</asr_text>")
    assert "<|audio_eos|>" not in result
    assert "Hello  world" in result


@pytest.mark.asyncio
async def test_transcribe_segment_integration(asr_client, tmp_path):
    """Integration test — requires running ASR server."""
    wav_path = create_test_wav(str(tmp_path / "test.wav"), duration=2.0)
    try:
        text = await asr_client.transcribe(wav_path)
        assert isinstance(text, str)
    except Exception as e:
        pytest.skip(f"ASR server not reachable: {e}")
