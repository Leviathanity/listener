# Verified ASR format for llama.cpp server running Qwen3-ASR-1.7B Q8_0 GGUF + mmproj
# Server: http://192.168.2.118:8080
# Endpoint: POST /v1/chat/completions
# Model alias: "asr"
#
# === WORKING REQUEST FORMAT ===
# {
#     "model": "asr",
#     "messages": [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Please transcribe this audio."},
#             {"type": "input_audio", "input_audio": {"data": "<base64-wav>", "format": "wav"}}
#         ]
#     }]
# }
#
# Key details:
#   - content array MUST contain a text message AND an input_audio message
#   - input_audio.data: base64-encoded PCM WAV bytes (NOT a data URI, just raw base64)
#   - input_audio.format: "wav" or "mp3"
#   - Supported types in content[]: "text", "input_audio" (NOT "audio_url", "audio", "image_url")
#
# === RESPONSE FORMAT ===
# {
#     "choices": [{
#         "finish_reason": "stop",
#         "index": 0,
#         "message": {
#             "role": "assistant",
#             "content": "language <LANG_CODE><asr_text>TRANSCRIPTION_TEXT</asr_text>"
#         }
#     }],
#     "created": <unix_timestamp>,
#     "model": "qwen3-asr",
#     "system_fingerprint": "<fingerprint>",
#     "object": "chat.completion",
#     "usage": { "completion_tokens": <N>, "prompt_tokens": <N>, "total_tokens": <N> },
#     "id": "chatcmpl-...",
#     "timings": { ... }
# }
#
# The content format includes tags:
#   "language None<asr_text>...</asr_text>"   - when no language detected or no speech
#   "language en<asr_text>Hello world</asr_text>" - with detected language and transcription

import base64
import json
import wave
import numpy as np
import urllib.request
import urllib.error

ASR_URL = "http://192.168.2.118:8080"


def make_wav_file(path, samples, samplerate=16000):
    samples_int16 = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(samples_int16.tobytes())


def post_json(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, json.loads(body) if body else {}
    except Exception as e:
        return None, {"error": str(e)}


def test_asr_audio_format():
    samplerate = 16000
    duration = 2
    t = np.linspace(0, duration, samplerate * duration, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float64)

    wav_path = "test_audio.wav"
    make_wav_file(wav_path, samples, samplerate)

    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
    }

    print(f"Payload keys: {list(payload.keys())}")
    print(f"Content types: {[c['type'] for c in payload['messages'][0]['content']]}")
    print(f"Audio b64 length: {len(audio_b64)}")

    status, data = post_json(f"{ASR_URL}/v1/chat/completions", payload)
    print(f"Status: {status}")
    print(f"Body: {json.dumps(data, indent=2)}")

    assert status == 200, f"Expected 200, got {status}"
    assert "choices" in data, f"No choices in response: {data}"
    assert len(data["choices"]) > 0, "Empty choices"
    text = data["choices"][0]["message"]["content"]
    print(f"Transcription: '{text}'")
    print("ASR audio format VERIFIED")


if __name__ == "__main__":
    test_asr_audio_format()
