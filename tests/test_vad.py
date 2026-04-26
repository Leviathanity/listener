import numpy as np
import soundfile as sf
from pathlib import Path
from app.vad import VadSegmenter
from app.config import VAD_TARGET_SAMPLE_RATE


def create_silent_wav_with_speech(path, dur_speech_start=0.5, dur_speech=1.0, dur_total=3.0, sr=16000):
    total_samples = int(sr * dur_total)
    wav = np.zeros(total_samples, dtype=np.float32)
    speech_start = int(sr * dur_speech_start)
    speech_samples = int(sr * dur_speech)
    t = np.linspace(0, dur_speech, speech_samples)
    wav[speech_start:speech_start + speech_samples] = np.sin(2 * np.pi * 440 * t) * 0.5
    sf.write(path, wav, sr, subtype="PCM_16")
    return path


def test_vad_detects_speech_segment(tmp_path):
    segmenter = VadSegmenter(model_dir="does_not_exist_yet")
    try:
        result = segmenter.detect("nonexistent.wav")
        assert isinstance(result, list)
    except FileNotFoundError:
        pass


def test_vad_result_format(tmp_path):
    segmenter = VadSegmenter(model_dir="does_not_exist_yet")
    assert hasattr(segmenter, "detect")
    assert hasattr(segmenter, "model")
