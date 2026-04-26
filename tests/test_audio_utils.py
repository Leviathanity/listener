import numpy as np
import soundfile as sf
from pathlib import Path
from app.audio_utils import load_audio, save_wav, chunk_audio


def create_test_wav(path, duration=1.0, sr=16000):
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.5).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")
    return path


def test_load_audio_16k_wav(tmp_path):
    wav_path = create_test_wav(str(tmp_path / "test.wav"), duration=1.0, sr=16000)
    data = load_audio(wav_path)
    assert data.ndim == 1
    assert len(data) == 16000
    assert data.dtype == np.float32


def test_save_wav(tmp_path):
    samples = np.ones(8000, dtype=np.float32) * 0.5
    out_path = str(tmp_path / "out.wav")
    save_wav(samples, out_path)
    assert Path(out_path).exists()
    data, sr = sf.read(out_path, dtype="float32")
    assert sr == 16000
    assert len(data) == 8000


def test_chunk_audio_basic():
    wav = np.zeros(480000, dtype=np.float32)
    timestamps = [(0.0, 5.0), (5.0, 15.0), (15.0, 30.0)]
    chunks = chunk_audio(wav, 16000, timestamps, target_duration_s=10, max_duration_s=30)
    assert len(chunks) == 3
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 5.0


def test_chunk_audio_splits_long_segments():
    wav = np.zeros(960000, dtype=np.float32)
    timestamps = [(0.0, 60.0)]
    chunks = chunk_audio(wav, 16000, timestamps, target_duration_s=10, max_duration_s=30)
    assert len(chunks) == 2
    for chunk in chunks:
        duration = chunk[1] - chunk[0]
        assert duration <= 30.0


def test_chunk_audio_short_audio():
    wav = np.zeros(8000, dtype=np.float32)
    timestamps = [(0.0, 0.5)]
    chunks = chunk_audio(wav, 16000, timestamps, target_duration_s=10, max_duration_s=30)
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 0.5


def test_chunk_audio_empty_timestamps():
    wav = np.zeros(16000, dtype=np.float32)
    chunks = chunk_audio(wav, 16000, [], target_duration_s=120, max_duration_s=180)
    assert len(chunks) == 1
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 1.0
