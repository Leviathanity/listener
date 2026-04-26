# tests/test_integration.py
"""
Full pipeline integration test using mocks for external services.

Run: pytest tests/test_integration.py -v
"""
import json
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.task_tracker import TaskTracker
from app.pipeline import process_task


def create_test_wav(path, duration=3.0):
    sr = 16000
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.5).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(tmp_path):
    wav_path = tmp_path / "test.wav"
    create_test_wav(str(wav_path), duration=3.0)

    result_dir = tmp_path / "results"
    chunk_dir = tmp_path / "chunks"
    result_dir.mkdir()
    chunk_dir.mkdir()

    tracker = TaskTracker(str(tmp_path / "test.db"))
    await tracker.init()
    task_id = "integration-test-1"
    await tracker.create(task_id, "test.wav", str(wav_path))

    vad = MagicMock()
    vad.detect.return_value = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    asr = AsyncMock()
    asr.transcribe.return_value = "集成测试文本"

    await process_task(
        task_id=task_id,
        file_path=str(wav_path),
        tracker=tracker,
        vad_segmenter=vad,
        asr_client=asr,
        chunk_dir=str(chunk_dir),
        result_dir=str(result_dir),
    )

    task = await tracker.get(task_id)
    assert task["status"] == "completed"
    assert task["progress"] == 1.0

    assert asr.transcribe.call_count == 3
    assert vad.detect.call_count == 1

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)

    assert result["task_id"] == task_id
    assert result["status"] == "completed"
    assert len(result["segments"]) == 3
    assert "集成测试文本" in result["full_text"]

    assert not (Path(chunk_dir) / task_id).exists()

    await tracker.close()


@pytest.mark.asyncio
async def test_full_pipeline_no_speech(tmp_path):
    wav_path = tmp_path / "silent.wav"
    create_test_wav(str(wav_path), duration=1.0)

    result_dir = tmp_path / "results"
    chunk_dir = tmp_path / "chunks"
    result_dir.mkdir()
    chunk_dir.mkdir()

    tracker = TaskTracker(str(tmp_path / "test.db"))
    await tracker.init()
    task_id = "integration-test-2"
    await tracker.create(task_id, "silent.wav", str(wav_path))

    vad = MagicMock()
    vad.detect.return_value = []

    asr = AsyncMock()

    await process_task(
        task_id=task_id,
        file_path=str(wav_path),
        tracker=tracker,
        vad_segmenter=vad,
        asr_client=asr,
        chunk_dir=str(chunk_dir),
        result_dir=str(result_dir),
    )

    task = await tracker.get(task_id)
    assert task["status"] == "completed"

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)
    assert result["status"] == "no_speech"
    assert asr.transcribe.call_count == 0

    await tracker.close()


@pytest.mark.asyncio
async def test_full_pipeline_partial_failure(tmp_path):
    wav_path = tmp_path / "partial.wav"
    create_test_wav(str(wav_path), duration=3.0)

    result_dir = tmp_path / "results"
    chunk_dir = tmp_path / "chunks"
    result_dir.mkdir()
    chunk_dir.mkdir()

    tracker = TaskTracker(str(tmp_path / "test.db"))
    await tracker.init()
    task_id = "integration-test-3"
    await tracker.create(task_id, "partial.wav", str(wav_path))

    vad = MagicMock()
    vad.detect.return_value = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    asr = AsyncMock()
    async def transcribe_side_effect(path):
        if "chunk_0001" in path:
            raise Exception("Simulated failure")
        return "success"
    asr.transcribe = transcribe_side_effect

    await process_task(
        task_id=task_id,
        file_path=str(wav_path),
        tracker=tracker,
        vad_segmenter=vad,
        asr_client=asr,
        chunk_dir=str(chunk_dir),
        result_dir=str(result_dir),
    )

    task = await tracker.get(task_id)
    assert task["status"] == "completed"

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)

    assert len(result["segments"]) == 2
    assert "warning" in result

    await tracker.close()
