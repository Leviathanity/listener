import numpy as np
import soundfile as sf
import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from app.pipeline import process_task
from app.task_tracker import TaskTracker


def create_test_audio(path, duration=2.0):
    sr = 16000
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.5).astype(np.float32)
    sf.write(path, samples, sr, subtype="PCM_16")
    return path


@pytest.mark.asyncio
async def test_process_task_completes_with_mocks(tmp_path):
    upload_path = create_test_audio(str(tmp_path / "test.wav"), duration=2.0)
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    tracker = TaskTracker(str(tmp_path / "test.db"))
    await tracker.init()
    task_id = "test-task-1"
    await tracker.create(task_id, "test.wav", upload_path)

    vad_segmenter = MagicMock()
    vad_segmenter.detect.return_value = [(0.0, 5.0), (7.0, 12.0)]

    asr_client = AsyncMock()
    asr_client.transcribe.return_value = "\u6d4b\u8bd5\u6587\u672c"

    with patch("app.pipeline.load_audio", return_value=np.zeros(192000, dtype=np.float32)):
        with patch("app.pipeline.clean_text", return_value="\u6d4b\u8bd5\u6587\u672c"):
            with patch("app.pipeline.save_wav"):
                await process_task(
                    task_id=task_id,
                    file_path=upload_path,
                    tracker=tracker,
                    vad_segmenter=vad_segmenter,
                    asr_client=asr_client,
                    chunk_dir=str(tmp_path / "chunks"),
                    result_dir=str(result_dir),
                )

    task = await tracker.get(task_id)
    assert task["status"] == "completed"
    assert task["result_path"] is not None

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)
    assert result["task_id"] == task_id
    assert result["status"] == "completed"
    assert len(result["segments"]) == 2
    assert result["full_text"] == "\u6d4b\u8bd5\u6587\u672c\u6d4b\u8bd5\u6587\u672c"

    await tracker.close()


@pytest.mark.asyncio
async def test_process_task_no_speech(tmp_path):
    upload_path = create_test_audio(str(tmp_path / "silent.wav"), duration=1.0)
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    tracker = TaskTracker(str(tmp_path / "test.db"))
    await tracker.init()
    task_id = "test-task-2"
    await tracker.create(task_id, "silent.wav", upload_path)

    vad_segmenter = MagicMock()
    vad_segmenter.detect.return_value = []

    asr_client = AsyncMock()

    with patch("app.pipeline.load_audio", return_value=np.zeros(16000, dtype=np.float32)):
        with patch("app.pipeline.save_wav"):
            await process_task(
                task_id=task_id,
                file_path=upload_path,
                tracker=tracker,
                vad_segmenter=vad_segmenter,
                asr_client=asr_client,
                chunk_dir=str(tmp_path / "chunks"),
                result_dir=str(result_dir),
            )

    task = await tracker.get(task_id)
    assert task["status"] == "completed"

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)
    assert result["status"] == "no_speech"

    await tracker.close()
