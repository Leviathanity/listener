import uuid
import pytest
import pytest_asyncio
import asyncio
import os
from pathlib import Path
from app.task_tracker import TaskTracker


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_tasks.db")


@pytest_asyncio.fixture
async def tracker(db_path):
    t = TaskTracker(db_path)
    await t.init()
    yield t
    await t.close()


@pytest.mark.asyncio
async def test_create_and_get_task(tracker):
    task_id = str(uuid.uuid4())
    await tracker.create(task_id, "test.mp4", "/tmp/test.mp4")

    task = await tracker.get(task_id)
    assert task is not None
    assert task["id"] == task_id
    assert task["filename"] == "test.mp4"
    assert task["status"] == "pending"
    assert task["progress"] == 0.0


@pytest.mark.asyncio
async def test_get_nonexistent_task(tracker):
    task = await tracker.get("nonexistent")
    assert task is None


@pytest.mark.asyncio
async def test_update_status(tracker):
    task_id = str(uuid.uuid4())
    await tracker.create(task_id, "test.mp4", "/tmp/test.mp4")

    await tracker.update(task_id, status="processing", progress_detail="Segmenting audio...")
    task = await tracker.get(task_id)
    assert task["status"] == "processing"
    assert task["progress_detail"] == "Segmenting audio..."


@pytest.mark.asyncio
async def test_update_progress(tracker):
    task_id = str(uuid.uuid4())
    await tracker.create(task_id, "test.mp4", "/tmp/test.mp4")

    await tracker.update(task_id, progress=0.5, progress_detail="Transcribing segment 3/6")
    task = await tracker.get(task_id)
    assert task["progress"] == 0.5
    assert task["progress_detail"] == "Transcribing segment 3/6"


@pytest.mark.asyncio
async def test_mark_completed(tracker):
    task_id = str(uuid.uuid4())
    await tracker.create(task_id, "test.mp4", "/tmp/test.mp4")
    await tracker.update(task_id, status="completed", result_path="/tmp/result.json", progress=1.0)

    task = await tracker.get(task_id)
    assert task["status"] == "completed"
    assert task["result_path"] == "/tmp/result.json"
    assert task["progress"] == 1.0


@pytest.mark.asyncio
async def test_mark_failed(tracker):
    task_id = str(uuid.uuid4())
    await tracker.create(task_id, "test.mp4", "/tmp/test.mp4")
    await tracker.update(task_id, status="failed", error_message="VAD detected no speech")

    task = await tracker.get(task_id)
    assert task["status"] == "failed"
    assert task["error_message"] == "VAD detected no speech"


@pytest.mark.asyncio
async def test_count_processing_tasks(tracker):
    assert await tracker.count_processing() == 0

    await tracker.create("a", "a.mp4", "/tmp/a.mp4")
    await tracker.create("b", "b.mp4", "/tmp/b.mp4")
    await tracker.update("a", status="processing")
    await tracker.update("b", status="processing")

    assert await tracker.count_processing() == 2
