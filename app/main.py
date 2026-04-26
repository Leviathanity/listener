import os
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import (
    DB_PATH, UPLOAD_DIR, CHUNK_DIR, RESULT_DIR,
    SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, MAX_CONCURRENT_TASKS,
    VAD_MODEL_DIR, ASR_BASE_URL, ASR_MODEL, ASR_MAX_CONCURRENT,
)
from app.task_tracker import TaskTracker
from app.vad import VadSegmenter
from app.asr import AsrClient
from app.pipeline import process_task


_test_mode = False
_TEST_DB_PATH = None
_TEST_DATA_DIR = None

_tracker: TaskTracker | None = None
_vad_segmenter: VadSegmenter | None = None
_asr_client: AsrClient | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _tracker, _vad_segmenter, _asr_client

    db_path = _TEST_DB_PATH if _test_mode else DB_PATH
    _tracker = TaskTracker(db_path)
    await _tracker.init()

    if not _test_mode:
        _vad_segmenter = VadSegmenter(model_dir=VAD_MODEL_DIR)
        _asr_client = AsrClient(
            base_url=ASR_BASE_URL,
            model=ASR_MODEL,
            max_concurrent=ASR_MAX_CONCURRENT,
        )

    yield

    await _tracker.close()
    if _asr_client:
        await _asr_client.close()


app = FastAPI(title="Listener ASR Service", version="1.0.0", lifespan=lifespan)


async def _ensure_tracker():
    global _tracker
    if _tracker is None:
        db_path = _TEST_DB_PATH if _test_mode else DB_PATH
        _tracker = TaskTracker(db_path)
        await _tracker.init()


def get_tracker() -> TaskTracker:
    return _tracker


def _data_dir(subdir: str) -> Path:
    if _test_mode and _TEST_DATA_DIR:
        return Path(_TEST_DATA_DIR) / subdir
    upload_dir = UPLOAD_DIR if subdir == "uploads" else Path(CHUNK_DIR if subdir == "chunks" else RESULT_DIR)
    return upload_dir


@app.post("/v1/tasks")
async def create_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    await _ensure_tracker()
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file format: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    processing_count = await _tracker.count_processing()
    if processing_count >= MAX_CONCURRENT_TASKS:
        raise HTTPException(503, f"Server busy: {processing_count} tasks already processing. Please try again later.")

    task_id = str(uuid.uuid4())
    upload_dir = _data_dir("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(upload_dir / f"{task_id}{ext}")

    total_size = 0
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE_BYTES:
                os.remove(save_path)
                raise HTTPException(413, "File too large (max 2GB)")
            f.write(chunk)

    await _tracker.create(task_id, file.filename, save_path)

    if not _test_mode:
        background_tasks.add_task(
            process_task,
            task_id=task_id,
            file_path=save_path,
            tracker=_tracker,
            vad_segmenter=_vad_segmenter,
            asr_client=_asr_client,
            chunk_dir=str(_data_dir("chunks")),
            result_dir=str(_data_dir("results")),
        )

    return {"task_id": task_id, "status": "pending"}


@app.get("/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    await _ensure_tracker()
    task = await _tracker.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return {
        "task_id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
        "progress_detail": task.get("progress_detail"),
    }


@app.get("/v1/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    task = await _tracker.get(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if task["status"] == "pending" or task["status"] == "processing":
        raise HTTPException(409, "Task not yet completed")
    if task["status"] == "failed":
        return JSONResponse(
            status_code=422,
            content={"task_id": task_id, "status": "failed", "error": task.get("error_message")},
        )

    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
