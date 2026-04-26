# Listener: Meeting Recording ASR Service Design

Date: 2026-04-26

## Overview

An HTTP API service for uploading meeting recordings (1-3 hours) and receiving transcribed text with timestamps. Built on FastAPI with FireRedVAD for voice activity detection and a local llama.cpp Qwen3-ASR server for speech recognition.

## Architecture

Single-container FastAPI service with background task processing:

```
Client ──POST /v1/tasks──▶ FastAPI ──▶ Save file + return task_id
                                    ──▶ BackgroundTasks:
                                          Audio → VAD → Chunk → ASR(4 concurrent) → Clean → Result
Client ──GET /v1/tasks/{id}──▶  task status + progress
Client ──GET /v1/tasks/{id}/result──▶  transcript JSON
```

- **Web framework**: FastAPI + uvicorn
- **Task tracking**: SQLite (task status, progress, result paths)
- **ASR client**: httpx.AsyncClient, semaphore-limited to 4 concurrent
- **Deployment**: WSL Docker

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Web layer | `app/main.py` | FastAPI endpoints, task creation, status/result queries |
| Pipeline | `app/pipeline.py` | End-to-end orchestration: load → VAD → chunk → ASR → clean → save |
| VAD | `app/vad.py` | FireRedVAD wrapper: load model, detect speech segments |
| ASR client | `app/asr.py` | llama.cpp `/v1/chat/completions` caller with retry logic |
| Task tracker | `app/task_tracker.py` | SQLite CRUD for task state machine |
| Postprocessor | `app/postprocess.py` | Remove character/pattern repetitions from ASR output |

## Data Flow

```
upload file (.mp4/.wav/.m4a/...)
  │
  ▼
① audio extraction → ffmpeg → 16kHz/16bit/mono WAV (librosa fallback)
  │
  ▼
② VAD segmentation → FireRedVAD.detect() → [(t_start, t_end), ...]
  │
  ▼
③ smart chunking → split at VAD boundaries, target 60-120s, max 180s
  │
  ▼
④ parallel ASR → httpx.AsyncClient (limit=4), base64 WAV → POST /v1/chat/completions
  │                retry: exponential backoff 2s/4s/8s/16s/32s (max 5)
  ▼
⑤ result assembly → sort by segment order, concatenate text
  │
  ▼
⑥ post-processing → remove char repeats, remove pattern repeats
  │
  ▼
⑦ output → JSON: { task_id, status, segments: [{start, end, text}], full_text }
```

## Task State Machine

```
pending → processing → completed
                     → failed
```

## SQLite Schema

```sql
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL DEFAULT 0.0,
    progress_detail TEXT,
    result_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Endpoints

### POST /v1/tasks
Upload audio file for transcription.

- Request: multipart/form-data, field `file`
- Response: `{ "task_id": "uuid", "status": "pending" }`
- Errors: 400 (unsupported format), 413 (file too large > 2GB)

### GET /v1/tasks/{task_id}
Query task status and progress.

- Response: `{ "task_id": "...", "status": "pending|processing|completed|failed", "progress": 0.75, "progress_detail": "Transcribing segment 9/12" }`
- Errors: 404 (task not found)

### GET /v1/tasks/{task_id}/result
Get transcription result for completed task.

- Response: `{ "task_id": "...", "segments": [{ "start": 0.0, "end": 5.2, "text": "..." }], "full_text": "..." }`
- Errors: 404 (task not found), 409 (task not yet completed)

## Directory Structure

```
/opt/listener/
  app/
    __init__.py
    main.py           # FastAPI endpoints
    pipeline.py       # processing orchestration
    vad.py            # FireRedVAD wrapper
    asr.py            # llama.cpp ASR client
    task_tracker.py   # SQLite task management
    postprocess.py    # text cleanup
  data/
    tasks.db          # SQLite database
    uploads/          # original uploaded files
    chunks/           # temporary VAD-chunked WAVs (deleted after processing)
    results/          # transcription result JSON
  pretrained_models/
    FireRedVAD/       # FireRedVAD model weights
  requirements.txt
  Dockerfile
```

## VAD Integration

| Aspect | Original (silero_vad) | New (FireRedVAD) |
|--------|----------------------|-------------------|
| Package | `pip install silero_vad` | `pip install fireredvad` |
| API | `get_speech_timestamps(wav, model)` | `vad.detect(wav_path)` |
| Output | `[{'start': 0.2, 'end': 1.5}]` | `{'timestamps': [(0.2, 1.5)]}` |
| Input | numpy array (16kHz) | file path to 16kHz WAV |
| Accuracy | F1 95.95% | F1 97.57% |

Key change: audio must be written to temporary WAV file before calling FireRedVAD (does not accept numpy array directly).

## ASR Service Details

- **Server**: llama.cpp at `http://192.168.2.118:8080`
- **Model**: `qwen3-asr` (Qwen3-ASR-1.7B Q8_0 GGUF with mmproj)
- **Endpoint**: `POST /v1/chat/completions`
- **Concurrency**: max 4 instances
- **Retry**: exponential backoff on 500/503 (2s/4s/8s/16s/32s, max 5 retries)
- **Audio format**: base64-encoded WAV in multimodal chat message content

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Unsupported file format | 400 error |
| File > 2GB | 413 error |
| No speech detected by VAD | Result with `no_speech` flag, skip ASR |
| ASR 500/503 | Exponential backoff retry (5 attempts) |
| ASR retries exhausted | Mark segment failed, continue with remaining segments |
| All segments fail | Task status = `failed` |
| Service restart | In-progress BackgroundTasks lost; uploaded files and completed results preserved; lost tasks return 404 |
| Max concurrent tasks | Limit 3 simultaneous processing tasks |

## Dependencies

```
fastapi, uvicorn, python-multipart, httpx, fireredvad, librosa, soundfile, pydub, numpy, aiofiles
```

## Open Items / Risks

1. **ASR audio format**: The llama.cpp multimodal chat completions audio format needs verification via a test call. Expected format: base64-encoded WAV in `content` array with `type: "audio_url"`. This should be confirmed before implementing the ASR client.

2. **FireRedVAD model path**: Model weights must be downloaded to `pretrained_models/FireRedVAD/` before use. Path should be configurable via environment variable.

## Reference

Based on [Qwen3-ASR-Toolkit](https://github.com/QwenLM/Qwen3-ASR-Toolkit) pipeline architecture, adapted for local ASR and FireRedVAD.
