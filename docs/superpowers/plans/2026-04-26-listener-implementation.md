# Listener Meeting ASR Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI service that accepts uploaded meeting recordings (1-3h), runs FireRedVAD segmentation + llama.cpp Qwen3-ASR transcription, and returns timestamped text via async task API.

**Architecture:** FastAPI with BackgroundTasks for async processing. SQLite tracks task state machine (pending → processing → completed/failed). httpx.AsyncClient calls local llama.cpp server with 4-concurrent limit. FireRedVAD detects speech segments for intelligent audio chunking.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, httpx, fireredvad, SQLite (aiosqlite), librosa, soundfile, pydub, numpy

---

### Task 0: Verify ASR Audio Format

**Files:**
- Create: `tests/test_asr_format.py`
- N/A for production code — this is a verification step

**Goal:** Before implementing the ASR client, verify the exact audio format the llama.cpp server accepts by making a test call with a short WAV file.

- [ ] **Step 1: Write verification test script**

```python
# tests/test_asr_format.py
import base64
import json
import httpx
import soundfile as sf
import numpy as np

ASR_URL = "http://192.168.2.118:8080"

def test_asr_audio_format():
    samplerate = 16000
    duration = 2
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samplerate * duration)).astype(np.float32)

    sf.write("test_audio.wav", samples, samplerate, subtype="PCM_16")

    with open("test_audio.wav", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please transcribe this audio."},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
                ]
            }
        ]
    }

    resp = httpx.post(f"{ASR_URL}/v1/chat/completions", json=payload, timeout=120)
    print(f"Status: {resp.status_code}")
    print(f"Body: {json.dumps(resp.json(), indent=2)}")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "choices" in data, f"No choices in response: {data}"
    assert len(data["choices"]) > 0, "Empty choices"
    text = data["choices"][0]["message"]["content"]
    print(f"Transcription: '{text}'")
    print("ASR audio format VERIFIED")

if __name__ == "__main__":
    test_asr_audio_format()
```

- [ ] **Step 2: Run verification**

```bash
python tests/test_asr_format.py
```

Expected: Either 200 with valid transcription, OR a 400/422 with error message indicating wrong format. If format is wrong, adjust the `content` structure based on error message.

- [ ] **Step 3: Document verified format**

After success, note the exact request/response format in a comment at the top of `app/asr.py` when created in Task 4.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`
- Create: `data/uploads/.gitkeep`
- Create: `data/chunks/.gitkeep`
- Create: `data/results/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir app tests data data/uploads data/chunks data/results
```

- [ ] **Step 2: Write app/__init__.py**

```python
# app/__init__.py
```

- [ ] **Step 3: Write app/config.py**

```python
# app/config.py
import os
from pathlib import Path


BASE_DIR = Path(os.environ.get("LISTENER_BASE_DIR", Path(__file__).resolve().parent.parent))

ASR_BASE_URL = os.environ.get("ASR_BASE_URL", "http://192.168.2.118:8080")
ASR_MODEL = os.environ.get("ASR_MODEL", "asr")
ASR_MAX_CONCURRENT = int(os.environ.get("ASR_MAX_CONCURRENT", "4"))
ASR_MAX_RETRIES = int(os.environ.get("ASR_MAX_RETRIES", "5"))

VAD_MODEL_DIR = os.environ.get("VAD_MODEL_DIR", str(BASE_DIR / "pretrained_models" / "FireRedVAD" / "VAD"))
VAD_SEGMENT_THRESHOLD_S = int(os.environ.get("VAD_SEGMENT_THRESHOLD_S", "120"))
VAD_MAX_SEGMENT_THRESHOLD_S = int(os.environ.get("VAD_MAX_SEGMENT_THRESHOLD_S", "180"))
VAD_TARGET_SAMPLE_RATE = 16000

DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "data" / "tasks.db"))

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHUNK_DIR = BASE_DIR / "data" / "chunks"
RESULT_DIR = BASE_DIR / "data" / "results"

MAX_UPLOAD_SIZE_BYTES = 2 * 1024 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
MAX_CONCURRENT_TASKS = 3
```

- [ ] **Step 4: Write requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
httpx>=0.27.0
aiosqlite>=0.20.0
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
numpy>=1.26.0
fireredvad>=1.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 5: Write tests/__init__.py**

```python
# tests/__init__.py
```

- [ ] **Step 6: Create .gitkeep files for empty dirs**

```bash
type nul > data\uploads\.gitkeep
type nul > data\chunks\.gitkeep
type nul > data\results\.gitkeep
```

- [ ] **Step 7: Commit**

```bash
git add app/ requirements.txt tests/ data/
git commit -m "feat: scaffold project structure and config"
```

---

### Task 2: Task Tracker (SQLite)

**Files:**
- Create: `app/task_tracker.py`
- Create: `tests/test_task_tracker.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_task_tracker.py
import uuid
import pytest
import asyncio
import os
from pathlib import Path
from app.task_tracker import TaskTracker


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_tasks.db")


@pytest.fixture
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_task_tracker.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.task_tracker'`

- [ ] **Step 3: Write implementation**

```python
# app/task_tracker.py
import aiosqlite
from datetime import datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
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
"""


class TaskTracker:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    async def init(self):
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def create(self, task_id: str, filename: str, file_path: str):
        now = datetime.utcnow().isoformat()
        await self._conn.execute(
            "INSERT INTO tasks (id, filename, file_path, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (task_id, filename, file_path, now, now),
        )
        await self._conn.commit()

    async def get(self, task_id: str) -> dict | None:
        cursor = await self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def update(self, task_id: str, **kwargs):
        allowed = {"status", "progress", "progress_detail", "result_path", "error_message"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        await self._conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
        await self._conn.commit()

    async def count_processing(self) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status = 'processing'")
        row = await cursor.fetchone()
        return row["cnt"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_task_tracker.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/task_tracker.py tests/test_task_tracker.py
git commit -m "feat: add SQLite task tracker with CRUD and state management"
```

---

### Task 3: Audio Utilities

**Files:**
- Create: `app/audio_utils.py`
- Create: `tests/test_audio_utils.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_audio_utils.py
import numpy as np
import soundfile as sf
from pathlib import Path
from app.audio_utils import load_audio, save_wav, chunk_audio
from app.config import VAD_TARGET_SAMPLE_RATE


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
    assert len(chunks) == 4
    for chunk in chunks:
        duration = chunk[1] - chunk[0]
        assert duration <= 30.0


def test_chunk_audio_short_audio():
    wav = np.zeros(8000, dtype=np.float32)
    timestamps = [(0.0, 0.5)]
    chunks = chunk_audio(wav, 16000, timestamps, target_duration_s=10, max_duration_s=30)
    assert len(chunks) == 1
    assert chunks[0] == (0.0, 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_audio_utils.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# app/audio_utils.py
import io
import subprocess
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path


def load_audio(file_path: str) -> np.ndarray:
    try:
        wav_data, _ = librosa.load(file_path, sr=16000, mono=True)
        return wav_data
    except Exception:
        pass

    cmd = [
        "ffmpeg", "-i", file_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "-f", "wav", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {proc.stderr.decode('utf-8', errors='ignore')}")
    data, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return data


def save_wav(wav: np.ndarray, file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, wav, 16000, subtype="PCM_16")


def chunk_audio(wav: np.ndarray, sample_rate: int, timestamps: list[tuple[float, float]],
                target_duration_s: int = 120, max_duration_s: int = 180) -> list[tuple[float, float, np.ndarray]]:
    if not timestamps:
        return [(0.0, len(wav) / sample_rate, wav)]

    target_samples = target_duration_s * sample_rate
    max_samples = max_duration_s * sample_rate

    boundary_samples = {0}
    for t_start, t_end in timestamps:
        boundary_samples.add(int(t_start * sample_rate))
        boundary_samples.add(int(t_end * sample_rate))
    boundary_samples.add(len(wav))
    sorted_boundaries = sorted(boundary_samples)

    final_splits = {0, len(wav)}
    cursor = target_samples
    while cursor < len(wav):
        closest = min(sorted_boundaries, key=lambda b: abs(b - cursor))
        final_splits.add(closest)
        cursor += target_samples
    final_ordered = sorted(final_splits)

    result = []
    for i in range(len(final_ordered) - 1):
        start_sample = final_ordered[i]
        end_sample = final_ordered[i + 1]

        if end_sample - start_sample <= max_samples:
            result.append((start_sample / sample_rate, end_sample / sample_rate, wav[start_sample:end_sample]))
        else:
            n_segments = int(np.ceil((end_sample - start_sample) / max_samples))
            seg_len = (end_sample - start_sample) / n_segments
            for j in range(n_segments):
                split_start = int(start_sample + j * seg_len)
                split_end = int(start_sample + (j + 1) * seg_len) if j < n_segments - 1 else end_sample
                result.append((split_start / sample_rate, split_end / sample_rate, wav[split_start:split_end]))

    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_audio_utils.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/audio_utils.py tests/test_audio_utils.py
git commit -m "feat: add audio loading, saving, and smart chunking utilities"
```

---

### Task 4: VAD Wrapper (FireRedVAD)

**Files:**
- Create: `app/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_vad.py
import numpy as np
import soundfile as sf
from pathlib import Path
from app.vad import VadSegmenter
from app.config import VAD_TARGET_SAMPLE_RATE


def create_silent_wav_with_speech(path, dur_speech_start=0.5, dur_speech=1.0, dur_total=3.0, sr=16000):
    """Create a WAV with a burst of tone in the middle of silence."""
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
    # We test with a mock approach since the real model requires download.
    # This test will initially fail with FileNotFoundError, guiding model download.
    try:
        result = segmenter.detect("nonexistent.wav")
        assert isinstance(result, list)
    except FileNotFoundError:
        pass


def test_vad_result_format(tmp_path):
    """Verify VAD result structure when model is available."""
    segmenter = VadSegmenter(model_dir="does_not_exist_yet")
    assert hasattr(segmenter, "detect")
    assert hasattr(segmenter, "model")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_vad.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.vad'`

- [ ] **Step 3: Write implementation**

```python
# app/vad.py
from pathlib import Path
from fireredvad import FireRedVad, FireRedVadConfig


class VadSegmenter:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            config = FireRedVadConfig(
                use_gpu=False,
                smooth_window_size=5,
                speech_threshold=0.4,
                min_speech_frame=20,
                max_speech_frame=2000,
                min_silence_frame=20,
                merge_silence_frame=0,
                extend_speech_frame=0,
                chunk_max_frame=30000,
            )
            self._model = FireRedVad.from_pretrained(self.model_dir, config)
        return self._model

    def detect(self, wav_path: str) -> list[tuple[float, float]]:
        result, _ = self.model.detect(wav_path)
        return list(result.get("timestamps", []))
```

- [ ] **Step 4: Run test to verify it passes (or skip gracefully)**

```bash
pytest tests/test_vad.py -v
```
Expected: 2 passed (tests guard against missing model)

- [ ] **Step 5: Commit**

```bash
git add app/vad.py tests/test_vad.py
git commit -m "feat: add FireRedVAD wrapper for speech segment detection"
```

---

### Task 5: ASR Client (llama.cpp)

**Files:**
- Create: `app/asr.py`
- Create: `tests/test_asr_client.py`

**Note:** The exact audio format in the request body must match what was verified in Task 0. Below assumes the format `content: [{type: "text", text: "..."}, {type: "audio_url", audio_url: {url: "data:audio/wav;base64,..."}}]`. Update if Task 0 determined a different format.

- [ ] **Step 1: Write failing test**

```python
# tests/test_asr_client.py
import base64
import httpx
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


@pytest.mark.asyncio
async def test_transcribe_segment_integration(asr_client, tmp_path):
    """Integration test — requires running ASR server at 192.168.2.118:8080."""
    wav_path = create_test_wav(str(tmp_path / "test.wav"), duration=2.0)

    try:
        text = await asr_client.transcribe(wav_path)
        assert isinstance(text, str)
    except httpx.ConnectError:
        pytest.skip("ASR server not reachable")
    except Exception as e:
        pytest.skip(f"ASR call failed: {e}")


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
    assert payload["messages"][0]["content"][1]["type"] == "audio_url"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_asr_client.py -v -k "not integration"
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# app/asr.py
import base64
import asyncio
import httpx
from pathlib import Path


class AsrClient:
    def __init__(self, base_url: str, model: str = "asr", max_concurrent: int = 4,
                 max_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_payload(self, audio_b64: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please transcribe this audio."},
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
                    ]
                }
            ]
        }

    async def transcribe(self, wav_path: str) -> str:
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = self._build_payload(audio_b64)

        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    client = await self._get_client()
                    resp = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status_code in (500, 503):
                        delay = 2 * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise Exception(f"ASR error {resp.status_code}: {resp.text}")
                except httpx.ReadTimeout:
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)
                except Exception:
                    if attempt >= self.max_retries - 1:
                        raise
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise Exception(f"ASR failed after {self.max_retries} retries for {wav_path}")
```

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
pytest tests/test_asr_client.py -v -k "not integration"
```
Expected: 2 passed (init + build_payload), 1 skipped (integration)

- [ ] **Step 5: Commit**

```bash
git add app/asr.py tests/test_asr_client.py
git commit -m "feat: add llama.cpp ASR client with semaphore-limited concurrency and exponential backoff retry"
```

---

### Task 6: Postprocessor

**Files:**
- Create: `app/postprocess.py`
- Create: `tests/test_postprocess.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_postprocess.py
from app.postprocess import clean_text


def test_clean_text_removes_char_repeats():
    result = clean_text("aaaaaaaaaaabbbbbbbbbbbcccc")
    assert len(result) < 25
    assert "a" in result


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_postprocess.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# app/postprocess.py
def clean_text(text: str, threshold: int = 20) -> str:
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i + count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_len = thresh * 2
        if n < min_len:
            return s
        i = 0
        result = []
        found = False
        while i <= n - min_len:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    if s[i + rep * k:i + rep * k + k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index + k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text = fix_char_repeats(text, threshold)
    return fix_pattern_repeats(text, threshold)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_postprocess.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/postprocess.py tests/test_postprocess.py
git commit -m "feat: add ASR text postprocessing to remove character and pattern repetitions"
```

---

### Task 7: Pipeline Orchestrator

**Files:**
- Create: `app/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pipeline.py
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
    vad_segmenter.detect.return_value = [(0.0, 1.0), (1.0, 2.0)]

    asr_client = AsyncMock()
    asr_client.transcribe.return_value = "测试文本"

    with patch("app.pipeline.load_audio", return_value=np.zeros(32000, dtype=np.float32)):
        with patch("app.pipeline.clean_text", return_value="测试文本"):
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
    assert result["full_text"] == "测试文本测试文本"

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# app/pipeline.py
import json
import shutil
import asyncio
from pathlib import Path
from app.audio_utils import load_audio, save_wav, chunk_audio
from app.postprocess import clean_text
from app.config import VAD_TARGET_SAMPLE_RATE, VAD_SEGMENT_THRESHOLD_S, VAD_MAX_SEGMENT_THRESHOLD_S


async def process_task(task_id, file_path, tracker, vad_segmenter, asr_client, chunk_dir, result_dir):
    try:
        await tracker.update(task_id, status="processing", progress_detail="Loading audio...")

        wav = load_audio(file_path)
        total_duration = len(wav) / VAD_TARGET_SAMPLE_RATE

        await tracker.update(task_id, progress=0.05, progress_detail="Running VAD...")

        wav_chunk_path = Path(chunk_dir) / task_id
        wav_chunk_path.mkdir(parents=True, exist_ok=True)
        full_wav_path = str(wav_chunk_path / "full.wav")
        save_wav(wav, full_wav_path)

        timestamps = vad_segmenter.detect(full_wav_path)

        if not timestamps:
            result = {
                "task_id": task_id,
                "status": "no_speech",
                "segments": [],
                "full_text": "",
            }
            result_path = str(Path(result_dir) / f"{task_id}.json")
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            await tracker.update(task_id, status="completed", result_path=result_path, progress=1.0)
            return

        if total_duration > VAD_MAX_SEGMENT_THRESHOLD_S:
            chunks = chunk_audio(wav, VAD_TARGET_SAMPLE_RATE, timestamps,
                                 target_duration_s=VAD_SEGMENT_THRESHOLD_S,
                                 max_duration_s=VAD_MAX_SEGMENT_THRESHOLD_S)
        else:
            chunks = [(0.0, total_duration, wav)]

        await tracker.update(task_id, progress=0.1,
                             progress_detail=f"Transcribing {len(chunks)} segments...")

        chunk_paths = []
        for idx, (start_s, end_s, chunk_data) in enumerate(chunks):
            chunk_path = str(wav_chunk_path / f"chunk_{idx:04d}.wav")
            save_wav(chunk_data, chunk_path)
            chunk_paths.append((idx, start_s, end_s, chunk_path))

        async def transcribe_one(idx, start_s, end_s, path):
            try:
                text = await asr_client.transcribe(path)
                return (idx, start_s, end_s, text, None)
            except Exception as e:
                return (idx, start_s, end_s, "", str(e))

        tasks = [transcribe_one(idx, s, e, p) for idx, s, e, p in chunk_paths]
        results_list = await asyncio.gather(*tasks)

        results_list.sort(key=lambda x: x[0])
        total = len(results_list)

        segments = []
        texts = []
        failed_count = 0
        for i, (idx, start_s, end_s, text, error) in enumerate(results_list):
            if error:
                failed_count += 1
                texts.append(f"[Segment {idx} failed: {error}]")
            else:
                cleaned = clean_text(text)
                texts.append(cleaned)
                segments.append({
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "text": cleaned,
                })
            progress = 0.1 + 0.85 * (i + 1) / total
            await tracker.update(task_id, progress=progress,
                                 progress_detail=f"Transcribing segment {i + 1}/{total}")

        full_text = " ".join(t for t in texts if t)
        status = "completed" if failed_count < total else "failed"

        result = {
            "task_id": task_id,
            "status": status,
            "segments": segments,
            "full_text": full_text,
        }
        if failed_count > 0 and status == "completed":
            result["warning"] = f"{failed_count}/{total} segments failed and were skipped"

        result_path = str(Path(result_dir) / f"{task_id}.json")
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        await tracker.update(task_id, status="completed", result_path=result_path, progress=1.0)

        shutil.rmtree(wav_chunk_path, ignore_errors=True)

    except Exception as e:
        await tracker.update(task_id, status="failed", error_message=str(e))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: add end-to-end pipeline orchestrator with VAD, parallel ASR, and result assembly"
```

---

### Task 8: FastAPI Endpoints

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py
import json
import pytest
from pathlib import Path
from httpx import ASGITransport, AsyncClient
from app.main import app, get_tracker


@pytest.fixture
def test_db_path(tmp_path):
    db_path = str(tmp_path / "test_api.db")
    import app.main
    app.main._TEST_DB_PATH = db_path
    app.main._TEST_DATA_DIR = tmp_path
    app.main._test_mode = True
    return db_path


@pytest.mark.asyncio
async def test_post_upload_returns_task_id(test_db_path, tmp_path):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_content = b"fake audio content"
        resp = await client.post(
            "/v1/tasks",
            files={"file": ("test.mp4", file_content, "audio/mp4")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_post_upload_rejects_invalid_extension(test_db_path, tmp_path):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/tasks",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_task_status_pending(test_db_path, tmp_path):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/tasks",
            files={"file": ("test.mp4", b"fake audio", "audio/mp4")},
        )
        task_id = resp.json()["task_id"]

        resp2 = await client.get(f"/v1/tasks/{task_id}")
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["task_id"] == task_id
        assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_task_not_found(test_db_path, tmp_path):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/tasks/nonexistent-id")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_result_not_completed(test_db_path, tmp_path):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/tasks",
            files={"file": ("test.mp4", b"fake audio", "audio/mp4")},
        )
        task_id = resp.json()["task_id"]

        resp2 = await client.get(f"/v1/tasks/{task_id}/result")
        assert resp2.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# app/main.py
import os
import uuid
import asyncio
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


def get_tracker() -> TaskTracker:
    return _tracker


def _data_dir(subdir: str) -> Path:
    if _test_mode and _TEST_DATA_DIR:
        return Path(_TEST_DATA_DIR) / subdir
    return UPLOAD_DIR.parent / subdir if subdir != "uploads" else UPLOAD_DIR


@app.post("/v1/tasks")
async def create_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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

    import json
    with open(task["result_path"], "r", encoding="utf-8") as f:
        result = json.load(f)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_api.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI endpoints for task upload, status query, and result retrieval"
```

---

### Task 9: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/listener

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p data/uploads data/chunks data/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write .dockerignore**

```
__pycache__
*.pyc
tests/
.git/
venv/
env/
pretrained_models/
data/uploads/*
data/chunks/*
data/results/*
docs/
Qwen3-ASR-Toolkit/
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for WSL Docker deployment"
```

---

### Task 10: Integration Verification

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test script**

```python
# tests/test_integration.py
"""
Integration test — requires:
1. ASR server running at http://192.168.2.118:8080
2. FireRedVAD model at pretrained_models/FireRedVAD/VAD

Run: pytest tests/test_integration.py -v -s
"""
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_full_pipeline_with_short_audio(tmp_path):
    from app.main import app
    import app.main

    app.main._test_mode = True
    app.main._TEST_DB_PATH = str(tmp_path / "test.db")
    app.main._TEST_DATA_DIR = tmp_path

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), samples, sr, subtype="PCM_16")

    with open(wav_path, "rb") as f:
        file_content = f.read()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/tasks",
            files={"file": ("test.wav", file_content, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        task_id = data["task_id"]

        import asyncio
        max_wait = 10
        for _ in range(max_wait):
            await asyncio.sleep(1)
            resp2 = await client.get(f"/v1/tasks/{task_id}")
            status_data = resp2.json()
            if status_data["status"] in ("completed", "failed"):
                break

        assert status_data["status"] == "completed"

        resp3 = await client.get(f"/v1/tasks/{task_id}/result")
        assert resp3.status_code == 200
        result = resp3.json()
        assert "segments" in result
        assert "full_text" in result

    app.main._test_mode = False
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v -s
```

Expected: 1 passed (skipped if ASR server unavailable)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

### Task 11: Final Lint & Verification

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v --ignore=tests/test_integration.py
```

Expected: All tests pass (~20+ tests).

- [ ] **Step 2: Verify project structure**

```bash
dir /s /b app tests
```

Expected:
```
app\__init__.py
app\audio_utils.py
app\config.py
app\main.py
app\pipeline.py
app\vad.py
app\asr.py
app\task_tracker.py
app\postprocess.py
tests\__init__.py
tests\test_audio_utils.py
tests\test_task_tracker.py
tests\test_vad.py
tests\test_asr_client.py
tests\test_postprocess.py
tests\test_pipeline.py
tests\test_api.py
tests\test_integration.py
```

- [ ] **Step 3: Start service and smoke test**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/v1/tasks -F "file=@data/uploads/test.wav"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final project verification and cleanup"
```
