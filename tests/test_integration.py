import numpy as np
import soundfile as sf
import pytest
import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_full_pipeline_with_short_audio(tmp_path):
    import app.main as main_module

    main_module._test_mode = True
    main_module._TEST_DB_PATH = str(tmp_path / "test.db")
    main_module._TEST_DATA_DIR = tmp_path

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
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
        assert data["status"] == "pending"

        resp2 = await client.get(f"/v1/tasks/{task_id}")
        status_data = resp2.json()
        assert status_data["status"] in ("pending", "completed", "failed")

        resp3 = await client.get(f"/v1/tasks/{task_id}/result")
        if status_data["status"] == "completed":
            assert resp3.status_code == 200
        elif status_data["status"] == "failed":
            assert resp3.status_code == 422
        else:
            assert resp3.status_code == 409

    main_module._test_mode = False
