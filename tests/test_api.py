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
