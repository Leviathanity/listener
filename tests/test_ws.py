"""Test WebSocket streaming with session reuse."""
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from app.ws import Session, SessionManager


def test_session_create_and_close(tmp_path):
    """Session lifecycle."""
    sess = Session("test-1", str(tmp_path), 16000)
    assert sess.id == "test-1"
    assert sess.active
    assert sess.buffer_seconds() == 0.0
    sess.feed(b"\x00\x00" * 16000)
    assert abs(sess.buffer_seconds() - 1.0) < 0.01
    sess.close()
    assert not sess.active


def test_session_rejects_short_buffer(tmp_path):
    """Sessions with <0.5s buffer return None."""
    sess = Session("test-2", str(tmp_path), 16000)
    sess.feed(b"\x00\x00" * 4000)
    vad = MagicMock()
    vad.detect.return_value = [(0.0, 0.25)]
    import asyncio
    result = asyncio.run(sess.transcribe(vad, AsyncMock()))
    assert result is None


@pytest.mark.asyncio
async def test_session_manager_lifecycle(tmp_path):
    """Manager create/end/cancel."""
    vad = MagicMock()
    asr = AsyncMock()
    asr.transcribe.return_value = "hello"
    mgr = SessionManager(vad, asr, str(tmp_path))

    mgr.create_session("s1")
    assert "s1" in mgr.sessions

    mgr.create_session("s2")
    assert "s2" in mgr.sessions

    assert len(mgr.sessions) == 2

    text = await mgr.end_session("s1")
    assert "s1" not in mgr.sessions
    assert "s2" in mgr.sessions

    mgr.cancel_session("s2")
    assert "s2" not in mgr.sessions

    mgr.close_all()


def test_session_buffer_trimming(tmp_path):
    """Buffer trims after transcribe."""
    sess = Session("test-3", str(tmp_path), 16000)
    sess.feed(b"\x00\x00" * 48000)
    assert abs(sess.buffer_seconds() - 3.0) < 0.01

    vad = MagicMock()
    vad.detect.return_value = [(0.0, 1.0)]
    asr = AsyncMock()
    asr.transcribe.return_value = "test"

    import asyncio
    asyncio.run(sess.transcribe(vad, asr))

    remaining = sess.buffer_seconds()
    assert 1.5 < remaining < 2.5, f"Expected ~2s, got {remaining}"


def test_ws_test_mode_returns_error(tmp_path):
    from starlette.testclient import TestClient
    from app.main import app
    import app.main as m
    m._test_mode = True
    m._TEST_DB_PATH = str(tmp_path / "test.db")
    m._TEST_DATA_DIR = tmp_path

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/transcribe") as ws:
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "test mode" in data["msg"].lower()
