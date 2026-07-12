"""WebSocket streaming transcription with session reuse."""
import json
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path


class Session:
    def __init__(self, session_id: str, session_dir: str, sample_rate: int):
        self.id = session_id
        self.dir = Path(session_dir) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.active = True

    def feed(self, data: bytes):
        self.buffer.extend(data)

    def buffer_seconds(self) -> float:
        return len(self.buffer) / (self.sample_rate * 2)

    async def transcribe(self, vad, asr_client) -> str | None:
        if len(self.buffer) < self.sample_rate * 0.5 * 2:
            return None

        wav_path = str(self.dir / "buf.wav")
        audio = np.frombuffer(bytes(self.buffer), dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(wav_path, audio, self.sample_rate, subtype="PCM_16")

        timestamps = vad.detect(wav_path)
        if not timestamps:
            return None

        texts = []
        for ts in timestamps:
            start = int(ts[0] * self.sample_rate)
            end = int(ts[1] * self.sample_rate)
            seg = audio[start:end]
            if len(seg) >= self.sample_rate * 0.3:
                seg_path = str(self.dir / f"seg_{ts[0]:.0f}.wav")
                sf.write(seg_path, seg, self.sample_rate, subtype="PCM_16")
                t = await asr_client.transcribe(seg_path)
                if t:
                    texts.append(t)

        self.buffer.clear()
        return " ".join(texts) if texts else None

    def close(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)
        self.active = False


class SessionManager:
    def __init__(self, vad, asr_client, chunk_dir: str):
        self.vad = vad
        self.asr = asr_client
        self.chunk_dir = chunk_dir
        self.sessions: dict[str, Session] = {}

    def create_session(self, session_id: str, sample_rate: int = 16000) -> Session:
        session = Session(session_id, self.chunk_dir, sample_rate)
        self.sessions[session_id] = session
        return session

    async def end_session(self, session_id: str) -> str | None:
        session = self.sessions.get(session_id)
        if not session:
            return None
        session.active = False

        text = None
        if session.buffer_seconds() >= 0.5:
            text = await session.transcribe(self.vad, self.asr)

        session.close()
        self.sessions.pop(session_id, None)
        return text

    def cancel_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if session:
            session.active = False
            session.close()
            self.sessions.pop(session_id, None)

    def close_all(self):
        for sid in list(self.sessions.keys()):
            self.cancel_session(sid)

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)
