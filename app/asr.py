import base64
import asyncio
import re
import httpx
from pathlib import Path


class AsrClient:
    def __init__(self, base_url: str = "http://192.168.2.118:8080", model: str = "asr",
                 max_concurrent: int = 4, max_retries: int = 5):
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
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                    ]
                }
            ]
        }

    @staticmethod
    def _read_file(wav_path: str) -> str:
        with open(wav_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_response(self, content: str) -> str:
        # Strip "language LANG<asr_text>" prefix
        content = re.sub(r'^language\s+\w*<asr_text>\s*', '', content)
        # Remove any closing tag
        content = content.replace('</asr_text>', '')
        content = re.sub(r'<\|[^|]+\|>', '', content)
        return content.strip()

    async def transcribe(self, wav_path: str) -> str:
        loop = asyncio.get_running_loop()
        audio_b64 = await loop.run_in_executor(None, self._read_file, wav_path)

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
                        content = data["choices"][0]["message"]["content"]
                        return self._parse_response(content)
                    elif resp.status_code in (500, 503):
                        delay = 2 * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        resp.raise_for_status()
                except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError):
                    if attempt >= self.max_retries - 1:
                        raise
                    delay = 2 * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise Exception(f"ASR failed after {self.max_retries} retries for {wav_path}")
