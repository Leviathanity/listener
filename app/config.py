# app/config.py
import os
from pathlib import Path


BASE_DIR = Path(os.environ.get("LISTENER_BASE_DIR", Path(__file__).resolve().parent.parent))

ASR_BASE_URL = os.environ.get("ASR_BASE_URL", "http://192.168.2.118:8080")
ASR_MODEL = os.environ.get("ASR_MODEL", "asr")
ASR_MAX_CONCURRENT = int(os.environ.get("ASR_MAX_CONCURRENT", "3"))
ASR_MAX_RETRIES = int(os.environ.get("ASR_MAX_RETRIES", "5"))

VAD_MODEL_DIR = os.environ.get("VAD_MODEL_DIR", str(BASE_DIR / "pretrained_models" / "FireRedVAD" / "VAD"))
VAD_SEGMENT_THRESHOLD_S = int(os.environ.get("VAD_SEGMENT_THRESHOLD_S", "60"))
VAD_MAX_SEGMENT_THRESHOLD_S = int(os.environ.get("VAD_MAX_SEGMENT_THRESHOLD_S", "60"))
VAD_TARGET_SAMPLE_RATE = 16000

VAD_SMOOTH_WINDOW_SIZE = 5
VAD_SPEECH_THRESHOLD = 0.4
VAD_MIN_SPEECH_FRAME = 20
VAD_MAX_SPEECH_FRAME = 2000
VAD_MIN_SILENCE_FRAME = 20
VAD_MERGE_SILENCE_FRAME = 0
VAD_EXTEND_SPEECH_FRAME = 0
VAD_CHUNK_MAX_FRAME = 30000

DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "data" / "tasks.db"))

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHUNK_DIR = BASE_DIR / "data" / "chunks"
RESULT_DIR = BASE_DIR / "data" / "results"

MAX_UPLOAD_SIZE_BYTES = 2 * 1024 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac"}
MAX_CONCURRENT_TASKS = 3
