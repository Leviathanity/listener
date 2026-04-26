from app.config import (
    VAD_CHUNK_MAX_FRAME,
    VAD_EXTEND_SPEECH_FRAME,
    VAD_MAX_SPEECH_FRAME,
    VAD_MERGE_SILENCE_FRAME,
    VAD_MIN_SILENCE_FRAME,
    VAD_MIN_SPEECH_FRAME,
    VAD_SMOOTH_WINDOW_SIZE,
    VAD_SPEECH_THRESHOLD,
)


class VadSegmenter:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fireredvad import FireRedVad, FireRedVadConfig
            except ImportError:
                return None
            config = FireRedVadConfig(
                use_gpu=False,
                smooth_window_size=VAD_SMOOTH_WINDOW_SIZE,
                speech_threshold=VAD_SPEECH_THRESHOLD,
                min_speech_frame=VAD_MIN_SPEECH_FRAME,
                max_speech_frame=VAD_MAX_SPEECH_FRAME,
                min_silence_frame=VAD_MIN_SILENCE_FRAME,
                merge_silence_frame=VAD_MERGE_SILENCE_FRAME,
                extend_speech_frame=VAD_EXTEND_SPEECH_FRAME,
                chunk_max_frame=VAD_CHUNK_MAX_FRAME,
            )
            self._model = FireRedVad.from_pretrained(self.model_dir, config)
        return self._model

    def detect(self, wav_path: str) -> list[tuple[float, float]]:
        if self.model is None:
            raise RuntimeError("VAD model not available, fireredvad may not be installed")
        result, _ = self.model.detect(wav_path)
        return list(result.get("timestamps", []))
