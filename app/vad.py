from pathlib import Path


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
        if self.model is None:
            raise FileNotFoundError("VAD model not available")
        result, _ = self.model.detect(wav_path)
        return list(result.get("timestamps", []))
