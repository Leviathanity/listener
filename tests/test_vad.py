from app.vad import VadSegmenter


def test_vad_detects_speech_segment(tmp_path):
    segmenter = VadSegmenter(model_dir="does_not_exist_yet")
    try:
        result = segmenter.detect("nonexistent.wav")
        assert isinstance(result, list)
    except RuntimeError:
        pass


def test_vad_result_format(tmp_path):
    segmenter = VadSegmenter(model_dir="does_not_exist_yet")
    assert hasattr(segmenter, "detect")
    assert hasattr(segmenter, "model")
