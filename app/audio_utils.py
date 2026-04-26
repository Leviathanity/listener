import io
import subprocess
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from app.config import VAD_TARGET_SAMPLE_RATE


def load_audio(file_path: str) -> np.ndarray:
    try:
        wav_data, _ = librosa.load(file_path, sr=VAD_TARGET_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        librosa_error = str(e)

    cmd = [
        "ffmpeg", "-i", file_path,
        "-ar", str(VAD_TARGET_SAMPLE_RATE), "-ac", "1",
        "-c:a", "pcm_s16le", "-f", "wav", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to load audio. librosa error: {librosa_error}. "
            f"ffmpeg error: {proc.stderr.decode('utf-8', errors='ignore')}"
        )
    data, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return data


def save_wav(wav: np.ndarray, file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, wav, VAD_TARGET_SAMPLE_RATE, subtype="PCM_16")


def chunk_audio(wav: np.ndarray, sample_rate: int, timestamps: list[tuple[float, float]],
                target_duration_s: int = 120, max_duration_s: int = 180) -> list[tuple[float, float, np.ndarray]]:
    if not timestamps:
        return [(0.0, len(wav) / sample_rate, wav)]

    target_samples = target_duration_s * sample_rate
    max_samples = max_duration_s * sample_rate

    potential_split_points = {0, len(wav)}
    for t_start, _t_end in timestamps:
        potential_split_points.add(int(t_start * sample_rate))
    sorted_potential_splits = sorted(potential_split_points)

    final_splits = {0, len(wav)}
    cursor = target_samples
    while cursor < len(wav):
        closest = min(sorted_potential_splits, key=lambda p: abs(p - cursor))
        final_splits.add(closest)
        cursor += target_samples
    final_ordered = sorted(final_splits)

    new_split_points = [0]
    for i in range(1, len(final_ordered)):
        start = final_ordered[i - 1]
        end = final_ordered[i]
        segment_length = end - start

        if segment_length <= max_samples:
            new_split_points.append(end)
        else:
            num_subsegments = int(np.ceil(segment_length / max_samples))
            subsegment_length = segment_length / num_subsegments
            for j in range(1, num_subsegments):
                split_point = start + j * subsegment_length
                new_split_points.append(split_point)
            new_split_points.append(end)

    result = []
    for i in range(len(new_split_points) - 1):
        start_sample = int(new_split_points[i])
        end_sample = int(new_split_points[i + 1])
        result.append((start_sample / sample_rate, end_sample / sample_rate, wav[start_sample:end_sample]))

    return result
