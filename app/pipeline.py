import json
import shutil
import asyncio
from pathlib import Path
from app.audio_utils import load_audio, save_wav, chunk_audio
from app.postprocess import clean_text
from app.config import VAD_TARGET_SAMPLE_RATE, VAD_SEGMENT_THRESHOLD_S, VAD_MAX_SEGMENT_THRESHOLD_S


async def process_task(task_id, file_path, tracker, vad_segmenter, asr_client, chunk_dir, result_dir):
    try:
        await tracker.update(task_id, status="processing", progress_detail="Loading audio...")

        wav = load_audio(file_path)
        total_duration = len(wav) / VAD_TARGET_SAMPLE_RATE

        await tracker.update(task_id, progress=0.05, progress_detail="Running VAD...")

        wav_chunk_path = Path(chunk_dir) / task_id
        wav_chunk_path.mkdir(parents=True, exist_ok=True)
        full_wav_path = str(wav_chunk_path / "full.wav")
        save_wav(wav, full_wav_path)

        timestamps = vad_segmenter.detect(full_wav_path)

        if not timestamps:
            result = {
                "task_id": task_id,
                "status": "no_speech",
                "segments": [],
                "full_text": "",
            }
            result_path = str(Path(result_dir) / f"{task_id}.json")
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            await tracker.update(task_id, status="completed", result_path=result_path, progress=1.0)
            shutil.rmtree(wav_chunk_path, ignore_errors=True)
            return

        if total_duration > VAD_MAX_SEGMENT_THRESHOLD_S:
            chunks = chunk_audio(wav, VAD_TARGET_SAMPLE_RATE, timestamps,
                                 target_duration_s=VAD_SEGMENT_THRESHOLD_S,
                                 max_duration_s=VAD_MAX_SEGMENT_THRESHOLD_S)
        else:
            sr = VAD_TARGET_SAMPLE_RATE
            chunks = []
            for t_start, t_end in timestamps:
                start_sample = int(t_start * sr)
                end_sample = int(t_end * sr)
                chunk_data = wav[start_sample:end_sample]
                chunks.append((t_start, t_end, chunk_data))

        await tracker.update(task_id, progress=0.1,
                             progress_detail=f"Transcribing {len(chunks)} segments...")

        chunk_paths = []
        for idx, (start_s, end_s, chunk_data) in enumerate(chunks):
            chunk_path = str(wav_chunk_path / f"chunk_{idx:04d}.wav")
            save_wav(chunk_data, chunk_path)
            chunk_paths.append((idx, start_s, end_s, chunk_path))

        async def transcribe_one(idx, start_s, end_s, path):
            try:
                text = await asr_client.transcribe(path)
                return (idx, start_s, end_s, text, None)
            except Exception as e:
                return (idx, start_s, end_s, "", str(e))

        tasks = [transcribe_one(idx, s, e, p) for idx, s, e, p in chunk_paths]
        results_list = await asyncio.gather(*tasks)

        results_list.sort(key=lambda x: x[0])
        total = len(results_list)

        segments = []
        texts = []
        failed_count = 0
        for i, (idx, start_s, end_s, text, error) in enumerate(results_list):
            if error:
                failed_count += 1
                texts.append(f"[Segment {idx} failed: {error}]")
            else:
                cleaned = clean_text(text)
                texts.append(cleaned)
                segments.append({
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "text": cleaned,
                })
            progress = 0.1 + 0.85 * (i + 1) / total
            await tracker.update(task_id, progress=progress,
                                 progress_detail=f"Transcribing segment {i + 1}/{total}")

        full_text = "".join(t for t in texts if t)
        status = "completed" if failed_count < total else "failed"

        result = {
            "task_id": task_id,
            "status": status,
            "segments": segments,
            "full_text": full_text,
        }
        if failed_count > 0 and status == "completed":
            result["warning"] = f"{failed_count}/{total} segments failed and were skipped"

        result_path = str(Path(result_dir) / f"{task_id}.json")
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        await tracker.update(task_id, status="completed", result_path=result_path, progress=1.0)

        shutil.rmtree(wav_chunk_path, ignore_errors=True)

    except Exception as e:
        await tracker.update(task_id, status="failed", error_message=str(e))
