import json
import shutil
import asyncio
from pathlib import Path
from app.audio_utils import load_audio, save_wav
from app.postprocess import clean_text
from app.config import VAD_TARGET_SAMPLE_RATE, VAD_SEGMENT_THRESHOLD_S, VAD_MAX_SEGMENT_THRESHOLD_S


async def process_task(task_id, file_path, tracker, vad_segmenter, asr_client, chunk_dir, result_dir):
    wav_chunk_path = None
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

        # Build chunks from VAD segments directly
        sr = VAD_TARGET_SAMPLE_RATE
        max_samples = VAD_MAX_SEGMENT_THRESHOLD_S * sr
        min_segment_duration = 5.0  # merge segments shorter than 5s

        chunks = []
        pending_start = None
        pending_end = None

        for t_start, t_end in timestamps:
            if pending_start is None:
                pending_start = t_start
                pending_end = t_end
            elif t_start - pending_end < 1.0 and (pending_end - pending_start) < 60:
                # Extend current pending segment (gap < 1s)
                pending_end = t_end
            else:
                # Flush pending segment
                seg_dur = pending_end - pending_start
                if seg_dur <= VAD_MAX_SEGMENT_THRESHOLD_S:
                    chunks.append((pending_start, pending_end, seg_dur))
                else:
                    # Split long segment
                    n_parts = int(seg_dur // VAD_SEGMENT_THRESHOLD_S) + 1
                    part_dur = seg_dur / n_parts
                    for j in range(n_parts):
                        cs = pending_start + j * part_dur
                        ce = pending_start + (j + 1) * part_dur if j < n_parts - 1 else pending_end
                        chunks.append((cs, ce, ce - cs))
                pending_start = t_start
                pending_end = t_end

        # Flush last segment
        if pending_start is not None:
            seg_dur = pending_end - pending_start
            if seg_dur <= VAD_MAX_SEGMENT_THRESHOLD_S:
                chunks.append((pending_start, pending_end, seg_dur))
            else:
                n_parts = int(seg_dur // VAD_SEGMENT_THRESHOLD_S) + 1
                part_dur = seg_dur / n_parts
                for j in range(n_parts):
                    cs = pending_start + j * part_dur
                    ce = pending_start + (j + 1) * part_dur if j < n_parts - 1 else pending_end
                    chunks.append((cs, ce, ce - cs))

        # Merge very short chunks with neighbors
        merged = []
        i = 0
        while i < len(chunks):
            cs, ce, cd = chunks[i]
            if cd < min_segment_duration and merged:
                prev = merged.pop()
                merged.append((prev[0], ce, ce - prev[0]))
            elif cd < min_segment_duration and i + 1 < len(chunks):
                nxt = chunks[i + 1]
                chunks[i + 1] = (cs, nxt[1], nxt[1] - cs)
            else:
                merged.append((cs, ce, cd))
            i += 1

        # Extract actual audio for each chunk
        final_chunks = []
        for cs, ce, _ in merged:
            start_sample = int(cs * sr)
            end_sample = int(ce * sr)
            final_chunks.append((cs, ce, wav[start_sample:end_sample]))

        await tracker.update(task_id, progress=0.1,
                             progress_detail=f"Transcribing {len(final_chunks)} segments...")

        chunk_paths = []
        for idx, (start_s, end_s, chunk_data) in enumerate(final_chunks):
            chunk_path = str(wav_chunk_path / f"chunk_{idx:04d}.wav")
            save_wav(chunk_data, chunk_path)
            chunk_paths.append((idx, start_s, end_s, chunk_path))

        async def transcribe_one(idx, start_s, end_s, path):
            try:
                text = await asr_client.transcribe(path)
                return (idx, start_s, end_s, text, None)
            except Exception as e:
                return (idx, start_s, end_s, "", str(e))

        total = len(chunk_paths)
        segments = []
        transcribed_texts = []
        failed_count = 0

        coros = [transcribe_one(idx, s, e, p) for idx, s, e, p in chunk_paths]
        for completed_idx, coro in enumerate(asyncio.as_completed(coros), 1):
            idx, start_s, end_s, text, error = await coro
            if error:
                failed_count += 1
            else:
                cleaned = clean_text(text)
                transcribed_texts.append(cleaned)
                segments.append({
                    "start": round(start_s, 2),
                    "end": round(end_s, 2),
                    "text": cleaned,
                })
            progress = 0.1 + 0.85 * completed_idx / total
            await tracker.update(task_id, progress=progress,
                                 progress_detail=f"Transcribing segment {completed_idx}/{total}")

        segments.sort(key=lambda x: x["start"])

        full_text = "".join(transcribed_texts)
        status = "completed" if failed_count < total else "failed"

        low_density_segments = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            chars = len(seg["text"])
            density = chars / dur if dur > 0 else 0
            if density < 1.0:
                low_density_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "density": round(density, 1),
                })

        result = {
            "task_id": task_id,
            "status": status,
            "segments": segments,
            "full_text": full_text,
        }
        if failed_count > 0 and status == "completed":
            result["warning"] = f"{failed_count}/{total} segments failed and were skipped"
        if low_density_segments:
            result["low_quality_segments"] = low_density_segments

        result_path = str(Path(result_dir) / f"{task_id}.json")
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        await tracker.update(task_id, status="completed", result_path=result_path, progress=1.0)

        shutil.rmtree(wav_chunk_path, ignore_errors=True)

    except Exception as e:
        await tracker.update(task_id, status="failed", error_message=str(e))
        if wav_chunk_path and Path(wav_chunk_path).exists():
            shutil.rmtree(wav_chunk_path, ignore_errors=True)
