# Listener ASR Service API

OpenAI 兼容的会议录音转录服务。上传音频/视频文件，异步获取带时间戳的转录文本。

## Base URL

```
http://localhost:8000
```

## 认证

无需认证。

## 端点

### GET /health

健康检查。

**响应**
```json
{"status": "ok"}
```

---

### POST /v1/tasks

上传音频或视频文件，创建转录任务。

**请求**

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | multipart file | 上传的音频/视频文件 |

**支持格式**

`.mp3`, `.wav`, `.mp4`, `.mov`, `.mkv`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.aac`

**限制**

- 最大文件大小：2 GB
- 最大并发任务：3

**响应** `200`

```json
{
  "task_id": "e3a4c02f-6022-42ea-9f3b-7216e067e927",
  "status": "pending"
}
```

**错误**

| 状态码 | 说明 |
|--------|------|
| 400 | 不支持的文件格式 |
| 413 | 文件超过 2GB |
| 503 | 服务器繁忙（并发任务已满） |

---

### GET /v1/tasks/{task_id}

查询任务状态和进度。

**请求参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | path (UUID) | 任务 ID |

**响应** `200`

```json
{
  "task_id": "e3a4c02f-6022-42ea-9f3b-7216e067e927",
  "status": "processing",
  "progress": 0.45,
  "progress_detail": "Transcribing segment 62/137"
}
```

**任务状态**

| 状态 | 说明 |
|------|------|
| `pending` | 任务已创建，等待处理 |
| `processing` | 正在转录中 |
| `completed` | 转录完成 |
| `failed` | 转录失败 |

**错误**

| 状态码 | 说明 |
|--------|------|
| 404 | 任务不存在 |

---

### GET /v1/tasks/{task_id}/result

获取已完成任务的转录结果。

**请求参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | path (UUID) | 任务 ID |

**响应** `200`

```json
{
  "task_id": "e3a4c02f-6022-42ea-9f3b-7216e067e927",
  "status": "completed",
  "segments": [
    {
      "start": 0.77,
      "end": 17.64,
      "text": "好，那我们进入正式一下了。前几节课我们只是介绍了一下人工智能编程的发展的轨迹？"
    },
    {
      "start": 18.73,
      "end": 33.8,
      "text": "人工智能编程经历了哪几个阶段？第一阶段是什么？"
    }
  ],
  "full_text": "好，那我们进入正式一下了...。人工智能编程经历了哪几个阶段？...",
  "low_quality_segments": [
    {"start": 1503.8, "end": 1558.1, "density": 0.5}
  ]
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `status` | string | `completed` / `no_speech` / `failed` |
| `segments` | array | 转录分段，按时间排序 |
| `segments[].start` | float | 段起始时间（秒） |
| `segments[].end` | float | 段结束时间（秒） |
| `segments[].text` | string | 该段转录文本 |
| `full_text` | string | 全部文本拼接 |
| `low_quality_segments` | array | 低密度段标记（<1 字符/秒），可选 |
| `warning` | string | 部分失败警告，可选 |

**错误**

| 状态码 | 说明 |
|--------|------|
| 404 | 任务不存在 |
| 409 | 任务尚未完成 |
| 422 | 任务失败 |

---

## 处理流程

```
上传文件 (.mp4/.wav/.mp3/...)
  │
  ▼
① FFmpeg 音频提取 → 16kHz/16bit/mono WAV
  │
  ▼
② FireRedVAD 语音检测 → 分割语音段
  │
  ▼
③ 智能分段（合并短段，拆分长段 >60s）
  │
  ▼
④ Qwen3-ASR (vLLM) 转录每个分段
  │
  ▼
⑤ 后处理清洗（去重复、去幻觉）
  │
  ▼
⑥ 输出 JSON（文本 + 时间戳）
```

## 使用示例

### 上传文件

```bash
curl -X POST http://localhost:8000/v1/tasks \
  -F "file=@/path/to/meeting.mp3"
```

### 轮询状态

```bash
# 使用返回的 task_id
curl http://localhost:8000/v1/tasks/e3a4c02f-6022-42ea-9f3b-7216e067e927
```

### 获取结果

```bash
curl http://localhost:8000/v1/tasks/e3a4c02f-6022-42ea-9f3b-7216e067e927/result
```

### Python 调用

```python
import httpx
import time

BASE = "http://localhost:8000"

# 上传
with open("meeting.mp3", "rb") as f:
    resp = httpx.post(f"{BASE}/v1/tasks", files={"file": f})
    task_id = resp.json()["task_id"]

# 轮询
while True:
    resp = httpx.get(f"{BASE}/v1/tasks/{task_id}")
    data = resp.json()
    if data["status"] in ("completed", "failed"):
        break
    time.sleep(10)

# 取结果
resp = httpx.get(f"{BASE}/v1/tasks/{task_id}/result")
result = resp.json()
print(result["full_text"])
```

## 错误码总览

| HTTP | 说明 |
|------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 任务不存在 |
| 409 | 任务状态冲突 |
| 413 | 文件过大 |
| 422 | 任务处理失败 |
| 503 | 服务器繁忙 |
