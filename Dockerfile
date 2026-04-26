FROM python:3.10-slim

RUN sed -i "s@http://deb.debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i "s@http://deb.debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/listener

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p data/uploads data/chunks data/results

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
