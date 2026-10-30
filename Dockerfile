FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY prompts/ prompts/
COPY static/ static/
COPY scripts/ scripts/

# 非 root 使用者；storage/ 用 volume 掛進來
RUN useradd --create-home --uid 1000 app && mkdir -p /app/storage /app/data && chown -R app:app /app
USER app

EXPOSE 8000

# /health 在 Chroma 或 Ollama 不正常時回 503
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# 只能跑 1 個 worker：限流和同時生成上限都存在這個行程的記憶體裡
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
