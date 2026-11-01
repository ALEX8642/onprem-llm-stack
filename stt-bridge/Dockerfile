# OpenAI /v1/audio/transcriptions -> Wyoming bridge (see app.py). No model, tiny footprint.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastapi==0.115.14 uvicorn==0.34.0 python-multipart==0.0.20
WORKDIR /app
COPY app.py .
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s CMD curl -sf http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
