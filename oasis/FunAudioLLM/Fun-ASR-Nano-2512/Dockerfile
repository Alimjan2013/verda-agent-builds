FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/data
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    funasr \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    soundfile \
    librosa

COPY server.py /app/server.py

EXPOSE 8000

CMD ["python3", "/app/server.py"]
