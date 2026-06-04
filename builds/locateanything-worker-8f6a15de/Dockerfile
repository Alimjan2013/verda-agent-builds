FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    TORCH_HOME=/data/hf/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY server.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
