import os
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="LocateAnything-3B Server")

model = None
tokenizer = None
processor = None


class ImageTextRequest(BaseModel):
    image_url: Optional[str] = None
    question: str = Field(default="Describe the image.")
    generation_mode: str = Field(default="hybrid")
    max_new_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GroundRequest(BaseModel):
    image_url: Optional[str] = None
    phrase: str
    mode: str = Field(default="single")
    max_new_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generation_mode: str = Field(default="hybrid")


@app.on_event("startup")
def _load():
    global model, tokenizer, processor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE).eval()


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_ID}


@app.get("/")
def root():
    return {"name": "LocateAnything-3B Server", "health": "/health"}


@app.post("/predict")
def predict(req: ImageTextRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    raise HTTPException(status_code=400, detail="This build exposes /health only; use /v1/chat/completions-style serving in a future revision.")
