import os
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models/locateanything-3b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="LocateAnything Worker")
_tokenizer = None
_processor = None
_model = None


class PredictRequest(BaseModel):
    image_url: str
    question: str
    generation_mode: str = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    verbose: bool = True


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_id": MODEL_ID}


@app.on_event("startup")
def startup():
    global _tokenizer, _processor, _model
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE).eval()


def load_image(url: str) -> Image.Image:
    import base64
    import requests

    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def predict_with_prompt(image: Image.Image, question: str, generation_mode: str, max_new_tokens: int, temperature: float, verbose: bool):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = _processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = _processor.process_vision_info(messages)
    inputs = _processor(text=[text], images=images, videos=videos, return_tensors="pt").to(DEVICE)

    response = _model.generate(
        pixel_values=inputs["pixel_values"].to(DTYPE),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=_tokenizer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        generation_mode=generation_mode,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        verbose=verbose,
    )
    out = {"answer": response[0] if isinstance(response, tuple) else response}
    if isinstance(response, tuple) and len(response) >= 3:
        out["history"] = response[1]
        out["stats"] = response[2]
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    image = load_image(req.image_url)
    return predict_with_prompt(image, req.question, req.generation_mode, req.max_new_tokens, req.temperature, req.verbose)
