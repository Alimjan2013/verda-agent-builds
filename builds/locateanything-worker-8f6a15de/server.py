import os
import re
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models/locateanything-3b")

app = FastAPI(title="LocateAnything Worker", version="1.0")

_tokenizer = None
_processor = None
_model = None

class PredictRequest(BaseModel):
    image_url: Optional[str] = None
    question: str
    generation_mode: str = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    verbose: bool = True

class TaskRequest(BaseModel):
    image_url: Optional[str] = None
    task: str
    phrase: Optional[str] = None
    categories: Optional[list[str]] = None
    generation_mode: str = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    verbose: bool = True


def load_image_from_url(url: str) -> Image.Image:
    from io import BytesIO
    import base64
    import requests

    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def ensure_model():
    global _tokenizer, _processor, _model
    if _model is not None:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()


def run_prediction(image: Image.Image, question: str, generation_mode: str, max_new_tokens: int, temperature: float, verbose: bool) -> Any:
    ensure_model()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = _processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = _processor.process_vision_info(messages)
    inputs = _processor(text=[text], images=images, videos=videos, return_tensors="pt").to(DEVICE)

    pixel_values = inputs["pixel_values"].to(DTYPE)
    response = _model.generate(
        pixel_values=pixel_values,
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
    result = {"answer": response[0] if isinstance(response, tuple) else response}
    if isinstance(response, tuple) and len(response) >= 3:
        result["history"] = response[1]
        result["stats"] = response[2]
    return result


@app.get("/health")
def health():
    return {"ok": True, "model_id": MODEL_ID, "device": DEVICE}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    image = load_image_from_url(req.image_url)
    return run_prediction(image, req.question, req.generation_mode, req.max_new_tokens, req.temperature, req.verbose)


@app.post("/task")
def task(req: TaskRequest):
    if not req.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    image = load_image_from_url(req.image_url)
    if req.task == "detect":
        cats = req.categories or []
        prompt = f"Locate all the instances that matches the following description: {'</c>'.join(cats)}."
    elif req.task == "ground_single":
        prompt = f"Locate a single instance that matches the following description: {req.phrase or ''}."
    elif req.task == "ground_multi":
        prompt = f"Locate all the instances that match the following description: {req.phrase or ''}."
    elif req.task == "ground_text":
        prompt = f"Please locate the text referred as {req.phrase or ''}."
    elif req.task == "detect_text":
        prompt = "Detect all the text in box format."
    elif req.task == "ground_gui":
        prompt = f"Locate the region that matches the following description: {req.phrase or ''}."
    elif req.task == "point":
        prompt = f"Point to: {req.phrase or ''}."
    else:
        raise HTTPException(status_code=400, detail=f"unknown task: {req.task}")
    return run_prediction(image, prompt, req.generation_mode, req.max_new_tokens, req.temperature, req.verbose)


@app.post("/parse_boxes")
def parse_boxes(payload: dict):
    answer = payload.get("answer", "")
    image_width = int(payload.get("image_width", 0))
    image_height = int(payload.get("image_height", 0))
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = [int(g) for g in m.groups()]
        boxes.append({"x1": x1 / 1000 * image_width, "y1": y1 / 1000 * image_height, "x2": x2 / 1000 * image_width, "y2": y2 / 1000 * image_height})
    return {"boxes": boxes}
