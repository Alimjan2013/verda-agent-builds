import io
import os
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI()

model = None
tokenizer = None
processor = None
ready = False


class PredictRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    question: str
    generation_mode: str = "hybrid"
    max_new_tokens: int = 8192
    temperature: float = 0.7


def load_image_from_request(req: PredictRequest) -> Image.Image:
    if req.image_base64:
        import base64
        raw = base64.b64decode(req.image_base64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if req.image_url:
        import requests
        raw = requests.get(req.image_url, timeout=30).content
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("image_base64 or image_url is required")


@app.on_event("startup")
def startup():
    global model, tokenizer, processor, ready
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    ready = True


@app.get("/health")
def health():
    return {"ok": ready, "model": MODEL_ID, "device": DEVICE}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        image = load_image_from_request(req)
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": req.question},
            ]}
        ]
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(DEVICE)

        pixel_values = inputs["pixel_values"].to(DTYPE)
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tokenizer,
            max_new_tokens=req.max_new_tokens,
            use_cache=True,
            generation_mode=req.generation_mode,
            temperature=req.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=True,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
