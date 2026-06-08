import base64
import io
import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "nvidia/LocateAnything-3B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

app = FastAPI()
worker = None


def load_image(image_data_url: Optional[str] = None, image_base64: Optional[str] = None) -> Image.Image:
    if image_data_url:
        if "," in image_data_url:
            image_data_url = image_data_url.split(",", 1)[1]
        raw = base64.b64decode(image_data_url)
    elif image_base64:
        raw = base64.b64decode(image_base64)
    else:
        raise ValueError("No image provided")
    return Image.open(io.BytesIO(raw)).convert("RGB")


class LocateAnythingWorker:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str, generation_mode: str = "hybrid", max_new_tokens: int = 8192, temperature: float = 0.7, verbose: bool = True):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        return {"answer": response[0] if isinstance(response, tuple) else response}


@app.on_event("startup")
def startup():
    global worker
    worker = LocateAnythingWorker(MODEL_ID, device=DEVICE, dtype=DTYPE)


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_ID}


@app.post("/predict")
def predict(payload: dict):
    try:
        image = load_image(payload.get("image_data_url"), payload.get("image_base64"))
        question = payload.get("question", "Locate all the instances that matches the following description: object.")
        generation_mode = payload.get("generation_mode", "hybrid")
        max_new_tokens = int(payload.get("max_new_tokens", 8192))
        temperature = float(payload.get("temperature", 0.7))
        return worker.predict(image, question, generation_mode=generation_mode, max_new_tokens=max_new_tokens, temperature=temperature)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
