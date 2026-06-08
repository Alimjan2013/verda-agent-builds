import re
from typing import List, Optional

import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


MODEL_ID = "nvidia/LocateAnything-3B"


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Grounding prompt", default="Locate all the instances that matches the following description: object."),
        generation_mode: str = Input(description="Generation mode", default="hybrid", choices=["fast", "slow", "hybrid"]),
        max_new_tokens: int = Input(description="Maximum new tokens", default=2048, ge=1, le=8192),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(description="Top-p", default=0.9, ge=0.0, le=1.0),
    ) -> str:
        img = Image.open(str(image)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
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
            top_p=top_p,
            repetition_penalty=1.1,
            verbose=False,
        )
        return response[0] if isinstance(response, tuple) else str(response)

