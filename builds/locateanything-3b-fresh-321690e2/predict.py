import os
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load model once at startup."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        model_path = "nvidia/LocateAnything-3B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        question: str = Input(description="Question or prompt for the model", default="Locate all the instances that matches the following description: person."),
        generation_mode: str = Input(description="Generation mode: fast, slow, or hybrid", default="hybrid", choices=["fast", "slow", "hybrid"]),
        max_new_tokens: int = Input(description="Maximum new tokens to generate", default=2048),
        temperature: float = Input(description="Sampling temperature", default=0.7),
    ) -> str:
        """Run inference on an image."""
        img = Image.open(str(image)).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )

        result = response[0] if isinstance(response, tuple) else response
        return result
