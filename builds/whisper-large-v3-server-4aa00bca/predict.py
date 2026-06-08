import os
import io
import base64
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import librosa
from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        language: Optional[str] = Input(description="Language of the audio (optional, auto-detected if not provided)", default=None),
        task: str = Input(description="Task to perform: 'transcribe' or 'translate'", default="transcribe", choices=["transcribe", "translate"]),
        return_timestamps: str = Input(description="Return timestamps: 'none', 'sentence', or 'word'", default="none", choices=["none", "sentence", "word"]),
    ) -> dict:
        """Transcribe or translate audio using Whisper large-v3."""
        # Load audio
        audio_array, sr = librosa.load(str(audio), sr=16000, mono=True)

        # Prepare generate_kwargs
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        generate_kwargs["task"] = task

        # Prepare return_timestamps
        timestamps = None
        if return_timestamps == "sentence":
            timestamps = True
        elif return_timestamps == "word":
            timestamps = "word"

        # Run inference
        result = self.pipe(
            audio_array,
            generate_kwargs=generate_kwargs,
            return_timestamps=timestamps,
        )

        # Format output
        output = {"text": result["text"]}
        if "chunks" in result:
            output["chunks"] = result["chunks"]

        return output