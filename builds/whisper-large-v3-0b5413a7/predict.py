import os
import io
import numpy as np
import torch
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import cog

class Predictor(cog.BasePredictor):
    def setup(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def predict(self, audio: cog.Path) -> str:
        audio_seg = AudioSegment.from_file(str(audio))
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
        samples /= (1 << (audio_seg.sample_width * 8 - 1))

        if audio_seg.channels > 1:
            samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)

        sampling_rate = audio_seg.frame_rate
        max_secs = 30
        step = max_secs * sampling_rate

        transcripts = []
        for start in range(0, len(samples), step):
            chunk = samples[start:start + step]
            out = self.pipe({"array": chunk, "sampling_rate": sampling_rate})
            transcripts.append(out["text"])

        return " ".join(transcripts)