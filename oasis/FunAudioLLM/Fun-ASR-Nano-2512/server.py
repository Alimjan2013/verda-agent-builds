import os
import tempfile
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from funasr import AutoModel

MODEL_ID = "FunAudioLLM/Fun-ASR-Nano-2512"

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    device = "cuda:0" if os.path.exists("/dev/nvidia0") else "cpu"
    _model = AutoModel(
        model=MODEL_ID,
        hub="hf",
        trust_remote_code=True,
        vad_model="funasr/fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )
    yield


app = FastAPI(title="Fun-ASR-Nano-2512 Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    response_format: str = Form(default="json"),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "wav")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        lang_map = {
            "zh": "中文", "en": "英文", "ja": "日文",
            "ko": "韩文", "auto": "中文",
        }
        lang = lang_map.get(language, "中文")
        res = _model.generate(
            input=[tmp_path],
            cache={},
            batch_size=1,
            language=lang,
            itn=True,
        )
        text = res[0]["text"] if res else ""
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return text
    return JSONResponse({"text": text})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
