from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch

# ===== Device Setup =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
model = whisper.load_model("base", device=DEVICE)

app = FastAPI()


@app.post("/whisper")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required"
        )

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        # Get detailed result with timestamps
        result = model.transcribe(
            temp.name,
            verbose=False
        )

    # Extract sentence-level segments
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip()
        })

    return JSONResponse(content={
        "filename": file.filename,
        "language": result.get("language"),
        "segments": segments
    })


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"