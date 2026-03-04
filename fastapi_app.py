from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import whisper
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=DEVICE)

app = FastAPI()


@app.post("/whisper")
async def transcribe_and_translate(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")

    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        # 1️⃣ Transcribe (original language)
        transcription = model.transcribe(
            temp.name,
            task="transcribe"
        )

        # 2️⃣ Translate to Vietnamese
        translation = model.transcribe(
            temp.name,
            task="translate",
            language="vi"
        )

    results = []

    # Ensure segment alignment
    for index, segment in enumerate(transcription["segments"]):
        vietnamese_text = translation["segments"][index]["text"] \
            if index < len(translation["segments"]) else ""

        results.append({
            "englishText": segment["text"].strip(),
            "vietnameseText": vietnamese_text.strip(),
            "startTimeMs": int(segment["start"] * 1000),
            "endTimeMs": int(segment["end"] * 1000),
            "order": index + 1
        })

    return JSONResponse(content=results)