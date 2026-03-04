from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Load Whisper
whisper_model = whisper.load_model("base", device=DEVICE)

# 2️⃣ Load English → Vietnamese model
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name).to(DEVICE)

app = FastAPI()


def translate_en_to_vi(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    translated = translator_model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


@app.post("/whisper")
async def transcribe_and_translate(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")

    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        transcription = whisper_model.transcribe(temp.name)

    results = []

    for index, segment in enumerate(transcription["segments"]):
        english_text = segment["text"].strip()
        vietnamese_text = translate_en_to_vi(english_text)

        results.append({
            "englishText": english_text,
            "vietnameseText": vietnamese_text,
            "startTimeMs": int(segment["start"] * 1000),
            "endTimeMs": int(segment["end"] * 1000),
            "order": index + 1
        })

    return JSONResponse(content=results)