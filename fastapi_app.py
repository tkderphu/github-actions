from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio

# =====================================
# Device Setup
# =====================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Load Whisper Model
# =====================================
whisper_model = whisper.load_model("base", device=DEVICE)

# =====================================
# Load NLLB Translation Model
# =====================================
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    NLLB_MODEL_NAME,
    src_lang="eng_Latn",
    use_fast=False
)

translator = AutoModelForSeq2SeqLM.from_pretrained(
    NLLB_MODEL_NAME
).to(DEVICE)

translator.eval()

# Pre-calculate Vietnamese BOS token safely
VI_BOS_TOKEN_ID = tokenizer.convert_tokens_to_ids("vie_Latn")

if VI_BOS_TOKEN_ID is None:
    raise ValueError("Vietnamese language token 'vie_Latn' not found in tokenizer.")

# =====================================
# FastAPI App
# =====================================
app = FastAPI(title="Whisper + NLLB Translation API")

# =====================================
# Batch Translation Function
# =====================================
def translate_batch_en_to_vi(texts: list[str]) -> list[str]:
    if not texts:
        return []

    with torch.no_grad():
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)

        generated = translator.generate(
            **tokens,
            forced_bos_token_id=VI_BOS_TOKEN_ID,
            max_length=512,
            num_beams=4  # better translation quality
        )

        return tokenizer.batch_decode(generated, skip_special_tokens=True)

# =====================================
# Whisper Endpoint
# =====================================
@app.post("/whisper")
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        # Run Whisper in separate thread
        transcription = await asyncio.to_thread(
            whisper_model.transcribe,
            temp.name
        )

    segments = transcription.get("segments", [])

    if not segments:
        return JSONResponse(content=[])

    # Extract English texts
    english_texts = [seg["text"].strip() for seg in segments]

    # Batch translate
    vietnamese_texts = await asyncio.to_thread(
        translate_batch_en_to_vi,
        english_texts
    )

    # Build response structure
    results = []

    for idx, segment in enumerate(segments):
        results.append({
            "englishText": english_texts[idx],
            "vietnameseText": vietnamese_texts[idx] if idx < len(vietnamese_texts) else "",
            "startTimeMs": int(segment["start"] * 1000),
            "endTimeMs": int(segment["end"] * 1000),
            "order": idx + 1
        })

    return JSONResponse(content=results)

# =====================================
# Redirect root to Swagger docs
# =====================================
@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"