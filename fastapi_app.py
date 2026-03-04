from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =============================
# Device Setup
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# Load Whisper
# =============================
whisper_model = whisper.load_model("base", device=DEVICE)

# =============================
# Load NLLB Translation Model
# =============================
nllb_model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    nllb_model_name,
    src_lang="eng_Latn"
)

translator = AutoModelForSeq2SeqLM.from_pretrained(
    nllb_model_name
).to(DEVICE)

translator.eval()

app = FastAPI()


# =============================
# Batch Translation Function
# =============================
def translate_batch_en_to_vi(texts: list[str]) -> list[str]:
    if not texts:
        return []

    with torch.no_grad():
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        generated = translator.generate(
            **tokens,
            forced_bos_token_id=tokenizer.lang_code_to_id["vie_Latn"],
            max_length=512
        )

        return tokenizer.batch_decode(generated, skip_special_tokens=True)


# =============================
# API Endpoint
# =============================
@app.post("/whisper")
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")

    # Save file temporarily
    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        transcription = whisper_model.transcribe(temp.name)

    segments = transcription.get("segments", [])

    # Extract English texts
    english_texts = [seg["text"].strip() for seg in segments]

    # Batch translate (FAST)
    vietnamese_texts = translate_batch_en_to_vi(english_texts)

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


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"