from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Whisper Setup ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=DEVICE)

# === NLLB Translation Setup ===
nllb_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_name, src_lang="eng_Latn")
translator = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

app = FastAPI()


def translate_en_to_vi(text: str) -> str:
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    generated = translator.generate(
        **tokens,
        forced_bos_token_id=tokenizer.lang_code_to_id["vie_Latn"]
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


@app.post("/whisper")
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Audio file is required")

    with NamedTemporaryFile(delete=True, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp.flush()

        # Transcribe English segments
        transcription = whisper_model.transcribe(temp.name)

    results = []
    for idx, segment in enumerate(transcription["segments"]):
        english_text = segment["text"].strip()
        vietnamese_text = translate_en_to_vi(english_text)

        results.append({
            "englishText": english_text,
            "vietnameseText": vietnamese_text,
            "startTimeMs": int(segment["start"] * 1000),
            "endTimeMs": int(segment["end"] * 1000),
            "order": idx + 1
        })

    return JSONResponse(content=results)


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"