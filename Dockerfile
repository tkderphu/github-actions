FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (single layer)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install Whisper
RUN pip install --no-cache-dir "git+https://github.com/openai/whisper.git"

# ---- HuggingFace Auth ----
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Optional: install HF CLI
RUN pip install --no-cache-dir huggingface_hub

# Pre-download models at build time (VERY IMPORTANT)
RUN python -c "import whisper; whisper.load_model('base')"

RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"

# Copy source code last (better Docker cache)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]