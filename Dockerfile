FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Default port for Hugging Face Spaces; can be overridden by $PORT env var
ENV PORT=7860
EXPOSE 7860

# Bind to runtime $PORT (HF uses 7860, Railway/Render inject their own)
CMD ["sh", "-c", "gunicorn run:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 300 --log-level info"]

