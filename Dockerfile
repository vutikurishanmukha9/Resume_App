# Stage 1: Build the React Frontend
FROM node:20-alpine AS build-stage
WORKDIR /app
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm ci
COPY frontend ./
RUN npm run build

# Stage 2: Build the FastAPI Backend
FROM python:3.10-slim AS production-stage

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code including data handling modules
COPY backend ./backend

# Copy the built React UI from the frontend stage
COPY --from=build-stage /app/frontend/dist ./frontend/dist

# Secure AppUser configuration
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 300 --log-level info"]

