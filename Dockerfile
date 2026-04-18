FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and required directories
COPY app/ ./app/
COPY models/ ./models/
# --- ADD THIS LINE ---
COPY static/ ./static/ 
# ---------------------

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}