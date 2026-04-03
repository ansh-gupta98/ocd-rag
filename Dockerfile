FROM python:3.11-slim

WORKDIR /app

# Only what's needed to compile faiss-cpu + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ocd_documentation ocd_documentation_vector

ENV PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT