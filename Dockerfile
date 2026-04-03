# Use Python 3.11 slim — good balance of size and compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Railway injects $PORT at runtime — default to 8000 locally
ENV PORT=8000

# Create directories for knowledge base and vector store
RUN mkdir -p ocd_documentation ocd_documentation_vector

# Start the FastAPI app with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
