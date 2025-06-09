# Use official python base image
FROM python:3.11-slim

# Install system dependencies for ffmpeg and Rust build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libffi-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Pre-download the model to cache it in the image
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True); AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)"
# Pre-download toxic-bert and emotion models to cache in image
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoTokenizer.from_pretrained('unitary/toxic-bert'); \
    AutoModel.from_pretrained('unitary/toxic-bert')"

RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base'); \
    AutoModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')"

# Copy your app code
COPY . .

# Expose FastAPI default port
EXPOSE 8080

# Run the app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
