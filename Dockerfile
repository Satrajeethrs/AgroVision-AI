# AgroVision-AI Dockerfile
# Python 3.10 on Debian for stable IndicTrans2 support

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed data/raw translations

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py
ENV SKIP_INDICTRANS2_MODELS=0
ENV TOKENIZERS_PARALLELISM=false
ENV RAYON_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
# Default LM Studio inside container to host.docker.internal (macOS/Windows)
ENV LMSTUDIO_BASE_URL=http://host.docker.internal:1234
ENV GEMINI_API_KEY=""
ENV OPENAI_API_KEY=""

# Expose Flask port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001')" || exit 1

# Run the application
CMD ["python", "app.py"]
