# Docker Deployment Guide for AgroVision-AI

## Overview

This guide explains how to deploy AgroVision-AI in a Docker container on Linux, enabling full IndicTrans2 dynamic translation without the macOS mutex crashes.

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 1.29+ installed
- Hugging Face account with access to gated models:
  - [ai4bharat/indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
  - [ai4bharat/indictrans2-indic-en-1B](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B)
- HF_TOKEN with read access

## Quick Start

```bash
# 1. Set your Hugging Face token
export HF_TOKEN=REDACTED

# 2. Build the container
docker-compose build

# 3. Run the application
docker-compose up

# 4. Access at http://localhost:5001
```

## File Structure

Create these files in your project root:

### 1. Dockerfile

```dockerfile
# AgroVision-AI Dockerfile
# Python 3.10 on Debian for stable IndicTrans2 support

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
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
ENV FLASK_APP=app.py
ENV SKIP_INDICTRANS2_MODELS=0
ENV TOKENIZERS_PARALLELISM=false
ENV RAYON_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Expose Flask port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001')" || exit 1

# Run the application
CMD ["python", "app.py"]
```

### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  agrovision:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: agrovision-ai
    ports:
      - "5001:5001"
    volumes:
      # Persist uploaded images
      - ./data:/app/data
      # Cache HuggingFace models to avoid re-downloading
      - huggingface_cache:/root/.cache/huggingface
    environment:
      # Pass HF token from host environment or .env file
      - HF_TOKEN=${HF_TOKEN}
      - HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}
      # Flask configuration
      - FLASK_DEBUG=0
      - FLASK_ENV=production
      # Translation settings
      - ENABLE_TRANSLATION=True
      - DEFAULT_LANGUAGE=en
      - TRANSLATION_CACHE_SIZE=1000
      - SKIP_INDICTRANS2_MODELS=0
      # Stability settings
      - TOKENIZERS_PARALLELISM=false
      - RAYON_NUM_THREADS=2
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
    restart: unless-stopped
    mem_limit: 4g
    cpus: 2.0
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  huggingface_cache:
    driver: local
```

### 3. .dockerignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore
.gitattributes

# Documentation
*.md
docs/

# Test files
tests/
__pycache__/
*.pyc

# Large data files (keep only necessary)
archive/
.cache/

# Logs
*.log

# Docker
Dockerfile
docker-compose.yml
.dockerignore
```

## Building and Running

### Build the Image

```bash
# Build with cache
docker-compose build

# Build without cache (fresh install)
docker-compose build --no-cache

# Check image size
docker images agrovision-ai
```

### Run the Container

```bash
# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Development Mode

For development with live code reloading:

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  agrovision-dev:
    extends:
      file: docker-compose.yml
      service: agrovision
    volumes:
      - ./:/app
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - FLASK_DEBUG=1
      - FLASK_ENV=development
    command: python app.py

volumes:
  huggingface_cache:
```

Run with:
```bash
docker-compose -f docker-compose.dev.yml up
```

## Verifying IndicTrans2

### Check Model Loading

```bash
# View container logs to see model loading
docker-compose logs -f agrovision

# Expected output:
# INFO:src.utils.translation:Loading IndicTrans2 models...
# INFO:src.utils.translation:Authenticated with Hugging Face Hub...
# INFO:src.utils.translation:Loading ai4bharat/indictrans2-en-indic-1B...
# INFO:src.utils.translation:IndicTrans2 models loaded successfully!
```

### Test Dynamic Translation

```bash
# Access the container shell
docker-compose exec agrovision bash

# Test translation
python3 -c "
from src.utils.translation import translate
print(translate('rice', 'kn', 'en'))  # Should print: ಅಕ್ಕಿ
print(translate('Good morning', 'hi', 'en'))  # Should print Hindi translation
"

# Exit container
exit
```

### Test Web Interface

1. Open browser: http://localhost:5001
2. Select a non-English language (e.g., Kannada)
3. Submit the form with test data
4. Verify:
   - ✅ Crop name translates (e.g., "rice" → "ಅಕ್ಕಿ")
   - ✅ AI narrative translates
   - ✅ No English leakage in UI

## Performance Optimization

### Model Caching

Models are cached in a Docker volume to avoid re-downloading (~2-3GB):

```bash
# Check cache volume
docker volume inspect agrovision-ai_huggingface_cache

# Backup cache
docker run --rm -v agrovision-ai_huggingface_cache:/cache \
  -v $(pwd):/backup alpine tar czf /backup/REDACTED.tar.gz -C /cache .

# Restore cache
docker run --rm -v agrovision-ai_huggingface_cache:/cache \
  -v $(pwd):/backup alpine tar xzf /backup/REDACTED.tar.gz -C /cache
```

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
# For low-memory systems (minimum 2GB RAM)
mem_limit: 2g
cpus: 1.0

# For high-performance systems (recommended 4GB+ RAM)
mem_limit: 8g
cpus: 4.0
```

### Multi-stage Build (Smaller Image)

```dockerfile
# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

CMD ["python", "app.py"]
```

## Troubleshooting

### Models Not Loading

**Symptom:** IndicTrans2 models fail to load
**Solutions:**
1. Check HF_TOKEN is set: `docker-compose config | grep HF_TOKEN`
2. Request access to gated repos on Hugging Face
3. Check disk space: `docker system df`
4. Increase memory limit in docker-compose.yml

### Out of Memory

**Symptom:** Container restarts or crashes
**Solutions:**
1. Increase `mem_limit` in docker-compose.yml to 4g or 8g
2. Reduce concurrent users
3. Add swap space on host

### Port Already in Use

**Symptom:** "Port 5001 is already allocated"
**Solutions:**
```bash
# Find process using port
lsof -i :5001

# Change port in docker-compose.yml
ports:
  - "5002:5001"  # Use port 5002 instead
```

### Permission Errors

**Symptom:** Cannot write to volumes
**Solutions:**
```bash
# Fix volume permissions
docker-compose down
docker volume rm agrovision-ai_huggingface_cache
docker-compose up -d
```

## Production Deployment

### Using Docker Hub

```bash
# Tag image
docker tag agrovision-ai:latest yourusername/agrovision-ai:latest

# Push to registry
docker push yourusername/agrovision-ai:latest

# Pull on production server
docker pull yourusername/agrovision-ai:latest
```

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml agrovision

# Scale service
docker service scale agrovision_agrovision=3

# Check status
docker stack services agrovision
```

### Using Kubernetes

Create `k8s/deployment.yml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agrovision-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agrovision
  template:
    metadata:
      labels:
        app: agrovision
    spec:
      containers:
      - name: agrovision
        image: yourusername/agrovision-ai:latest
        ports:
        - containerPort: 5001
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: agrovision-service
spec:
  selector:
    app: agrovision
  ports:
  - port: 80
    targetPort: 5001
  type: LoadBalancer
```

Apply with:
```bash
kubectl apply -f k8s/deployment.yml
```

## Security Best Practices

1. **Never commit HF_TOKEN** to version control
2. **Use Docker secrets** for production:
   ```bash
   echo "your_REDACTED" | docker secret create REDACTED -
   ```
3. **Scan images** for vulnerabilities:
   ```bash
   docker scan agrovision-ai:latest
   ```
4. **Run as non-root** user:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```
5. **Use HTTPS** in production with reverse proxy (nginx/traefik)

## Monitoring

### Container Health

```bash
# Check health status
docker ps

# View resource usage
docker stats agrovision-ai

# Check logs
docker-compose logs --tail=100 -f
```

### Application Metrics

Add to `app.py`:
```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'models_loaded': translation_service.models_loaded}

@app.route('/metrics')
def metrics():
    return translation_service.get_cache_stats()
```

## Backup and Recovery

### Backup Data

```bash
# Backup application data
docker-compose exec agrovision tar czf /tmp/backup.tar.gz /app/data
docker cp agrovision-ai:/tmp/backup.tar.gz ./backup_$(date +%Y%m%d).tar.gz

# Backup models cache
docker run --rm -v agrovision-ai_huggingface_cache:/cache \
  -v $(pwd):/backup alpine tar czf /backup/models_$(date +%Y%m%d).tar.gz -C /cache .
```

### Restore Data

```bash
# Restore application data
docker cp backup_20250116.tar.gz agrovision-ai:/tmp/
docker-compose exec agrovision tar xzf /tmp/backup_20250116.tar.gz -C /

# Restore models cache
docker run --rm -v agrovision-ai_huggingface_cache:/cache \
  -v $(pwd):/backup alpine tar xzf /backup/models_20250116.tar.gz -C /cache
```

## Summary

This Docker deployment:
- ✅ Solves macOS mutex crash
- ✅ Enables full IndicTrans2 translation
- ✅ Provides consistent cross-platform deployment
- ✅ Includes health checks and logging
- ✅ Supports development and production modes
- ✅ Optimized for resource usage
- ✅ Production-ready with security best practices

**Expected Result:** Full dynamic translation of crop names and AI narratives in all 12 languages!

---

**Next Steps:**
1. Request HuggingFace model access
2. Create Dockerfile and docker-compose.yml
3. Build and test container
4. Verify IndicTrans2 loads successfully
5. Test dynamic translation in browser
