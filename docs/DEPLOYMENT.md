# Deployment Guide

AgentShield uses sentence-transformers embedding models (~80MB) that are downloaded from HuggingFace Hub on first use. In production, you should pre-download the model during deployment to avoid cold-start latency.

## Quick Setup

### Python Script

```python
import pyagentshield

# Pre-download and validate the default model
result = pyagentshield.setup()
print(result.message)  # "Model 'all-MiniLM-L6-v2' ready (384d embeddings)."

# Or with a specific model
result = pyagentshield.setup(model_name="all-mpnet-base-v2")

# Or with full config
result = pyagentshield.setup(config={
    "embeddings": {"model": "./agentshield-embeddings-finetuned"},
    "cleaning": {"method": "llm"},
})
```

### CLI

```bash
# Download the default model
agentshield setup

# Download a specific model
agentshield setup --model all-mpnet-base-v2

# Use a config file
agentshield setup --config agentshield.yaml

# Check if model is cached (for health checks)
agentshield setup --check

# JSON output (for scripting)
agentshield setup --output json
```

## Docker

### Multi-stage Build (Recommended)

```dockerfile
FROM python:3.11-slim AS builder

# Install agentshield
RUN pip install pyagentshield[all]

# Pre-download the model during build
RUN agentshield setup --model all-MiniLM-L6-v2

# --- Production stage ---
FROM python:3.11-slim

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY . /app
WORKDIR /app

# Verify model is ready
HEALTHCHECK CMD agentshield setup --check

CMD ["python", "app.py"]
```

### Single-stage Build

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download model at build time
RUN agentshield setup

COPY . .

HEALTHCHECK CMD agentshield setup --check
CMD ["python", "app.py"]
```

### With Finetuned Model

If you have a finetuned model, copy it into the image:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your finetuned model
COPY ./agentshield-embeddings-finetuned /app/agentshield-embeddings-finetuned

# Validate it works
RUN agentshield setup --model /app/agentshield-embeddings-finetuned

COPY . .
CMD ["python", "app.py"]
```

## Framework Integration

### FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pyagentshield

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load the model
    result = pyagentshield.setup()
    print(f"AgentShield ready: {result.message}")
    yield
    # Shutdown: nothing to clean up

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    cached = pyagentshield.is_model_cached()
    return {"pyagentshield": "ready" if cached else "not_ready"}
```

### Django

```python
# myapp/apps.py
from django.apps import AppConfig

class MyAppConfig(AppConfig):
    name = "myapp"

    def ready(self):
        import pyagentshield
        result = pyagentshield.setup()
        print(f"AgentShield ready: {result.message}")
```

### Flask

```python
from flask import Flask
import pyagentshield

app = Flask(__name__)

# Setup during app initialization
with app.app_context():
    result = pyagentshield.setup()
    print(f"AgentShield ready: {result.message}")
```

### Plain Python Script

```python
import pyagentshield

# Check first, setup only if needed
if not pyagentshield.is_model_cached():
    print("Downloading model...")
    result = pyagentshield.setup()
    print(result.message)

# Now safe to use — no download latency
result = pyagentshield.scan("Some text to check")
```

## Kubernetes

### Health Check

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: myapp
      image: myapp:latest
      livenessProbe:
        exec:
          command: ["pyagentshield", "setup", "--check"]
        initialDelaySeconds: 5
        periodSeconds: 30
      readinessProbe:
        exec:
          command: ["pyagentshield", "setup", "--check"]
        initialDelaySeconds: 3
        periodSeconds: 10
```

### Init Container

```yaml
apiVersion: v1
kind: Pod
spec:
  initContainers:
    - name: model-downloader
      image: myapp:latest
      command: ["pyagentshield", "setup"]
      volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
  containers:
    - name: myapp
      image: myapp:latest
      volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
  volumes:
    - name: model-cache
      emptyDir: {}
```

## Readiness Check API

```python
import pyagentshield

# Lightweight — no model loading
pyagentshield.is_model_cached()  # True/False

# Full check — downloads + validates
result = pyagentshield.setup()
result.success       # bool
result.model_name    # str
result.dimensions    # int
result.download_time_ms   # float
result.validation_time_ms # float
```

## What Happens Without Setup

If you skip `agentshield setup`, the model will be downloaded automatically on the first `scan()` call. A warning will be emitted:

```
UserWarning: Embedding model 'all-MiniLM-L6-v2' is not cached and will be
downloaded (~80MB). This may cause slow startup in production.
Run 'agentshield setup' or call pyagentshield.setup() during deployment
to pre-download the model.
```

This is fine for development but not recommended for production.
