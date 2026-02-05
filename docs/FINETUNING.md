# Finetuning Embedding Models for RagShield (ZEDD Paper Approach)

This guide explains how to finetune embedding models following the ZEDD paper (arXiv:2601.12359v1) to improve RagShield's detection accuracy.

## Overview

The ZEDD algorithm detects prompt injections by measuring "drift" between original and cleaned text embeddings. Finetuning the embedding model improves this drift detection.

| Configuration | Accuracy | Cost | Latency |
|--------------|----------|------|---------|
| Base embeddings + heuristic cleaner | ~70% | Free | Fast |
| Base embeddings + LLM cleaner | ~80% | ~$0.0003/doc | Medium |
| **Finetuned embeddings + LLM cleaner** | **~90%** | **~$0.0003/doc** | **Medium** |

## Quick Start

### Step 1: Install Dependencies

```bash
pip install datasets openai sentence-transformers transformers accelerate tqdm scikit-learn
```

### Step 2: Set Your OpenAI API Key

The script uses GPT-4o-mini for text cleaning. You'll need ~$5-10 in OpenAI credits.

```bash
export OPENAI_API_KEY=sk-...
```

### Step 3: Run the Finetuning Script

```bash
python scripts/finetune_local.py
```

The script will:
1. Load the LLMail-Inject dataset
2. Remove duplicates and filter by length
3. Clean injected text with OpenAI API
4. Generate clean-clean pairs
5. Finetune MPNet with CosineSimilarityLoss
6. Calibrate threshold using GMM (paper Section 4.3)
7. Evaluate and export

### Step 4: Use Your Finetuned Model

```python
from ragshield import RagShield

shield = RagShield(config={
    "embeddings": {
        "provider": "local",
        "model": "./ragshield-embeddings-finetuned",
    },
    "cleaning": {
        "method": "llm",
        "llm_model": "gpt-4o-mini",
    },
    "zedd": {
        "threshold": None,  # Auto-loads from model's calibration.json
    },
})

result = shield.scan("Some text to scan...")
```

## Script Options

```bash
# Default settings (5000 samples, 3 epochs)
python scripts/finetune_local.py

# Custom settings
python scripts/finetune_local.py \
    --max-samples 10000 \
    --output-dir ./my-model \
    --epochs 5 \
    --batch-size 32
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max-samples` | 5000 | Maximum samples to process |
| `--output-dir` | `./ragshield-embeddings-finetuned` | Where to save the model |
| `--cache-dir` | `./cache` | Cache directory for resuming |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |

## How It Works (ZEDD Paper)

### Training Data

The paper uses two types of training pairs:

1. **Injected-Clean pairs** (label = 0.0)
   - Original: Text with prompt injection
   - Cleaned: Same text with injection removed by LLM
   - Expected: High embedding drift

2. **Clean-Clean pairs** (label = 1.0)
   - Original: Clean professional email
   - Cleaned: Same email rephrased
   - Expected: Low embedding drift

### Loss Function

```python
from sentence_transformers import losses

# CosineSimilarityLoss trains the model so that:
# - Similar pairs (clean-clean) have cosine similarity → 1.0
# - Dissimilar pairs (injected-clean) have cosine similarity → 0.0
train_loss = losses.CosineSimilarityLoss(model)
```

### Threshold Calibration (Section 4.3)

The paper uses a 2-component GMM to find the optimal threshold:

1. **Fit GMM** on drift scores
2. **Identify components**: Lower mean = clean, higher mean = injected
3. **Find intersection**: Where `w_clean * f_clean(x) = w_inject * f_inject(x)`
4. **Apply FP cap**: Binary search to achieve ≤3% false positive rate

## Configuration Options

### YAML Configuration

```yaml
# ragshield.yaml

embeddings:
  provider: local
  model: ./ragshield-embeddings-finetuned

cleaning:
  method: llm
  llm_model: gpt-4o-mini

zedd:
  threshold: null  # Auto-load from calibration.json

behavior:
  on_detect: flag
```

### Environment Variables

```bash
export RAGSHIELD_EMBEDDINGS__MODEL=./ragshield-embeddings-finetuned
export RAGSHIELD_CLEANING__METHOD=llm
export RAGSHIELD_CLEANING__LLM_MODEL=gpt-4o-mini
```

## Training Details

### Base Model

The script uses `sentence-transformers/all-mpnet-base-v2`:

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Parameters | 110M |
| Speed | Medium |
| Quality | Best for sentence similarity |

### Hardware Requirements

**Training:**
- CPU: Works but slow
- GPU/MPS: Recommended
- RAM: 8GB+
- Time: ~1-2 hours (5K samples)

**Inference:**
- CPU: Works fine for small batches
- GPU/MPS: Recommended for production

## Cost Analysis

| Phase | Cost |
|-------|-----:|
| Cleaning injected text (~5K samples) | ~$1.50 |
| Generating clean pairs (~5K pairs) | ~$1.50 |
| Training (local) | Free |
| **Total one-time** | **~$3-5** |
| **Inference per document** | **~$0.0003** |

## Resumable Training

The script caches intermediate results in `--cache-dir`:

```
cache/
├── cleaned_injected.json  # Cleaned injected samples
└── clean_pairs.json       # Generated clean-clean pairs
```

If the script is interrupted, it will resume from where it left off.

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (try 8 or 4)
- The script uses MPS on Mac automatically

### Model Not Loading
- Ensure the output directory contains all model files
- Check `config.json` exists in the output directory

### Poor Detection
- Ensure you trained with enough data (5K+ samples)
- Check if threshold was properly calibrated
- Verify `calibration.json` exists in the model directory

### API Errors
- Check your OpenAI API key is valid
- Ensure you have sufficient credits
- The script checkpoints every 100 samples, so you can resume

## API Reference

### Using Finetuned Embeddings

```python
from ragshield import RagShield

# Auto-loads threshold from model's calibration.json
shield = RagShield(config={
    "embeddings": {
        "model": "./ragshield-embeddings-finetuned",
    },
})

# Scan documents
result = shield.scan("Document text here")
print(f"Suspicious: {result.is_suspicious}")
print(f"Confidence: {result.confidence:.2%}")
```

### Manual Threshold Override

```python
shield = RagShield(config={
    "embeddings": {
        "model": "./ragshield-embeddings-finetuned",
    },
    "zedd": {
        "threshold": 0.25,  # Override calibrated threshold
    },
})
```

## References

- **ZEDD Paper**: arXiv:2601.12359v1 - "Zero-Shot Embedding Drift Detection"
- **LLMail-Inject**: https://huggingface.co/datasets/microsoft/llmail-inject-challenge
- **sentence-transformers**: https://www.sbert.net/
- **GMM Threshold**: Paper Section 4.3
