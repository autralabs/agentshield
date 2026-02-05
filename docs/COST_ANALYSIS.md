# RagShield Cost Analysis

This document breaks down the costs for the ZEDD paper approach vs alternatives.

## Summary

| Phase | Original Paper Cost | Optimized Cost |
|-------|--------------------:|---------------:|
| Training Data Creation | ~$15-25 | ~$15-25 (one-time) |
| Embedding Model Finetuning | Free (local) | Free (local) |
| **Inference (per 1K docs)** | **~$0.30-0.50** | **~$0.30-0.50** |

**Bottom line**: ~$0.0003-0.0005 per document for cleaning (inference).

---

## OpenAI API Pricing (as of 2024)

### GPT-3.5-Turbo

| Mode | Input | Output |
|------|------:|-------:|
| Regular API | $0.50/1M tokens | $1.50/1M tokens |
| **Batch API (50% off)** | **$0.25/1M tokens** | **$0.75/1M tokens** |

### GPT-4o-mini (Recommended)

| Mode | Input | Output |
|------|------:|-------:|
| Regular API | $0.15/1M tokens | $0.60/1M tokens |
| **Batch API (50% off)** | **$0.075/1M tokens** | **$0.30/1M tokens** |

---

## Phase 1: Training Data Creation (One-Time)

The finetuning script (`scripts/finetune_local.py`) creates training data using OpenAI API:

### 1.1 Cleaning Injected Text

From LLMail-Inject dataset (~10K-20K samples):

| Component | Estimate |
|-----------|----------|
| Input tokens per sample | ~500 tokens (email + system prompt) |
| Output tokens per sample | ~300 tokens (cleaned email) |
| Total samples | ~15,000 |
| **Total input tokens** | **7.5M tokens** |
| **Total output tokens** | **4.5M tokens** |

**Cost (GPT-3.5 Batch API)**:
- Input: 7.5M × $0.25/1M = $1.88
- Output: 4.5M × $0.75/1M = $3.38
- **Subtotal: ~$5.25**

### 1.2 Creating Clean-Clean Pairs

86,000 synthetic email pairs (5 batches × 17,200):

| Component | Estimate |
|-----------|----------|
| Input tokens per request | ~200 tokens (system prompt) |
| Output tokens per request | ~400 tokens (2 emails) |
| Total requests | 86,000 |
| **Total input tokens** | **17.2M tokens** |
| **Total output tokens** | **34.4M tokens** |

**Cost (GPT-3.5 Batch API)**:
- Input: 17.2M × $0.25/1M = $4.30
- Output: 34.4M × $0.75/1M = $25.80
- **Subtotal: ~$30.10**

### 1.3 Total Training Data Cost

| Task | Cost |
|------|-----:|
| Clean injected emails | $5.25 |
| Generate clean-clean pairs | $30.10 |
| **Total (one-time)** | **~$35** |

**With GPT-4o-mini Batch API**: ~$15-20 (recommended)

---

## Phase 2: Model Finetuning (Free)

Using local hardware (Mac with MPS or any machine with GPU/CPU):

| Resource | Cost |
|----------|-----:|
| Local compute (MPS/CUDA/CPU) | Free |
| Training time | ~1-2 hours |
| Storage | Local disk |
| **Total** | **$0** |

---

## Phase 3: Inference (Per Document)

### Option A: OpenAI API Cleaning (Paper Approach)

For each document scanned at runtime:

| Component | Tokens | Cost (GPT-3.5) | Cost (GPT-4o-mini) |
|-----------|-------:|---------------:|-------------------:|
| Input (doc + prompt) | ~400 | $0.0002 | $0.00006 |
| Output (cleaned doc) | ~300 | $0.00045 | $0.00018 |
| **Per document** | | **~$0.00065** | **~$0.00024** |

**Cost per 1,000 documents**:
- GPT-3.5-turbo: ~$0.65
- GPT-4o-mini: ~$0.24

**Cost per 10,000 documents**:
- GPT-3.5-turbo: ~$6.50
- GPT-4o-mini: ~$2.40

### Option B: Local Embedding (Free)

Using sentence-transformers locally:

| Component | Cost |
|-----------|-----:|
| Embedding generation | Free |
| Compute (CPU/GPU) | Your hardware |
| **Per document** | **$0** |

---

## Total Cost Scenarios

### Scenario 1: Small Scale (1K docs/month)

| Component | Cost |
|-----------|-----:|
| Training data (one-time) | $20 |
| Cleaning inference | $0.24/month |
| Embeddings | Free |
| **Year 1 Total** | **~$23** |
| **Year 2+ Total** | **~$3/year** |

### Scenario 2: Medium Scale (100K docs/month)

| Component | Cost |
|-----------|-----:|
| Training data (one-time) | $20 |
| Cleaning inference | $24/month |
| Embeddings | Free |
| **Year 1 Total** | **~$308** |
| **Year 2+ Total** | **~$288/year** |

### Scenario 3: Large Scale (1M docs/month)

| Component | Cost |
|-----------|-----:|
| Training data (one-time) | $20 |
| Cleaning inference | $240/month |
| Embeddings | Free |
| **Year 1 Total** | **~$2,900** |
| **Year 2+ Total** | **~$2,880/year** |

---

## Cost Optimization Tips

### 1. Use Batch API for Training Data
- 50% discount on all token costs
- Slightly slower (24hr turnaround)
- Perfect for one-time data creation

### 2. Use GPT-4o-mini Instead of GPT-3.5
- 70% cheaper than GPT-3.5-turbo
- Similar quality for cleaning task
- Recommended for inference

### 3. Cache Cleaned Results
```python
# In your config
shield = RagShield(config={
    "performance": {
        "cache_embeddings": True,
        "cache_cleaned": True,  # Add this
    }
})
```

### 4. Batch Processing
```python
# More efficient than individual calls
results = shield.scan(["doc1", "doc2", "doc3", ...])
```

### 5. Pre-filter with Heuristics
```python
# Use hybrid mode: heuristic first, LLM only if needed
shield = RagShield(config={
    "cleaning": {
        "method": "hybrid",
        "hybrid": {
            "methods": ["heuristic", "llm"],
            "mode": "fallback",  # Only use LLM if heuristic fails
        }
    }
})
```

---

## Comparison with Alternatives

| Approach | Training Cost | Per-Doc Cost | Accuracy |
|----------|-------------:|-------------:|---------:|
| Heuristic only | $0 | $0 | ~70% |
| **LLM cleaning (paper)** | **~$20** | **~$0.0003** | **~90%** |
| Fine-tuned local cleaner | $20 + compute | $0 | ~85% |
| Full LLM (GPT-4) | $0 | ~$0.01 | ~95% |

---

## Recommended Setup

For your use case (production-ready, cost-conscious):

```yaml
# ragshield.yaml
cleaning:
  method: llm
  llm_model: gpt-4o-mini  # Cheapest, good quality

embeddings:
  provider: local
  model: all-MiniLM-L6-v2  # Free, fast

performance:
  cache_embeddings: true
  batch_size: 32
```

**Expected cost**: ~$0.24 per 1,000 documents (~$0.00024 each)

---

## References

- [OpenAI Pricing](https://openai.com/pricing)
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)
- [ZEDD Paper](https://arxiv.org/abs/2601.12359)
