# Changelog

All notable changes to this project are documented in this file.

## [0.1.3] - 2026-02-15

### Added

- Pipeline-aware threshold identity using fingerprints that include embedding backend host, embedding model, cleaning method, and cleaner identity where applicable.
- OpenAI-compatible endpoint support for both embeddings and LLM cleaning via `base_url`, `api_key`, and `default_headers`.
- Embedding capability discovery for unknown OpenAI-compatible models: explicit dimensions, static defaults, disk cache restore, and first-response discovery.

### Changed

- Threshold resolution now prioritizes pipeline fingerprint keys and only allows legacy model-name fallback for heuristic-equivalent pipelines.
- Threshold cache migration now preserves legacy keys, adds cache format versioning, and writes a backup of the original file.
- Host normalization now avoids cache/key fragmentation by handling canonical OpenAI hosts and default ports consistently.
- OpenAI embedding dimension discovery persists to `dimensions_cache.json` and is keyed by model plus endpoint host.

### Fixed

- Batch embedding flow now safely handles mixed empty/non-empty inputs when dimensions are discovered dynamically.
- Hybrid cleaner wiring now forwards OpenAI-compatible fields to nested LLM cleaners.
- Dimension and threshold key collisions caused by endpoint host/port ambiguity are resolved.

### Docs

- Updated README and configuration examples for pipeline fingerprinting, OpenAI-compatible providers, and capability discovery behavior.
- Updated deployment and environment documentation for `v0.1.3` release behavior.
