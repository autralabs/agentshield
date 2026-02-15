"""Pipeline fingerprint for threshold keying.

Thresholds depend on the full pipeline configuration — not just the embedding
model.  The same model paired with a different cleaner produces a different
drift distribution and therefore needs a different threshold.

Fingerprint format:
    {provider_host}::{embedding_model}::{cleaning_method}[::{cleaning_model}]

Each segment is URL-encoded before joining, so the literal ``"::"``
delimiter is safe even when values contain colons (e.g. IPv6 hosts).

Examples:
    local::all-MiniLM-L6-v2::heuristic
    openai.com::text-embedding-3-small::llm::gpt-4o-mini
    openrouter.ai::meta-llama/llama-3.1-8b::llm::gpt-4o-mini
"""

from __future__ import annotations

from typing import Dict, Optional
from urllib.parse import quote, unquote, urlparse


def extract_host(base_url: Optional[str], provider: str) -> str:
    """Extract host identifier from a base_url or provider name.

    Includes the port when present, so ``localhost:11434`` and
    ``localhost:8000`` produce distinct identifiers (important when
    Ollama and vLLM run on the same machine).

    Known canonical endpoints (e.g. ``https://api.openai.com/v1``) are
    normalised to the same value as ``base_url=None`` for that provider,
    so explicit-default and omitted-default produce the same key.

    Args:
        base_url: Optional API base URL (e.g. ``https://openrouter.ai/api/v1``)
        provider: Provider type (``"local"``, ``"openai"``, ``"mlx"``, …)

    Returns:
        Host string used in fingerprint (e.g. ``"openrouter.ai"``,
        ``"localhost:11434"``, ``"local"``)
    """
    # Default hosts per provider
    defaults: Dict[str, str] = {
        "local": "local",
        "openai": "openai.com",
        "mlx": "local",
    }

    # Canonical hostnames that should map back to the provider default.
    # Prevents cache fragmentation when users explicitly set the default URL.
    _CANONICAL_HOSTS: Dict[str, str] = {
        "api.openai.com": "openai.com",
    }

    # Default ports per scheme — these are invisible to the server and
    # should not appear in identity keys.
    _DEFAULT_PORTS = {"https": 443, "http": 80}

    if base_url:
        parsed = urlparse(base_url)
        host = parsed.hostname
        if host:
            port = parsed.port
            scheme_default = _DEFAULT_PORTS.get(parsed.scheme)
            # Strip default port for the scheme (e.g. :443 for https)
            if port is not None and port == scheme_default:
                port = None

            # Normalise canonical endpoints (after port stripping)
            canonical = _CANONICAL_HOSTS.get(host)
            if canonical is not None and port is None:
                return canonical

            if port is not None:
                return f"{host}:{port}"
            return host

    return defaults.get(provider, provider)


def create_pipeline_fingerprint(
    provider_host: str,
    embedding_model: str,
    cleaning_method: str,
    cleaning_model: Optional[str] = None,
    cleaning_host: Optional[str] = None,
) -> str:
    """Build a fingerprint string for a specific pipeline configuration.

    Args:
        provider_host: Host identifier (from :func:`extract_host`)
        embedding_model: Embedding model name
        cleaning_method: Cleaning method name (``"heuristic"``, ``"llm"``, …)
        cleaning_model: Optional model used by the cleaner (e.g. ``"gpt-4o-mini"``)
        cleaning_host: Optional host where the cleaner runs. Included when
            provided by the caller. Most current manager flows encode cleaner
            backend identity directly in ``cleaning_model`` for LLM/finetuned.

    Returns:
        Fingerprint string such as ``"local::all-MiniLM-L6-v2::heuristic"``
    """
    # Escape each part so delimiter collisions are impossible (notably IPv6
    # hosts like ::1) while keeping common model/host strings readable.
    def _escape(value: str) -> str:
        return quote(value, safe="-._~/+@()[]")

    parts = [_escape(provider_host), _escape(embedding_model), _escape(cleaning_method)]
    if cleaning_model:
        parts.append(_escape(cleaning_model))
    if cleaning_host:
        parts.append(_escape(cleaning_host))
    return "::".join(parts)


def parse_pipeline_fingerprint(fingerprint: str) -> Dict[str, Optional[str]]:
    """Parse a fingerprint string back into its components.

    Args:
        fingerprint: Fingerprint produced by :func:`create_pipeline_fingerprint`

    Returns:
        Dictionary with keys ``provider_host``, ``embedding_model``,
        ``cleaning_method``, and ``cleaning_model`` (may be ``None``).

    Raises:
        ValueError: If the fingerprint has fewer than 3 parts.
    """
    parts = fingerprint.split("::")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid fingerprint (expected at least 3 '::'-separated parts): {fingerprint!r}"
        )

    def _unescape(value: str) -> str:
        return unquote(value)

    return {
        "provider_host": _unescape(parts[0]),
        "embedding_model": _unescape(parts[1]),
        "cleaning_method": _unescape(parts[2]),
        "cleaning_model": _unescape(parts[3]) if len(parts) > 3 else None,
        "cleaning_host": _unescape(parts[4]) if len(parts) > 4 else None,
    }
