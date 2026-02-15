"""Tests for pipeline fingerprint."""

from __future__ import annotations

import pytest

from pyagentshield.threshold.fingerprint import (
    create_pipeline_fingerprint,
    extract_host,
    parse_pipeline_fingerprint,
)


class TestExtractHost:
    def test_from_base_url(self):
        assert extract_host("https://openrouter.ai/api/v1", "openai") == "openrouter.ai"

    def test_from_base_url_with_port(self):
        assert extract_host("http://localhost:11434/v1", "openai") == "localhost:11434"

    def test_default_openai(self):
        assert extract_host(None, "openai") == "openai.com"

    def test_default_local(self):
        assert extract_host(None, "local") == "local"

    def test_default_mlx(self):
        assert extract_host(None, "mlx") == "local"

    def test_unknown_provider_uses_name(self):
        assert extract_host(None, "custom") == "custom"

    def test_empty_url_falls_back(self):
        assert extract_host("", "openai") == "openai.com"

    def test_ipv6_with_port(self):
        assert extract_host("http://[::1]:11434/v1", "openai") == "::1:11434"


class TestCreatePipelineFingerprint:
    def test_heuristic_local(self):
        fp = create_pipeline_fingerprint("local", "all-MiniLM-L6-v2", "heuristic")
        assert fp == "local::all-MiniLM-L6-v2::heuristic"

    def test_llm_with_model(self):
        fp = create_pipeline_fingerprint(
            "openai.com", "text-embedding-3-small", "llm", "gpt-4o-mini"
        )
        assert fp == "openai.com::text-embedding-3-small::llm::gpt-4o-mini"

    def test_openrouter(self):
        fp = create_pipeline_fingerprint(
            "openrouter.ai", "meta-llama/llama-3.1-8b", "llm", "gpt-4o-mini"
        )
        assert fp == "openrouter.ai::meta-llama/llama-3.1-8b::llm::gpt-4o-mini"

    def test_finetuned(self):
        fp = create_pipeline_fingerprint("local", "all-MiniLM-L6-v2", "finetuned")
        assert fp == "local::all-MiniLM-L6-v2::finetuned"

    def test_same_model_different_hosts_differ(self):
        fp1 = create_pipeline_fingerprint(
            "openai.com", "text-embedding-3-small", "heuristic"
        )
        fp2 = create_pipeline_fingerprint(
            "openrouter.ai", "text-embedding-3-small", "heuristic"
        )
        assert fp1 != fp2

    def test_same_model_different_cleaners_differ(self):
        fp1 = create_pipeline_fingerprint("local", "all-MiniLM-L6-v2", "heuristic")
        fp2 = create_pipeline_fingerprint("local", "all-MiniLM-L6-v2", "llm", "gpt-4o-mini")
        assert fp1 != fp2


class TestParsePipelineFingerprint:
    def test_round_trip_without_cleaning_model(self):
        fp = create_pipeline_fingerprint("local", "all-MiniLM-L6-v2", "heuristic")
        parsed = parse_pipeline_fingerprint(fp)
        assert parsed == {
            "provider_host": "local",
            "embedding_model": "all-MiniLM-L6-v2",
            "cleaning_method": "heuristic",
            "cleaning_model": None,
            "cleaning_host": None,
        }

    def test_round_trip_with_cleaning_model(self):
        fp = create_pipeline_fingerprint(
            "openai.com", "text-embedding-3-small", "llm", "gpt-4o-mini"
        )
        parsed = parse_pipeline_fingerprint(fp)
        assert parsed == {
            "provider_host": "openai.com",
            "embedding_model": "text-embedding-3-small",
            "cleaning_method": "llm",
            "cleaning_model": "gpt-4o-mini",
            "cleaning_host": None,
        }

    def test_invalid_fingerprint_raises(self):
        with pytest.raises(ValueError, match="Invalid fingerprint"):
            parse_pipeline_fingerprint("only::two")

    def test_round_trip_with_cleaning_host(self):
        fp = create_pipeline_fingerprint(
            "local", "all-MiniLM-L6-v2", "llm", "gpt-4o-mini", "openrouter.ai"
        )
        parsed = parse_pipeline_fingerprint(fp)
        assert parsed == {
            "provider_host": "local",
            "embedding_model": "all-MiniLM-L6-v2",
            "cleaning_method": "llm",
            "cleaning_model": "gpt-4o-mini",
            "cleaning_host": "openrouter.ai",
        }

    def test_round_trip_ipv6_host(self):
        fp = create_pipeline_fingerprint("::1:11434", "custom-model", "heuristic")
        parsed = parse_pipeline_fingerprint(fp)
        assert parsed == {
            "provider_host": "::1:11434",
            "embedding_model": "custom-model",
            "cleaning_method": "heuristic",
            "cleaning_model": None,
            "cleaning_host": None,
        }

    def test_round_trip_literal_percent_sequence(self):
        fp = create_pipeline_fingerprint("local", "abc%3A%3Adef", "heuristic")
        parsed = parse_pipeline_fingerprint(fp)
        assert parsed == {
            "provider_host": "local",
            "embedding_model": "abc%3A%3Adef",
            "cleaning_method": "heuristic",
            "cleaning_model": None,
            "cleaning_host": None,
        }


class TestCollisionPrevention:
    """Verify distinct endpoints produce distinct identifiers."""

    def test_different_ports_different_hosts(self):
        """localhost:11434 (Ollama) vs localhost:8000 (vLLM) must differ."""
        h1 = extract_host("http://localhost:11434/v1", "openai")
        h2 = extract_host("http://localhost:8000/v1", "openai")
        assert h1 != h2
        assert h1 == "localhost:11434"
        assert h2 == "localhost:8000"

    def test_standard_port_omitted_for_https(self):
        """https://openai.com (no explicit port) gives just hostname."""
        h = extract_host("https://openai.com/v1", "openai")
        assert h == "openai.com"

    def test_explicit_443_stripped_for_https(self):
        """https://api.openai.com:443 should normalize to openai.com."""
        h = extract_host("https://api.openai.com:443/v1", "openai")
        assert h == "openai.com"

    def test_explicit_80_stripped_for_http(self):
        """http://example.com:80 should strip the port."""
        h = extract_host("http://example.com:80/v1", "openai")
        assert h == "example.com"

    def test_non_default_port_on_canonical_host_kept(self):
        """api.openai.com:8080 is non-standard, should not normalize."""
        h = extract_host("https://api.openai.com:8080/v1", "openai")
        assert h == "api.openai.com:8080"

    def test_same_model_different_ports_different_fingerprints(self):
        fp1 = create_pipeline_fingerprint("localhost:11434", "nomic-embed-text", "heuristic")
        fp2 = create_pipeline_fingerprint("localhost:8000", "nomic-embed-text", "heuristic")
        assert fp1 != fp2

    def test_same_llm_model_different_cleaner_hosts(self):
        """gpt-4o-mini via OpenAI vs OpenRouter = different fingerprints."""
        fp1 = create_pipeline_fingerprint(
            "local", "all-MiniLM-L6-v2", "llm", "gpt-4o-mini"
        )
        fp2 = create_pipeline_fingerprint(
            "local", "all-MiniLM-L6-v2", "llm", "gpt-4o-mini", "openrouter.ai"
        )
        assert fp1 != fp2

    def test_cleaner_host_none_when_default(self):
        """No cleaning_host = no extra segment in fingerprint."""
        fp = create_pipeline_fingerprint(
            "local", "all-MiniLM-L6-v2", "llm", "gpt-4o-mini"
        )
        assert fp.count("::") == 3  # 4 parts, 3 separators
