"""Decorator API for RagShield."""

from __future__ import annotations

import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from ragshield.core.config import ShieldConfig
from ragshield.core.exceptions import PromptInjectionDetected

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def shield(
    on_detect: str = "warn",
    confidence_threshold: float = 0.5,
    scan_args: Optional[List[str]] = None,
    config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to protect functions from prompt injection.

    Scans string and document arguments before the function executes.
    Can block, warn, or flag based on detection results.

    Args:
        on_detect: Action on detection
            - "block": Raise PromptInjectionDetected exception
            - "warn": Log warning but continue execution
            - "flag": Silent (for later inspection)
        confidence_threshold: Minimum confidence to trigger action (0.0-1.0)
        scan_args: Names of arguments to scan. If None, scans all string/list args.
        config: Optional RagShield configuration

    Returns:
        Decorator function

    Example:
        >>> @shield(on_detect="warn")
        ... def process_documents(query: str, docs: list[str]) -> str:
        ...     return llm.invoke(build_prompt(query, docs))

        >>> @shield(on_detect="block", scan_args=["documents"])
        ... def answer_question(question: str, documents: list[str]) -> str:
        ...     # Only 'documents' will be scanned, not 'question'
        ...     return generate_answer(question, documents)
    """
    from ragshield.core.shield import RagShield

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Initialize shield once per decorated function
        _shield = RagShield(config=config)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Extract texts to scan from arguments
            texts_to_scan = _extract_texts(func, args, kwargs, scan_args)

            if texts_to_scan:
                # Scan all texts
                results = _shield.scan(texts_to_scan)
                if not isinstance(results, list):
                    results = [results]

                # Filter by confidence threshold
                suspicious = [
                    r for r in results
                    if r.confidence >= confidence_threshold and r.is_suspicious
                ]

                if suspicious:
                    if on_detect == "block":
                        raise PromptInjectionDetected(
                            f"Blocked: {len(suspicious)} suspicious input(s) detected",
                            results=suspicious,
                        )
                    elif on_detect == "warn":
                        for result in suspicious:
                            logger.warning(
                                f"Prompt injection detected "
                                f"(confidence={result.confidence:.2f}): "
                                f"{result.details.summary}"
                            )
                    # "flag" mode: do nothing, just continue

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _extract_texts(
    func: Callable[..., Any],
    args: tuple,
    kwargs: Dict[str, Any],
    scan_args: Optional[List[str]],
) -> List[str]:
    """
    Extract string/list arguments to scan from function call.

    Args:
        func: The decorated function
        args: Positional arguments
        kwargs: Keyword arguments
        scan_args: Specific argument names to scan (None = all)

    Returns:
        List of text strings to scan
    """
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    texts: List[str] = []

    for name, value in bound.arguments.items():
        # Skip if scan_args specified and this arg not in it
        if scan_args and name not in scan_args:
            continue

        texts.extend(_extract_texts_from_value(value))

    return texts


def _extract_texts_from_value(value: Any) -> List[str]:
    """Extract text strings from a value (handles nested structures)."""
    texts: List[str] = []

    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, list):
        for item in value:
            texts.extend(_extract_texts_from_value(item))
    elif isinstance(value, dict):
        for v in value.values():
            texts.extend(_extract_texts_from_value(v))
    elif hasattr(value, "page_content"):
        # LangChain Document
        texts.append(str(value.page_content))
    elif hasattr(value, "text"):
        # LlamaIndex Node/Document
        texts.append(str(value.text))

    return texts
