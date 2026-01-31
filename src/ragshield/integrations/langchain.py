"""LangChain integration for RagShield."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ragshield.core.config import ShieldConfig
from ragshield.core.exceptions import PromptInjectionDetected
from ragshield.core.results import ScanResult

logger = logging.getLogger(__name__)

# Type alias for LangChain documents (avoid hard dependency)
DocumentType = Any


class ShieldRunnable:
    """
    LangChain Runnable that scans inputs for prompt injections.

    Works with LangChain's LCEL (LangChain Expression Language) and
    can be inserted into any chain using the pipe operator.

    Usage:
        from ragshield.integrations.langchain import ShieldRunnable

        # Basic usage
        chain = retriever | ShieldRunnable() | prompt | llm

        # With configuration
        chain = retriever | ShieldRunnable(
            on_detect="block",
            confidence_threshold=0.7
        ) | prompt | llm

        # Filter mode (remove suspicious documents)
        chain = retriever | ShieldRunnable(on_detect="filter") | prompt | llm

    Modes:
        - "block": Raise PromptInjectionDetected exception
        - "filter": Remove suspicious documents from output
        - "flag": Add _ragshield metadata to documents
        - "warn": Log warning but pass through unchanged
    """

    def __init__(
        self,
        config: Optional[Union[ShieldConfig, Dict[str, Any], str, Path]] = None,
        on_detect: str = "flag",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the ShieldRunnable.

        Args:
            config: RagShield configuration
            on_detect: Action on detection ("block", "filter", "flag", "warn")
            confidence_threshold: Minimum confidence to trigger action
        """
        from ragshield.core.shield import RagShield

        self.shield = RagShield(config=config)
        self.on_detect = on_detect
        self.confidence_threshold = confidence_threshold

    def invoke(
        self,
        input: Union[str, List[str], List[DocumentType], Any],
        config: Any = None,
    ) -> Union[str, List[str], List[DocumentType], Any]:
        """
        Scan input and apply configured action.

        Handles various input types:
        - str: Single text
        - list[str]: List of texts
        - list[Document]: List of LangChain Documents
        - dict with "context" key: Extract and scan context

        Args:
            input: Input to scan
            config: LangChain RunnableConfig (unused but required for interface)

        Returns:
            Processed input based on on_detect mode
        """
        # Normalize input
        docs, input_type = self._normalize_input(input)

        if not docs:
            return input

        # Extract text content
        texts = self._extract_texts(docs)

        # Scan all texts
        results = self.shield.scan(texts)
        if not isinstance(results, list):
            results = [results]

        # Process results
        output = self._process_results(docs, results, input_type)

        # Denormalize output
        return self._denormalize_output(output, input_type, input)

    def _normalize_input(
        self,
        input: Any,
    ) -> Tuple[List[Any], str]:
        """
        Normalize input to list of documents/texts.

        Returns:
            Tuple of (normalized_list, input_type)
        """
        if isinstance(input, str):
            return [input], "string"

        if isinstance(input, list):
            if not input:
                return [], "list"

            if isinstance(input[0], str):
                return input, "list_str"

            # Assume Document-like objects
            return input, "list_doc"

        if isinstance(input, dict):
            # Try to extract relevant content
            for key in ["context", "documents", "docs", "content"]:
                if key in input:
                    docs, _ = self._normalize_input(input[key])
                    return docs, f"dict_{key}"

        # Unknown type, wrap in list
        return [input], "unknown"

    def _extract_texts(self, docs: List[Any]) -> List[str]:
        """Extract text content from documents."""
        texts = []
        for doc in docs:
            if isinstance(doc, str):
                texts.append(doc)
            elif hasattr(doc, "page_content"):
                texts.append(str(doc.page_content))
            elif hasattr(doc, "text"):
                texts.append(str(doc.text))
            elif hasattr(doc, "content"):
                texts.append(str(doc.content))
            else:
                texts.append(str(doc))
        return texts

    def _process_results(
        self,
        docs: List[Any],
        results: List[ScanResult],
        input_type: str,
    ) -> List[Any]:
        """Apply on_detect action to results."""
        # Identify suspicious documents
        suspicious_indices = [
            i for i, r in enumerate(results)
            if r.is_suspicious and r.confidence >= self.confidence_threshold
        ]

        if not suspicious_indices:
            return docs

        if self.on_detect == "block":
            suspicious_results = [results[i] for i in suspicious_indices]
            raise PromptInjectionDetected(
                f"Detected {len(suspicious_indices)} suspicious document(s)",
                results=suspicious_results,
            )

        elif self.on_detect == "filter":
            # Remove suspicious documents
            safe_indices = set(range(len(docs))) - set(suspicious_indices)
            return [docs[i] for i in sorted(safe_indices)]

        elif self.on_detect == "flag":
            # Add metadata to documents
            for i, (doc, result) in enumerate(zip(docs, results)):
                if hasattr(doc, "metadata"):
                    doc.metadata["_ragshield"] = result.to_dict()
                elif isinstance(doc, dict):
                    doc["_ragshield"] = result.to_dict()
            return docs

        elif self.on_detect == "warn":
            # Log warnings
            for i in suspicious_indices:
                result = results[i]
                logger.warning(
                    f"Suspicious document at index {i}: "
                    f"confidence={result.confidence:.2f}, "
                    f"{result.details.summary}"
                )
            return docs

        # Default: pass through
        return docs

    def _denormalize_output(
        self,
        output: List[Any],
        input_type: str,
        original_input: Any,
    ) -> Any:
        """Convert output back to original input type."""
        if input_type == "string":
            return output[0] if output else ""

        if input_type in ("list_str", "list_doc", "list"):
            return output

        if input_type.startswith("dict_"):
            key = input_type.split("_", 1)[1]
            if isinstance(original_input, dict):
                result = dict(original_input)
                result[key] = output
                return result

        return output

    # LangChain Runnable interface methods

    def __or__(self, other: Any) -> Any:
        """Support pipe operator: shield | next_step"""
        try:
            from langchain_core.runnables import RunnableSequence
            return RunnableSequence(first=self, last=other)
        except ImportError:
            raise ImportError(
                "langchain-core is required for pipe operator. "
                "Install with: pip install ragshield[langchain]"
            )

    def __ror__(self, other: Any) -> Any:
        """Support pipe operator: prev_step | shield"""
        try:
            from langchain_core.runnables import RunnableSequence
            return RunnableSequence(first=other, last=self)
        except ImportError:
            raise ImportError(
                "langchain-core is required for pipe operator. "
                "Install with: pip install ragshield[langchain]"
            )

    @property
    def InputType(self) -> type:
        """LangChain Runnable interface."""
        return Any

    @property
    def OutputType(self) -> type:
        """LangChain Runnable interface."""
        return Any


# Convenience alias
ShieldRetriever = ShieldRunnable
