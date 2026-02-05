"""Heuristic text cleaner using regex patterns."""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple


class HeuristicCleaner:
    """
    Text cleaner using heuristic regex patterns.

    This cleaner removes common injection patterns without using an LLM.
    It's fast, free, and achieves ~85% effectiveness compared to LLM cleaning.

    The patterns are derived from:
    - OWASP prompt injection documentation
    - Common jailbreak patterns
    - LLMail-Inject dataset analysis
    - BIPIA benchmark patterns
    """

    def __init__(self) -> None:
        """Initialize the heuristic cleaner."""
        self._method = "heuristic"
        self._patterns = self._compile_patterns()

    @property
    def method(self) -> str:
        """Get the cleaning method name."""
        return self._method

    def _compile_patterns(self) -> List[Tuple[Pattern[str], str]]:
        """Compile regex patterns for cleaning."""
        patterns: List[Tuple[str, str, int]] = [
            # Instruction override attempts
            (
                r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)",
                "",
                re.IGNORECASE,
            ),
            (
                r"disregard\s+(everything|all|any)\s+(above|before|previous|prior)",
                "",
                re.IGNORECASE,
            ),
            (
                r"forget\s+(everything|all)\s+(you\s+)?(know|learned|were\s+told)",
                "",
                re.IGNORECASE,
            ),
            (
                r"do\s+not\s+follow\s+(the\s+)?(previous|above|prior)\s+instructions?",
                "",
                re.IGNORECASE,
            ),
            (
                r"override\s+(previous|all|prior)\s+(instructions?|rules?|guidelines?)",
                "",
                re.IGNORECASE,
            ),

            # New instruction markers
            (
                r"new\s+(instructions?|task|prompt|rules?):\s*",
                "",
                re.IGNORECASE,
            ),
            (
                r"(start|begin)\s+(new|fresh)\s+(instructions?|conversation|session)",
                "",
                re.IGNORECASE,
            ),
            (
                r"(actual|real|true)\s+(instructions?|task|prompt):\s*",
                "",
                re.IGNORECASE,
            ),

            # System prompt manipulation
            (
                r"system\s*prompt:\s*",
                "",
                re.IGNORECASE,
            ),
            (
                r"(reveal|show|display|output|print)\s+(your\s+)?(system\s+)?(prompt|instructions?)",
                "",
                re.IGNORECASE,
            ),

            # Role manipulation
            (
                r"you\s+are\s+now\s+(a|an|the)?\s*\w+",
                "",
                re.IGNORECASE,
            ),
            (
                r"from\s+now\s+on,?\s+you\s+(are|will|must|should)",
                "",
                re.IGNORECASE,
            ),
            (
                r"pretend\s+(to\s+be|you\s+are|that\s+you)",
                "",
                re.IGNORECASE,
            ),
            (
                r"act\s+as\s+(if\s+)?(you\s+are\s+)?(a|an|the)?",
                "",
                re.IGNORECASE,
            ),
            (
                r"roleplay\s+as\s+",
                "",
                re.IGNORECASE,
            ),

            # Jailbreak patterns
            (
                r"DAN\s*(mode)?",
                "",
                re.IGNORECASE,
            ),
            (
                r"developer\s+mode\s*(enabled|on|activated)?",
                "",
                re.IGNORECASE,
            ),
            (
                r"jailbreak(ed)?",
                "",
                re.IGNORECASE,
            ),
            (
                r"bypass\s+(safety|filter|restriction|content\s+policy)",
                "",
                re.IGNORECASE,
            ),
            (
                r"no\s+(restrictions?|rules?|limits?|boundaries)",
                "",
                re.IGNORECASE,
            ),

            # Fake system boundaries / formatting
            (
                r"\[/?INST\]",
                "",
                0,
            ),
            (
                r"<\|im_(start|end)\|>",
                "",
                0,
            ),
            (
                r"<\|system\|>|<\|user\|>|<\|assistant\|>",
                "",
                0,
            ),
            (
                r"</?(system|user|assistant|human|ai)>",
                "",
                re.IGNORECASE,
            ),
            (
                r"```system\s*\n?",
                "",
                re.IGNORECASE,
            ),
            (
                r"###\s*(system|instruction|human|assistant)\s*:?\s*",
                "",
                re.IGNORECASE,
            ),

            # Authority establishment
            (
                r"!!!+[^!]+!!!+",  # !!!important text!!!
                "",
                0,
            ),
            (
                r"IMPORTANT:\s*",
                "",
                0,
            ),
            (
                r"(URGENT|CRITICAL|EMERGENCY):\s*",
                "",
                0,
            ),
            (
                r"(administrator|admin|root|sudo)\s+(mode|access|privileges?)",
                "",
                re.IGNORECASE,
            ),

            # Context manipulation
            (
                r"</?context[^>]*>",
                "",
                re.IGNORECASE,
            ),
            (
                r"</?(end_)?context[^>]*>",
                "",
                re.IGNORECASE,
            ),
            (
                r"---+\s*(end|start)\s*(of\s+)?(context|conversation|session)\s*---+",
                "",
                re.IGNORECASE,
            ),

            # Hidden content markers
            (
                r"<!--.*?-->",  # HTML comments
                "",
                re.DOTALL,
            ),
            (
                r"<style[^>]*>.*?</style>",  # CSS that might hide text
                "",
                re.DOTALL | re.IGNORECASE,
            ),
            (
                r"<script[^>]*>.*?</script>",  # Scripts
                "",
                re.DOTALL | re.IGNORECASE,
            ),

            # Zero-width and invisible characters
            (
                r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]",
                "",
                0,
            ),

            # Base64 encoded blocks (likely hidden instructions)
            (
                r"[A-Za-z0-9+/]{100,}={0,2}",
                "[ENCODED_REMOVED]",
                0,
            ),

            # XML/task injection patterns
            (
                r"</?assistant_task[^>]*>",
                "",
                re.IGNORECASE,
            ),
            (
                r"</?email_notification[^>]*>",
                "",
                re.IGNORECASE,
            ),
            (
                r"</?execute[^>]*>",
                "",
                re.IGNORECASE,
            ),
        ]

        return [
            (re.compile(pattern, flags), replacement)
            for pattern, replacement, flags in patterns
        ]

    def clean(self, text: str) -> str:
        """
        Clean text by removing injection patterns.

        Args:
            text: Original text

        Returns:
            Cleaned text
        """
        if not text:
            return text

        cleaned = text

        # Apply all patterns
        for pattern, replacement in self._patterns:
            cleaned = pattern.sub(replacement, cleaned)

        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean multiple texts."""
        return [self.clean(text) for text in texts]
