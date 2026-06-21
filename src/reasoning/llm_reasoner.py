"""
LLM reasoner for the Temporal Grounding and Verification Framework.

Provides grounded and ungrounded question-answering through the Ollama LLM
served via an OpenAI-compatible API.  Grounded reasoning uses retrieved
temporal facts as evidence; ungrounded reasoning serves as a baseline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI, APIConnectionError, APITimeoutError

from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from src.reasoning.prompt_templates import (
    SYSTEM_PROMPT_GROUNDED,
    SYSTEM_PROMPT_UNGROUNDED,
    USER_PROMPT_TEMPLATE,
    EVIDENCE_FORMAT_TEMPLATE,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY_SECONDS = 2.0


class LLMReasoner:
    """Interface to a local Ollama LLM for temporal question answering.

    Supports two modes:
        - **Grounded**: the model receives retrieved temporal facts and must
          base its answer solely on them.
        - **Ungrounded**: the model answers from its parametric knowledge
          (baseline comparison).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialise the LLM reasoner.

        Args:
            base_url: Ollama API base URL.  Defaults to config value.
            model: Model name (e.g. 'llama3.1').  Defaults to config value.
            temperature: Sampling temperature.  Defaults to config value.
            max_tokens: Maximum tokens in the response.  Defaults to config
                value.
        """
        self.base_url: str = base_url or OLLAMA_BASE_URL
        self.model: str = model or OLLAMA_MODEL
        self.temperature: float = (
            temperature if temperature is not None else LLM_TEMPERATURE
        )
        self.max_tokens: int = max_tokens or LLM_MAX_TOKENS

        self.client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
        )
        logger.info(
            "LLMReasoner initialised: model=%s, base_url=%s",
            self.model,
            self.base_url,
        )

    # ------------------------------------------------------------------
    # Evidence formatting
    # ------------------------------------------------------------------

    def _format_evidence(self, facts: list[dict[str, Any]]) -> str:
        """Format a list of temporal facts into a numbered evidence block.

        Uses ``EVIDENCE_FORMAT_TEMPLATE`` to produce one line per fact in
        the format: ``1. (subject, predicate, object, timestamp)``

        Args:
            facts: List of fact dicts, each with keys subject, predicate,
                object, and timestamp.

        Returns:
            Multi-line string of formatted evidence, or 'No evidence
            available.' if the list is empty.
        """
        if not facts:
            return "No evidence available."

        lines: list[str] = []
        for idx, fact in enumerate(facts, start=1):
            line = EVIDENCE_FORMAT_TEMPLATE.format(
                index=idx,
                subject=fact.get("subject", ""),
                predicate=fact.get("predicate", ""),
                object=fact.get("object", ""),
                timestamp=fact.get("timestamp", ""),
            )
            lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Grounded reasoning
    # ------------------------------------------------------------------

    def reason(
        self, question: str, evidence_facts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Answer a question using grounded reasoning with temporal evidence.

        The model receives a system prompt instructing it to rely only on
        the evidence, plus a user message containing the question and
        formatted evidence block.

        Args:
            question: The natural-language temporal question.
            evidence_facts: List of retrieved temporal fact dicts.

        Returns:
            Dictionary with keys:
                answer: The model's answer text.
                raw_response: The full raw response string.
                model: The model name used.
                evidence_used: The list of fact dicts provided as evidence.
        """
        evidence_str = self._format_evidence(evidence_facts)
        user_message = USER_PROMPT_TEMPLATE.format(
            question=question, evidence=evidence_str
        )

        raw = self._call_llm(SYSTEM_PROMPT_GROUNDED, user_message)

        return {
            "answer": raw.strip(),
            "raw_response": raw,
            "model": self.model,
            "evidence_used": evidence_facts,
        }

    # ------------------------------------------------------------------
    # Ungrounded (baseline) reasoning
    # ------------------------------------------------------------------

    def reason_ungrounded(self, question: str) -> dict[str, Any]:
        """Answer a question without any evidence (baseline mode).

        The model answers purely from its parametric knowledge.

        Args:
            question: The natural-language temporal question.

        Returns:
            Dictionary with keys:
                answer: The model's answer text.
                raw_response: The full raw response string.
                model: The model name used.
        """
        raw = self._call_llm(SYSTEM_PROMPT_UNGROUNDED, question)

        return {
            "answer": raw.strip(),
            "raw_response": raw,
            "model": self.model,
        }

    # ------------------------------------------------------------------
    # Internal LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request to the LLM with retry logic.

        Makes up to ``_DEFAULT_MAX_RETRIES`` attempts, with exponential
        back-off on connection or timeout errors.

        Args:
            system_prompt: The system-level instruction.
            user_message: The user-level message.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_error: Exception | None = None
        for attempt in range(1, _DEFAULT_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return content or ""
            except (APIConnectionError, APITimeoutError, ConnectionError) as exc:
                last_error = exc
                wait = _DEFAULT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call attempt %d/%d failed (%s). "
                    "Retrying in %.1f seconds...",
                    attempt,
                    _DEFAULT_MAX_RETRIES,
                    type(exc).__name__,
                    wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.exception("Unexpected error during LLM call.")
                raise RuntimeError(
                    f"LLM call failed with unexpected error: {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM call failed after {_DEFAULT_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
