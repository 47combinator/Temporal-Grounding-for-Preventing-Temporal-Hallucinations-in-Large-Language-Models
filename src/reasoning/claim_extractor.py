"""
Temporal claim extractor for the Temporal Grounding and Verification Framework.

Extracts structured temporal claims (subject, predicate, object, timestamp)
from LLM-generated text using either rule-based NLP (spaCy + regex) or a
secondary LLM call.  Extracted claims feed into the downstream verification
pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Span

from src.reasoning.prompt_templates import CLAIM_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# Regex for dates in generated text
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE
)

# Entity labels considered relevant for claim subjects and objects
_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "NORP", "EVENT", "FAC", "LOC"}


class ClaimExtractor:
    """Extract temporal claims from LLM-generated text.

    Supports two extraction methods:
        - **rule_based**: Uses spaCy NER and regex patterns.  Fast and
          deterministic but may miss complex phrasings.
        - **llm**: Uses a secondary LLM call with a structured extraction
          prompt.  More flexible but slower and requires an LLM connection.
    """

    def __init__(
        self,
        llm_reasoner: Any | None = None,
        nlp: Language | None = None,
    ) -> None:
        """Initialise the claim extractor.

        Args:
            llm_reasoner: Optional ``LLMReasoner`` instance for LLM-based
                extraction.  If not provided, the ``extract_claims_llm``
                method will be unavailable and the ``extract`` method
                will not fall back to LLM extraction.
            nlp: Optional pre-loaded spaCy Language model.  If not provided,
                ``en_core_web_sm`` is loaded automatically.
        """
        self.llm_reasoner = llm_reasoner

        if nlp is not None:
            self.nlp: Language = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                raise

    # ------------------------------------------------------------------
    # Rule-based extraction
    # ------------------------------------------------------------------

    def extract_claims_rule_based(self, text: str) -> list[dict[str, Any]]:
        """Extract temporal claims using spaCy NER and regex patterns.

        Strategy:
            1. Split text into sentences using spaCy.
            2. For each sentence, extract entities and temporal expressions.
            3. Construct (subject, predicate, object, timestamp) tuples by
               pairing the first entity (subject) with subsequent entities
               (object) and associating found dates.
            4. The predicate is approximated as the verb phrase or text
               between subject and object spans.

        Args:
            text: The LLM-generated text to extract claims from.

        Returns:
            List of claim dicts, each with keys: subject, predicate, object,
            timestamp, source_sentence, extraction_method.
        """
        doc = self.nlp(text)
        claims: list[dict[str, Any]] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Extract entities in this sentence
            entities = [
                ent for ent in sent.ents if ent.label_ in _ENTITY_LABELS
            ]
            # Extract dates from the sentence text
            dates = self._extract_dates_from_text(sent_text)

            if not entities or not dates:
                continue

            # Build claims from entity pairs
            sentence_claims = self._build_claims_from_sentence(
                sent, entities, dates, sent_text
            )
            claims.extend(sentence_claims)

        return claims

    def _build_claims_from_sentence(
        self,
        sent: Span,
        entities: list[Span],
        dates: list[str],
        sent_text: str,
    ) -> list[dict[str, Any]]:
        """Build claim dicts from entities and dates found in a sentence.

        When there are two or more entities, the first is treated as the
        subject and each subsequent entity becomes an object.  When only
        one entity is found, it becomes both subject and object placeholder.

        Args:
            sent: The spaCy Span representing the sentence.
            entities: Named entities found in the sentence.
            dates: Date strings extracted from the sentence.
            sent_text: The raw sentence text.

        Returns:
            List of claim dicts.
        """
        claims: list[dict[str, Any]] = []
        timestamp = dates[0]  # Use the first date found

        if len(entities) >= 2:
            subject_ent = entities[0]
            for obj_ent in entities[1:]:
                predicate = self._extract_predicate(sent, subject_ent, obj_ent)
                claims.append({
                    "subject": subject_ent.text,
                    "predicate": predicate,
                    "object": obj_ent.text,
                    "timestamp": timestamp,
                    "source_sentence": sent_text,
                    "extraction_method": "rule_based",
                })
        else:
            # Single entity -- extract a predicate from the verb phrase
            predicate = self._extract_verb_phrase(sent)
            claims.append({
                "subject": entities[0].text,
                "predicate": predicate,
                "object": "",
                "timestamp": timestamp,
                "source_sentence": sent_text,
                "extraction_method": "rule_based",
            })

        return claims

    @staticmethod
    def _extract_predicate(
        sent: Span, subject_ent: Span, object_ent: Span
    ) -> str:
        """Extract the predicate text between a subject and object entity.

        Falls back to the root verb of the sentence if the span between
        entities is empty or uninformative.

        Args:
            sent: The sentence span.
            subject_ent: The subject entity span.
            object_ent: The object entity span.

        Returns:
            Predicate string.
        """
        # Get text between the two entities
        start = subject_ent.end
        end = object_ent.start
        if start < end:
            between_tokens = sent.doc[start:end]
            between_text = between_tokens.text.strip()
            # Clean up leading/trailing punctuation
            between_text = between_text.strip(",.;: ")
            if between_text:
                return between_text

        # Fallback: use the root verb
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_

        return "related_to"

    @staticmethod
    def _extract_verb_phrase(sent: Span) -> str:
        """Extract the main verb or verb phrase from a sentence.

        Args:
            sent: The sentence span.

        Returns:
            Verb phrase string, or 'related_to' as default.
        """
        verbs: list[str] = []
        for token in sent:
            if token.pos_ == "VERB":
                verbs.append(token.lemma_)
        return " ".join(verbs) if verbs else "related_to"

    @staticmethod
    def _extract_dates_from_text(text: str) -> list[str]:
        """Extract date strings from raw text using regex patterns.

        Checks for ISO dates first, then month-day-year patterns, then
        standalone years.

        Args:
            text: The text to search for dates.

        Returns:
            List of date strings in ISO-like format.
        """
        dates: list[str] = []

        # ISO dates: 2014-03-15
        for match in _DATE_ISO_RE.finditer(text):
            dates.append(match.group(1))

        # Month Day, Year: March 15, 2014
        import calendar
        month_to_num = {
            name.lower(): i
            for i, name in enumerate(calendar.month_name) if name
        }
        for match in _MONTH_DAY_YEAR_RE.finditer(text):
            month_name = match.group(1)
            day = match.group(2)
            year = match.group(3)
            month_num = month_to_num.get(month_name.lower(), 1)
            dates.append(f"{year}-{month_num:02d}-{int(day):02d}")

        # Standalone years (only if no more specific dates found)
        if not dates:
            for match in _YEAR_RE.finditer(text):
                dates.append(match.group(1))

        return dates

    # ------------------------------------------------------------------
    # LLM-based extraction
    # ------------------------------------------------------------------

    def extract_claims_llm(self, text: str) -> list[dict[str, Any]]:
        """Extract temporal claims using a secondary LLM call.

        Sends the text to the LLM with ``CLAIM_EXTRACTION_PROMPT`` and
        parses the returned JSON array.

        Args:
            text: The LLM-generated text to extract claims from.

        Returns:
            List of claim dicts, each with keys: subject, predicate, object,
            timestamp, source_sentence, extraction_method.

        Raises:
            RuntimeError: If no LLM reasoner is configured.
        """
        if self.llm_reasoner is None:
            raise RuntimeError(
                "LLM-based extraction requires an LLMReasoner instance. "
                "Provide one via the llm_reasoner parameter."
            )

        try:
            raw_response = self.llm_reasoner._call_llm(
                CLAIM_EXTRACTION_PROMPT, text
            )
        except RuntimeError:
            logger.exception("LLM call for claim extraction failed.")
            return []

        return self._parse_llm_claims(raw_response, text)

    @staticmethod
    def _parse_llm_claims(
        raw_response: str, source_text: str
    ) -> list[dict[str, Any]]:
        """Parse the JSON output from an LLM claim extraction call.

        Handles common issues like markdown code fences and malformed JSON
        gracefully.

        Args:
            raw_response: Raw LLM response string (expected JSON array).
            source_text: The original text that was analysed (for context).

        Returns:
            List of claim dicts with standardised keys.
        """
        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            lines = [
                line for line in lines
                if not line.strip().startswith("```")
            ]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM claim extraction output as JSON. "
                "Raw response: %s",
                raw_response[:200],
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "LLM claim extraction returned non-list JSON: %s",
                type(parsed).__name__,
            )
            return []

        claims: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            claim: dict[str, Any] = {
                "subject": str(item.get("subject", "")),
                "predicate": str(item.get("predicate", "")),
                "object": str(item.get("object", "")),
                "timestamp": str(item.get("timestamp", "")),
                "source_sentence": source_text[:200],
                "extraction_method": "llm",
            }
            # Only include claims with at least subject and predicate
            if claim["subject"] and claim["predicate"]:
                claims.append(claim)

        return claims

    # ------------------------------------------------------------------
    # Unified extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self, text: str, method: str = "rule_based"
    ) -> list[dict[str, Any]]:
        """Extract temporal claims from text using the specified method.

        Falls back from rule_based to llm if rule_based finds no claims
        and an LLM reasoner is available.

        Args:
            text: The text to extract claims from.
            method: Extraction method -- 'rule_based' or 'llm'.

        Returns:
            List of claim dicts.
        """
        if method == "llm":
            if self.llm_reasoner is None:
                logger.warning(
                    "LLM extraction requested but no LLMReasoner available. "
                    "Falling back to rule_based extraction."
                )
                return self.extract_claims_rule_based(text)
            return self.extract_claims_llm(text)

        # Default: rule_based with LLM fallback
        claims = self.extract_claims_rule_based(text)

        if not claims and self.llm_reasoner is not None:
            logger.info(
                "Rule-based extraction found no claims; "
                "falling back to LLM extraction."
            )
            claims = self.extract_claims_llm(text)

        return claims
