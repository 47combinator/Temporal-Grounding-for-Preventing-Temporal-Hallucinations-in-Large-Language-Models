"""
Query parser for extracting entities and temporal expressions from natural
language questions.

Uses spaCy NER for entity extraction and regex patterns for temporal
expression recognition. Supports resolution of extracted entity names
against a known entity list and normalisation of temporal expressions
into concrete date ranges.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime
from typing import Any

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns for temporal expression extraction
# ---------------------------------------------------------------------------

# Full dates: 2014-03-15, 03/15/2014, March 15, 2014, 15 March 2014
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)
_DATE_LONG_RE = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE
)
_DATE_LONG_REV_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE
)

# Month-year: March 2014, Mar 2014
_MONTH_YEAR_RE = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE
)

# Standalone 4-digit years
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

# Relative temporal references
_BEFORE_RE = re.compile(r"\bbefore\s+(.+?)(?:\?|$|,)", re.IGNORECASE)
_AFTER_RE = re.compile(r"\bafter\s+(.+?)(?:\?|$|,)", re.IGNORECASE)
_DURING_RE = re.compile(r"\bduring\s+(.+?)(?:\?|$|,)", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$|,)", re.IGNORECASE
)
_FIRST_RE = re.compile(r"\b(first|earliest)\b", re.IGNORECASE)
_LAST_RE = re.compile(r"\b(last|latest|most\s+recent)\b", re.IGNORECASE)

# Entities to keep from spaCy NER
_TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "NORP", "EVENT"}

# Month name to number mapping
_MONTH_TO_NUM: dict[str, int] = {}
for _i, _name in enumerate(calendar.month_name):
    if _name:
        _MONTH_TO_NUM[_name.lower()] = _i


class QueryParser:
    """Extract entities and temporal expressions from natural-language questions.

    Combines spaCy named-entity recognition with regex-based temporal
    expression extraction to produce a structured parse of a temporal question.
    """

    def __init__(self, known_entities: list[str] | None = None) -> None:
        """Initialise the query parser.

        Args:
            known_entities: Optional list of entity names for resolution.
                If provided, extracted entity strings will be matched
                against this list using exact, case-insensitive, and
                substring containment strategies.
        """
        self.known_entities: list[str] = known_entities or []
        self._known_lower: dict[str, str] = {
            e.lower(): e for e in self.known_entities
        }
        try:
            self.nlp: Language = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            raise

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, question: str) -> list[str]:
        """Extract named entities from a question using spaCy NER.

        Only PERSON, ORG, GPE, NORP, and EVENT entity labels are kept.

        Args:
            question: The natural-language question.

        Returns:
            De-duplicated list of entity text strings in order of appearance.
        """
        doc = self.nlp(question)
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            if ent.label_ in _TARGET_ENTITY_LABELS and ent.text not in seen:
                seen.add(ent.text)
                entities.append(ent.text)
        return entities

    # ------------------------------------------------------------------
    # Temporal expression extraction
    # ------------------------------------------------------------------

    def extract_temporal_expressions(self, question: str) -> dict[str, Any]:
        """Extract temporal expressions from a question using regex patterns.

        Args:
            question: The natural-language question.

        Returns:
            Dictionary with keys:
                dates:  List of concrete date strings (ISO format preferred).
                years:  List of 4-digit year strings.
                relative: List of relative temporal reference strings
                    (e.g. 'before 2015', 'between X and Y').
                temporal_type: One of 'point', 'range', 'before', 'after',
                    'ordering', or 'none'.
        """
        dates: list[str] = []
        years: list[str] = []
        relative: list[str] = []

        # ISO dates: 2014-03-15
        for match in _DATE_ISO_RE.finditer(question):
            dates.append(match.group(1))

        # Slash dates: 03/15/2014  ->  convert to ISO
        for match in _DATE_SLASH_RE.finditer(question):
            parts = match.group(1).split("/")
            if len(parts) == 3:
                month, day, year = parts
                dates.append(f"{year}-{month.zfill(2)}-{day.zfill(2)}")

        # Long form: March 15, 2014
        for match in _DATE_LONG_RE.finditer(question):
            month_name, day, year = match.group(1), match.group(2), match.group(3)
            month_num = _MONTH_TO_NUM.get(month_name.lower(), 1)
            dates.append(f"{year}-{month_num:02d}-{int(day):02d}")

        # Reversed long form: 15 March 2014
        for match in _DATE_LONG_REV_RE.finditer(question):
            day, month_name, year = match.group(1), match.group(2), match.group(3)
            month_num = _MONTH_TO_NUM.get(month_name.lower(), 1)
            dates.append(f"{year}-{month_num:02d}-{int(day):02d}")

        # Month-year: March 2014 (only if not already captured as a full date)
        for match in _MONTH_YEAR_RE.finditer(question):
            month_name, year = match.group(1), match.group(2)
            month_num = _MONTH_TO_NUM.get(month_name.lower(), 1)
            month_str = f"{year}-{month_num:02d}"
            # Avoid duplicate with already-captured full dates
            if not any(d.startswith(month_str) for d in dates):
                dates.append(month_str)

        # Years
        for match in _YEAR_RE.finditer(question):
            year_str = match.group(1)
            # Avoid duplicating years that are part of full dates
            if year_str not in years:
                years.append(year_str)

        # Relative expressions
        for match in _BEFORE_RE.finditer(question):
            relative.append(f"before {match.group(1).strip()}")

        for match in _AFTER_RE.finditer(question):
            relative.append(f"after {match.group(1).strip()}")

        for match in _DURING_RE.finditer(question):
            relative.append(f"during {match.group(1).strip()}")

        for match in _BETWEEN_RE.finditer(question):
            relative.append(
                f"between {match.group(1).strip()} and {match.group(2).strip()}"
            )

        if _FIRST_RE.search(question):
            relative.append("first")

        if _LAST_RE.search(question):
            relative.append("last")

        # Determine temporal type
        temporal_type = self._classify_temporal_type(dates, years, relative)

        return {
            "dates": dates,
            "years": years,
            "relative": relative,
            "temporal_type": temporal_type,
        }

    @staticmethod
    def _classify_temporal_type(
        dates: list[str], years: list[str], relative: list[str]
    ) -> str:
        """Classify the overall temporal type of the extracted expressions.

        Args:
            dates: Extracted date strings.
            years: Extracted year strings.
            relative: Extracted relative temporal references.

        Returns:
            One of 'point', 'range', 'before', 'after', 'ordering', or 'none'.
        """
        has_before = any(r.startswith("before") for r in relative)
        has_after = any(r.startswith("after") for r in relative)
        has_between = any(r.startswith("between") for r in relative)
        has_ordering = any(r in ("first", "last") for r in relative)

        if has_between:
            return "range"
        if has_before:
            return "before"
        if has_after:
            return "after"
        if has_ordering:
            return "ordering"

        total_temporal = len(dates) + len(years)
        if total_temporal == 0:
            return "none"
        if total_temporal == 1:
            return "point"
        return "range"

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def resolve_entities(self, extracted: list[str]) -> list[str]:
        """Match extracted entity names against the known entity list.

        Matching strategy (applied per extracted entity):
            1. Exact string match.
            2. Case-insensitive match.
            3. Substring containment (known entity contains the extracted
               string, or vice-versa, case-insensitive).

        Args:
            extracted: List of entity strings from NER.

        Returns:
            List of matched entity names from the known_entities list.
            Preserves order and avoids duplicates.
        """
        if not self.known_entities:
            return extracted

        resolved: list[str] = []
        seen: set[str] = set()

        for name in extracted:
            matched = self._match_single(name)
            if matched and matched not in seen:
                seen.add(matched)
                resolved.append(matched)

        return resolved

    def _match_single(self, name: str) -> str | None:
        """Attempt to resolve a single entity name against known entities.

        Args:
            name: The entity string to resolve.

        Returns:
            The matching known entity name, or None if no match is found.
        """
        # Strategy 1: exact match
        if name in self.known_entities:
            return name

        # Strategy 2: case-insensitive match
        name_lower = name.lower()
        if name_lower in self._known_lower:
            return self._known_lower[name_lower]

        # Strategy 3: substring containment (bidirectional)
        for known in self.known_entities:
            known_lower = known.lower()
            if name_lower in known_lower or known_lower in name_lower:
                return known

        return None

    # ------------------------------------------------------------------
    # Date normalisation
    # ------------------------------------------------------------------

    def normalize_date_range(
        self, temporal_info: dict[str, Any]
    ) -> tuple[str, str] | None:
        """Convert extracted temporal expressions into a (start_date, end_date) range.

        Conversion rules:
            - A 4-digit year '2014' becomes ('2014-01-01', '2014-12-31').
            - A month-year 'YYYY-MM' becomes ('YYYY-MM-01', 'YYYY-MM-{last_day}').
            - A full ISO date passes through as both start and end.
            - If two dates or years are present, the earlier is start, later is end.
            - 'between X and Y' in relative references is also handled.

        Args:
            temporal_info: The dict returned by extract_temporal_expressions.

        Returns:
            Tuple of (start_date, end_date) in ISO format, or None if no
            temporal information is available.
        """
        dates: list[str] = temporal_info.get("dates", [])
        years: list[str] = temporal_info.get("years", [])
        relative: list[str] = temporal_info.get("relative", [])

        all_ranges: list[tuple[str, str]] = []

        # Full or partial dates
        for d in dates:
            rng = self._single_date_to_range(d)
            if rng:
                all_ranges.append(rng)

        # Standalone years (only those not already covered by a date)
        covered_years = {d[:4] for d in dates if len(d) >= 4}
        for y in years:
            if y not in covered_years:
                all_ranges.append((f"{y}-01-01", f"{y}-12-31"))

        # Handle 'between X and Y' from relative
        for ref in relative:
            if ref.startswith("between"):
                between_match = re.match(
                    r"between\s+(.+?)\s+and\s+(.+)", ref, re.IGNORECASE
                )
                if between_match:
                    start_ref = between_match.group(1).strip()
                    end_ref = between_match.group(2).strip()
                    start_rng = self._try_parse_reference(start_ref)
                    end_rng = self._try_parse_reference(end_ref)
                    if start_rng and end_rng:
                        all_ranges.append((start_rng[0], end_rng[1]))

        if not all_ranges:
            return None

        # If multiple ranges, span from the earliest start to the latest end
        starts = sorted(r[0] for r in all_ranges)
        ends = sorted(r[1] for r in all_ranges)
        return (starts[0], ends[-1])

    @staticmethod
    def _single_date_to_range(date_str: str) -> tuple[str, str] | None:
        """Convert a single date string to a start/end range.

        Args:
            date_str: A date string in ISO format (YYYY, YYYY-MM, or YYYY-MM-DD).

        Returns:
            Tuple (start_date, end_date) or None if unparseable.
        """
        parts = date_str.split("-")
        if len(parts) == 3:
            # Full date
            return (date_str, date_str)
        if len(parts) == 2:
            # Month-year
            year, month = int(parts[0]), int(parts[1])
            last_day = calendar.monthrange(year, month)[1]
            return (f"{date_str}-01", f"{date_str}-{last_day:02d}")
        if len(parts) == 1 and len(date_str) == 4:
            # Year only
            return (f"{date_str}-01-01", f"{date_str}-12-31")
        return None

    @staticmethod
    def _try_parse_reference(ref: str) -> tuple[str, str] | None:
        """Attempt to parse a temporal reference string into a date range.

        Args:
            ref: A reference string that might be a year or date.

        Returns:
            Tuple (start, end) or None.
        """
        ref = ref.strip()
        # Try year
        if re.fullmatch(r"\d{4}", ref):
            return (f"{ref}-01-01", f"{ref}-12-31")
        # Try ISO date
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ref):
            return (ref, ref)
        # Try month-year with name
        month_match = re.match(
            rf"({_MONTH_NAMES})\s+(\d{{4}})", ref, re.IGNORECASE
        )
        if month_match:
            month_num = _MONTH_TO_NUM.get(month_match.group(1).lower(), 1)
            year = int(month_match.group(2))
            last_day = calendar.monthrange(year, month_num)[1]
            return (
                f"{year}-{month_num:02d}-01",
                f"{year}-{month_num:02d}-{last_day:02d}",
            )
        return None

    # ------------------------------------------------------------------
    # Full parse
    # ------------------------------------------------------------------

    def parse(self, question: str) -> dict[str, Any]:
        """Perform a complete parse of a temporal question.

        Steps:
            1. Extract named entities via spaCy NER.
            2. Extract temporal expressions via regex.
            3. Resolve entities against the known entity list.
            4. Normalise temporal expressions into a date range.
            5. Determine the overall query type.

        Args:
            question: The natural-language question.

        Returns:
            Dictionary with keys:
                entities: Resolved entity names (list[str]).
                temporal: Raw temporal extraction dict.
                date_range: Normalised (start_date, end_date) tuple or None.
                query_type: High-level query classification string.
                original_question: The input question.
        """
        entities = self.extract_entities(question)
        temporal = self.extract_temporal_expressions(question)
        resolved = self.resolve_entities(entities)
        date_range = self.normalize_date_range(temporal)

        query_type = self._determine_query_type(
            resolved, temporal, date_range
        )

        return {
            "entities": resolved,
            "temporal": temporal,
            "date_range": date_range,
            "query_type": query_type,
            "original_question": question,
        }

    @staticmethod
    def _determine_query_type(
        entities: list[str],
        temporal: dict[str, Any],
        date_range: tuple[str, str] | None,
    ) -> str:
        """Determine the high-level type of the parsed query.

        Args:
            entities: Resolved entity list.
            temporal: Temporal extraction dict.
            date_range: Normalised date range or None.

        Returns:
            One of: 'entity_at_time', 'entity_pair', 'events_before',
            'events_after', 'entity_all_events', 'predicate_search', or
            'general'.
        """
        temporal_type = temporal.get("temporal_type", "none")
        num_entities = len(entities)

        if num_entities >= 2 and date_range:
            return "entity_pair"
        if num_entities >= 2:
            return "entity_pair"
        if temporal_type == "before" and num_entities >= 1:
            return "events_before"
        if temporal_type == "after" and num_entities >= 1:
            return "events_after"
        if num_entities >= 1 and date_range:
            return "entity_at_time"
        if num_entities >= 1:
            return "entity_all_events"
        if date_range:
            return "predicate_search"
        return "general"
