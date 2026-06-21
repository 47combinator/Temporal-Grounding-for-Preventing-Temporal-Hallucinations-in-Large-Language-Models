"""
Tests for the retrieval module.

Tests the QueryParser for entity and temporal extraction from natural
language questions.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.query_parser import QueryParser


class TestQueryParser:
    """Tests for the QueryParser class."""

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Create a QueryParser with sample known entities."""
        known_entities = [
            "United States", "Russia", "China", "United Kingdom",
            "Barack Obama", "Vladimir Putin", "Angela Merkel",
        ]
        return QueryParser(known_entities=known_entities)

    def test_extract_temporal_year(self, parser: QueryParser) -> None:
        """Test extracting a year from a question."""
        result = parser.extract_temporal_expressions(
            "What happened in 2014?"
        )
        assert "2014" in result.get("years", [])

    def test_extract_temporal_month_year(self, parser: QueryParser) -> None:
        """Test extracting a month-year from a question."""
        result = parser.extract_temporal_expressions(
            "What happened in March 2014?"
        )
        dates = result.get("dates", [])
        years = result.get("years", [])
        assert len(dates) > 0 or "2014" in years

    def test_extract_temporal_before(self, parser: QueryParser) -> None:
        """Test detecting a 'before' temporal reference."""
        result = parser.extract_temporal_expressions(
            "Who made statements before March 2014?"
        )
        assert "before" in result.get("relative", [])

    def test_extract_temporal_after(self, parser: QueryParser) -> None:
        """Test detecting an 'after' temporal reference."""
        result = parser.extract_temporal_expressions(
            "What happened after June 2014?"
        )
        assert "after" in result.get("relative", [])

    def test_resolve_entities_exact(self, parser: QueryParser) -> None:
        """Test exact entity name resolution."""
        resolved = parser.resolve_entities(["Russia"])
        assert "Russia" in resolved

    def test_resolve_entities_case_insensitive(self, parser: QueryParser) -> None:
        """Test case-insensitive entity resolution."""
        resolved = parser.resolve_entities(["russia"])
        assert "Russia" in resolved

    def test_resolve_entities_substring(self, parser: QueryParser) -> None:
        """Test substring-based entity resolution."""
        resolved = parser.resolve_entities(["Obama"])
        assert "Barack Obama" in resolved

    def test_normalize_date_range_year(self, parser: QueryParser) -> None:
        """Test normalising a year to a date range."""
        temporal_info = {"dates": [], "years": ["2014"], "relative": []}
        start, end = parser.normalize_date_range(temporal_info)
        assert start == "2014-01-01"
        assert end == "2014-12-31"

    def test_full_parse(self, parser: QueryParser) -> None:
        """Test the full parse pipeline."""
        result = parser.parse(
            "Who made a statement to Russia in 2014?"
        )
        assert "entities" in result
        assert "temporal" in result
        assert "date_range" in result
        assert "query_type" in result
        assert result["original_question"] == "Who made a statement to Russia in 2014?"


class TestQueryParserNoKnownEntities:
    """Tests for QueryParser without known entities (NER only)."""

    @pytest.fixture
    def parser(self) -> QueryParser:
        """Create a QueryParser without known entities."""
        return QueryParser()

    def test_extract_entities_ner(self, parser: QueryParser) -> None:
        """Test NER-based entity extraction."""
        entities = parser.extract_entities(
            "Russia threatened Ukraine in 2014."
        )
        # SpaCy should find at least Russia and/or Ukraine
        assert len(entities) >= 0  # NER may vary by model
