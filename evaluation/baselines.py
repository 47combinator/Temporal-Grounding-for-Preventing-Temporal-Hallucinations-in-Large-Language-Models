"""
Baseline runners for comparative evaluation.

Implements four experimental conditions:
1. **Vanilla LLM** -- direct LLM prompting with no retrieval or verification.
2. **Standard RAG** -- retrieve facts by simple text search, no verification.
3. **Graph RAG** -- retrieve via graph traversal (1-hop and 2-hop), no verification.
4. **TKG-Verify** -- the full proposed pipeline with retrieval and verification.
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Run baseline experiments for comparative evaluation.

    Each method implements a different experimental condition using the
    same underlying LLM and (optionally) the same knowledge graph.

    Parameters
    ----------
    graph_store :
        A ``TemporalKGStore`` instance for Neo4j access.
    reasoner :
        An ``LLMReasoner`` for answer generation.
    """

    def __init__(self, graph_store: Any, reasoner: Any) -> None:
        """Initialise the baseline runner with graph store and LLM."""
        self.graph_store = graph_store
        self.reasoner = reasoner

    # ------------------------------------------------------------------
    # Baseline 1: Vanilla LLM
    # ------------------------------------------------------------------

    def run_vanilla_llm(
        self, questions: list[str]
    ) -> list[dict[str, Any]]:
        """Run the vanilla LLM baseline (no retrieval, no verification).

        The LLM receives each question directly with no context from
        the knowledge graph and no post-generation verification.

        Parameters
        ----------
        questions :
            List of natural-language temporal questions.

        Returns
        -------
        list[dict[str, Any]]
            One result dictionary per question with keys ``"question"``,
            ``"answer"``, and ``"claims"``.
        """
        results: list[dict[str, Any]] = []
        for question in tqdm(questions, desc="Vanilla LLM"):
            try:
                answer = self.reasoner.generate_answer(question)
                claims = self._extract_claims_safe(answer)
                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "claims": claims,
                    }
                )
            except Exception:
                logger.exception(
                    "Vanilla LLM failed for: %s", question
                )
                results.append(
                    {
                        "question": question,
                        "answer": "",
                        "claims": [],
                        "error": True,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Baseline 2: Standard RAG
    # ------------------------------------------------------------------

    def run_standard_rag(
        self, questions: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Run the standard RAG baseline (text-search retrieval, no verification).

        Facts are retrieved by simple entity-name matching in the graph.
        The LLM receives the question plus retrieved facts as context.

        Parameters
        ----------
        questions :
            List of natural-language temporal questions.
        top_k :
            Maximum number of facts to retrieve per question.

        Returns
        -------
        list[dict[str, Any]]
            One result dictionary per question with keys ``"question"``,
            ``"answer"``, ``"claims"``, and ``"retrieved_facts"``.
        """
        results: list[dict[str, Any]] = []
        for question in tqdm(questions, desc="Standard RAG"):
            try:
                facts = self._text_search_retrieve(question, top_k)
                answer = self.reasoner.generate_grounded_answer(
                    question=question, facts=facts
                )
                claims = self._extract_claims_safe(answer)
                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "claims": claims,
                        "retrieved_facts": [
                            self._serialise_fact(f) for f in facts
                        ],
                    }
                )
            except Exception:
                logger.exception(
                    "Standard RAG failed for: %s", question
                )
                results.append(
                    {
                        "question": question,
                        "answer": "",
                        "claims": [],
                        "retrieved_facts": [],
                        "error": True,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Baseline 3: Graph RAG
    # ------------------------------------------------------------------

    def run_graph_rag(
        self, questions: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Run the Graph RAG baseline (graph traversal retrieval, no verification).

        Facts are retrieved via 1-hop and 2-hop graph traversal from
        entities mentioned in the question. The LLM receives the
        question plus retrieved facts as context.

        Parameters
        ----------
        questions :
            List of natural-language temporal questions.
        top_k :
            Maximum number of facts to retrieve per question.

        Returns
        -------
        list[dict[str, Any]]
            One result dictionary per question with keys ``"question"``,
            ``"answer"``, ``"claims"``, and ``"retrieved_facts"``.
        """
        results: list[dict[str, Any]] = []
        for question in tqdm(questions, desc="Graph RAG"):
            try:
                facts = self._graph_traversal_retrieve(question, top_k)
                answer = self.reasoner.generate_grounded_answer(
                    question=question, facts=facts
                )
                claims = self._extract_claims_safe(answer)
                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "claims": claims,
                        "retrieved_facts": [
                            self._serialise_fact(f) for f in facts
                        ],
                    }
                )
            except Exception:
                logger.exception(
                    "Graph RAG failed for: %s", question
                )
                results.append(
                    {
                        "question": question,
                        "answer": "",
                        "claims": [],
                        "retrieved_facts": [],
                        "error": True,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Baseline 4: TKG-Verify (full pipeline)
    # ------------------------------------------------------------------

    def run_tkg_verify(
        self,
        pipeline: Any,
        questions: list[str],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run the full proposed TKG-Verify pipeline.

        This delegates to the ``TemporalVerificationPipeline.process_batch``
        method, which includes retrieval, verification, hallucination
        detection, and explanation generation.

        Parameters
        ----------
        pipeline :
            A ``TemporalVerificationPipeline`` instance.
        questions :
            List of natural-language temporal questions.
        top_k :
            Maximum number of facts to retrieve per question.

        Returns
        -------
        list[dict[str, Any]]
            Full pipeline results, one dictionary per question.
        """
        return pipeline.process_batch(questions, top_k=top_k)

    # ------------------------------------------------------------------
    # Run all baselines
    # ------------------------------------------------------------------

    def run_all(
        self,
        questions: list[str],
        pipeline: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run all four baselines and return results keyed by method name.

        Parameters
        ----------
        questions :
            List of natural-language temporal questions.
        pipeline :
            An optional ``TemporalVerificationPipeline`` instance,
            required for the TKG-Verify baseline.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            A dictionary with keys ``"vanilla_llm"``, ``"standard_rag"``,
            ``"graph_rag"``, and ``"tkg_verify"``, each mapping to a
            list of per-question result dictionaries.
        """
        logger.info("Running all baselines on %d questions.", len(questions))

        results: dict[str, list[dict[str, Any]]] = {}

        logger.info("--- Baseline 1: Vanilla LLM ---")
        results["vanilla_llm"] = self.run_vanilla_llm(questions)

        logger.info("--- Baseline 2: Standard RAG ---")
        results["standard_rag"] = self.run_standard_rag(questions)

        logger.info("--- Baseline 3: Graph RAG ---")
        results["graph_rag"] = self.run_graph_rag(questions)

        if pipeline is not None:
            logger.info("--- Baseline 4: TKG-Verify ---")
            results["tkg_verify"] = self.run_tkg_verify(
                pipeline, questions
            )
        else:
            logger.warning(
                "Skipping TKG-Verify baseline: no pipeline provided."
            )
            results["tkg_verify"] = []

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _text_search_retrieve(
        self, question: str, top_k: int
    ) -> list[Any]:
        """Retrieve facts by simple entity-name text search.

        Extracts entity-like tokens from the question and queries the
        graph for events where those tokens appear as entity names.

        Parameters
        ----------
        question :
            The natural-language question.
        top_k :
            Maximum number of facts to return.

        Returns
        -------
        list
            Retrieved temporal facts.
        """
        # Extract candidate entity tokens (capitalised words / phrases).
        tokens = self._extract_entity_tokens(question)
        all_facts: list[Any] = []
        for token in tokens:
            try:
                facts = self.graph_store.search_entities_by_name(
                    name=token, limit=top_k
                )
                all_facts.extend(facts)
            except Exception:
                logger.debug(
                    "Text search failed for token '%s'.", token
                )
        # Deduplicate by event_id and trim.
        seen: set[str] = set()
        unique: list[Any] = []
        for f in all_facts:
            eid = getattr(f, "event_id", id(f))
            if eid not in seen:
                seen.add(eid)
                unique.append(f)
        return unique[:top_k]

    def _graph_traversal_retrieve(
        self, question: str, top_k: int
    ) -> list[Any]:
        """Retrieve facts via 1-hop and 2-hop graph traversal.

        Starting from entities mentioned in the question, traverses
        outgoing and incoming edges up to two hops.

        Parameters
        ----------
        question :
            The natural-language question.
        top_k :
            Maximum number of facts to return.

        Returns
        -------
        list
            Retrieved temporal facts.
        """
        tokens = self._extract_entity_tokens(question)
        all_facts: list[Any] = []
        for token in tokens:
            try:
                # 1-hop neighbours.
                one_hop = self.graph_store.get_entity_events(
                    entity_name=token, limit=top_k
                )
                all_facts.extend(one_hop)

                # 2-hop: for each 1-hop result, get related events.
                for fact in one_hop[:3]:
                    related_entity = getattr(fact, "object", None)
                    if related_entity:
                        two_hop = self.graph_store.get_entity_events(
                            entity_name=related_entity, limit=top_k // 2
                        )
                        all_facts.extend(two_hop)
            except Exception:
                logger.debug(
                    "Graph traversal failed for token '%s'.", token
                )

        # Deduplicate and trim.
        seen: set[str] = set()
        unique: list[Any] = []
        for f in all_facts:
            eid = getattr(f, "event_id", id(f))
            if eid not in seen:
                seen.add(eid)
                unique.append(f)
        return unique[:top_k]

    @staticmethod
    def _extract_entity_tokens(question: str) -> list[str]:
        """Extract candidate entity tokens from a question.

        Uses a simple heuristic: split on whitespace and collect
        sequences of capitalised words as candidate entity mentions.
        This is intentionally simplistic for the baseline -- the full
        pipeline uses spaCy-based NER via ``QueryParser``.

        Parameters
        ----------
        question :
            The natural-language question.

        Returns
        -------
        list[str]
            Candidate entity tokens.
        """
        # Collect capitalised words (very simple NER heuristic).
        words = question.split()
        entities: list[str] = []
        current: list[str] = []

        for word in words:
            # Strip punctuation for checking but keep original.
            cleaned = word.strip("?.,!:;\"'()")
            if cleaned and cleaned[0].isupper():
                current.append(cleaned)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []
        if current:
            entities.append(" ".join(current))

        # Filter out very short tokens and common question words.
        stop_words = {
            "What", "When", "Where", "Who", "Which", "How",
            "Did", "Does", "Do", "Is", "Are", "Was", "Were",
            "The", "A", "An", "In", "On", "At", "By", "For",
        }
        filtered = [
            e for e in entities
            if e not in stop_words and len(e) > 1
        ]
        return filtered

    def _extract_claims_safe(self, answer: str) -> list[dict[str, Any]]:
        """Extract claims from an answer, returning dicts for serialisation.

        Falls back to an empty list on error.

        Parameters
        ----------
        answer :
            The LLM-generated answer text.

        Returns
        -------
        list[dict[str, Any]]
            Extracted claims as dictionaries.
        """
        try:
            from src.reasoning.claim_extractor import ClaimExtractor
            extractor = ClaimExtractor(
                base_url=self.reasoner.base_url,
                model=self.reasoner.model,
            )
            claims = extractor.extract(answer)
            from dataclasses import asdict
            return [asdict(c) for c in claims]
        except Exception:
            logger.debug("Claim extraction failed; returning empty list.")
            return []

    @staticmethod
    def _serialise_fact(fact: Any) -> dict[str, Any]:
        """Convert a fact object to a serialisable dictionary.

        Parameters
        ----------
        fact :
            A temporal fact (dataclass or object with attributes).

        Returns
        -------
        dict[str, Any]
            The fact as a plain dictionary.
        """
        try:
            from dataclasses import asdict
            return asdict(fact)
        except TypeError:
            return {
                "subject": getattr(fact, "subject", str(fact)),
                "predicate": getattr(fact, "predicate", ""),
                "object": getattr(fact, "object", ""),
                "timestamp": getattr(fact, "timestamp", ""),
            }
