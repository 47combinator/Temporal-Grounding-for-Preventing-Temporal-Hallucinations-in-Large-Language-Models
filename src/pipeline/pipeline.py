"""
End-to-end Temporal Verification Pipeline.

Orchestrates the full flow from question input to verified, annotated answer:
retrieval of temporal facts, grounded answer generation, claim extraction,
per-claim verification, hallucination detection, and explanation generation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from tqdm import tqdm

from src.verification.verification_result import (
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class TemporalVerificationPipeline:
    """End-to-end pipeline for temporally grounded question answering.

    The pipeline chains together seven components:

    1. **Graph store** -- retrieves raw temporal facts from Neo4j.
    2. **Retriever** -- selects the most relevant facts for a query.
    3. **Reasoner** -- generates an answer grounded in retrieved facts.
    4. **Claim extractor** -- pulls temporal claims from generated text.
    5. **Verifier** -- checks each claim against the knowledge graph.
    6. **Detector** -- classifies the answer as hallucinated or not.
    7. **Explainer** -- produces human-readable explanations.

    Parameters
    ----------
    graph_store :
        A ``TemporalKGStore`` instance for Neo4j access.
    retriever :
        A ``TemporalRetriever`` for fact retrieval.
    reasoner :
        An ``LLMReasoner`` for grounded answer generation.
    claim_extractor :
        A ``ClaimExtractor`` for pulling claims from text.
    verifier :
        A ``TemporalVerifier`` for claim verification.
    detector :
        A ``HallucinationDetector`` for hallucination classification.
    explainer :
        An ``ExplanationGenerator`` for producing explanations.
    """

    def __init__(
        self,
        graph_store: Any,
        retriever: Any,
        reasoner: Any,
        claim_extractor: Any,
        verifier: Any,
        detector: Any,
        explainer: Any,
    ) -> None:
        """Initialise the pipeline with all component instances."""
        self.graph_store = graph_store
        self.retriever = retriever
        self.reasoner = reasoner
        self.claim_extractor = claim_extractor
        self.verifier = verifier
        self.detector = detector
        self.explainer = explainer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, question: str, top_k: int = 10) -> dict[str, Any]:
        """Run the full verification pipeline on a single question.

        Parameters
        ----------
        question :
            The natural-language temporal question.
        top_k :
            Maximum number of temporal facts to retrieve.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - ``question``: the original question text
            - ``retrieved_facts``: list of temporal facts from the KG
            - ``answer``: the raw LLM-generated answer
            - ``claims``: list of extracted temporal claims
            - ``verification_results``: per-claim verification outcomes
            - ``hallucination_report``: classification of the answer
            - ``explanations``: human-readable explanations per claim
            - ``tcs``: temporal consistency score in ``[0.0, 1.0]``
            - ``final_answer``: answer with verification annotations
        """
        logger.info("Processing question: %s", question)

        # Step 1: Retrieve temporal facts from the knowledge graph.
        retrieved_facts = self._retrieve_facts(question, top_k)
        logger.info("Retrieved %d temporal facts.", len(retrieved_facts))

        # Step 2: Generate a grounded answer using the LLM.
        answer = self._generate_answer(question, retrieved_facts)
        logger.info("Generated answer (%d chars).", len(answer))

        # Step 3: Extract temporal claims from the generated answer.
        claims = self._extract_claims(answer)
        logger.info("Extracted %d temporal claims.", len(claims))

        # Step 4: Verify each claim against the knowledge graph.
        verification_results = self._verify_claims(claims)
        logger.info("Verified %d claims.", len(verification_results))

        # Step 5: Detect hallucinations based on verification outcomes.
        hallucination_report = self._detect_hallucinations(
            verification_results
        )
        logger.info(
            "Hallucination report: is_hallucinated=%s",
            hallucination_report.get("is_hallucinated", "unknown"),
        )

        # Step 6: Generate explanations for each verification result.
        explanations = self._generate_explanations(verification_results)

        # Step 7: Compute the temporal consistency score (TCS).
        tcs = self._compute_tcs(verification_results)
        logger.info("Temporal consistency score: %.4f", tcs)

        # Step 8: Annotate the answer with verification information.
        final_answer = self._annotate_answer(
            answer, verification_results, hallucination_report
        )

        return {
            "question": question,
            "retrieved_facts": [
                self._fact_to_dict(f) for f in retrieved_facts
            ],
            "answer": answer,
            "claims": [self._claim_to_dict(c) for c in claims],
            "verification_results": [
                self._vr_to_dict(vr) for vr in verification_results
            ],
            "hallucination_report": hallucination_report,
            "explanations": explanations,
            "tcs": tcs,
            "final_answer": final_answer,
        }

    def process_batch(
        self, questions: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Process multiple questions with a progress bar.

        Parameters
        ----------
        questions :
            List of natural-language temporal questions.
        top_k :
            Maximum number of temporal facts to retrieve per question.

        Returns
        -------
        list[dict[str, Any]]
            A list of result dictionaries, one per question.
        """
        results: list[dict[str, Any]] = []
        for question in tqdm(questions, desc="Processing questions"):
            try:
                result = self.process(question, top_k=top_k)
                results.append(result)
            except Exception:
                logger.exception(
                    "Failed to process question: %s", question
                )
                results.append(
                    {
                        "question": question,
                        "error": True,
                        "retrieved_facts": [],
                        "answer": "",
                        "claims": [],
                        "verification_results": [],
                        "hallucination_report": {},
                        "explanations": [],
                        "tcs": 0.0,
                        "final_answer": "",
                    }
                )
        return results

    @classmethod
    def from_config(cls) -> "TemporalVerificationPipeline":
        """Create a fully initialised pipeline from configuration settings.

        Instantiates all seven components using values from
        ``config.settings`` and returns a ready-to-use pipeline.

        Returns
        -------
        TemporalVerificationPipeline
            A fully wired pipeline instance.
        """
        # Import components here to avoid circular imports and to keep
        # the class importable even when not all components are installed.
        from config import settings
        from src.knowledge_graph.graph_store import TemporalKGStore
        from src.retrieval.query_parser import QueryParser
        from src.retrieval.temporal_retriever import TemporalRetriever
        from src.reasoning.llm_reasoner import LLMReasoner
        from src.reasoning.claim_extractor import ClaimExtractor
        from src.verification.temporal_verifier import TemporalVerifier
        from src.hallucination.detector import HallucinationDetector
        from src.explanation.explanation_generator import ExplanationGenerator

        logger.info("Building pipeline from configuration...")

        # 1. Knowledge graph store.
        graph_store = TemporalKGStore(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
        )

        # 2. Query parser -- load known entity names from the graph so
        #    the parser can perform entity linking.
        entity_names = graph_store.get_all_entity_names()
        query_parser = QueryParser(entity_names=entity_names)

        # 3. Temporal retriever.
        retriever = TemporalRetriever(
            graph_store=graph_store,
            query_parser=query_parser,
            top_k=settings.RETRIEVAL_TOP_K,
        )

        # 4. LLM reasoner.
        reasoner = LLMReasoner(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

        # 5. Claim extractor.
        claim_extractor = ClaimExtractor(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
        )

        # 6. Temporal verifier.
        verifier = TemporalVerifier(
            graph_store=graph_store,
            timestamp_tolerance_days=settings.TIMESTAMP_TOLERANCE_DAYS,
            partial_support_threshold=settings.PARTIAL_SUPPORT_THRESHOLD,
        )

        # 7. Hallucination detector.
        detector = HallucinationDetector()

        # 8. Explanation generator.
        explainer = ExplanationGenerator(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
        )

        logger.info("Pipeline built successfully.")

        return cls(
            graph_store=graph_store,
            retriever=retriever,
            reasoner=reasoner,
            claim_extractor=claim_extractor,
            verifier=verifier,
            detector=detector,
            explainer=explainer,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_facts(self, question: str, top_k: int) -> list[Any]:
        """Retrieve temporal facts relevant to the question.

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
        return self.retriever.retrieve(question, top_k=top_k)

    def _generate_answer(
        self, question: str, facts: list[Any]
    ) -> str:
        """Generate an answer grounded in the retrieved facts.

        Parameters
        ----------
        question :
            The natural-language question.
        facts :
            Temporal facts to use as context.

        Returns
        -------
        str
            The generated answer text.
        """
        return self.reasoner.generate_grounded_answer(
            question=question, facts=facts
        )

    def _extract_claims(self, answer: str) -> list[Any]:
        """Extract temporal claims from generated text.

        Parameters
        ----------
        answer :
            The LLM-generated answer text.

        Returns
        -------
        list
            Extracted ``TemporalClaim`` objects.
        """
        return self.claim_extractor.extract(answer)

    def _verify_claims(
        self, claims: list[Any]
    ) -> list[VerificationResult]:
        """Verify each temporal claim against the knowledge graph.

        Parameters
        ----------
        claims :
            List of ``TemporalClaim`` objects to verify.

        Returns
        -------
        list[VerificationResult]
            One verification result per claim.
        """
        results: list[VerificationResult] = []
        for claim in claims:
            try:
                result = self.verifier.verify(claim)
                results.append(result)
            except Exception:
                logger.exception(
                    "Verification failed for claim: %s", claim
                )
                results.append(
                    VerificationResult(
                        claim=claim,
                        status=VerificationStatus.CANNOT_VERIFY,
                        confidence=0.0,
                        explanation="Verification failed due to an error.",
                    )
                )
        return results

    def _detect_hallucinations(
        self, verification_results: list[VerificationResult]
    ) -> dict[str, Any]:
        """Classify the answer based on verification results.

        Parameters
        ----------
        verification_results :
            Per-claim verification outcomes.

        Returns
        -------
        dict[str, Any]
            A hallucination report with keys ``is_hallucinated``,
            ``hallucinated_claims``, ``summary``, etc.
        """
        return self.detector.detect(verification_results)

    def _generate_explanations(
        self, verification_results: list[VerificationResult]
    ) -> list[str]:
        """Generate human-readable explanations for each result.

        Parameters
        ----------
        verification_results :
            Per-claim verification outcomes.

        Returns
        -------
        list[str]
            One explanation string per verification result.
        """
        explanations: list[str] = []
        for vr in verification_results:
            try:
                explanation = self.explainer.explain(vr)
                explanations.append(explanation)
            except Exception:
                logger.exception("Explanation generation failed.")
                explanations.append(
                    f"Claim status: {vr.status.value}. "
                    f"Confidence: {vr.confidence:.2f}."
                )
        return explanations

    def _compute_tcs(
        self, verification_results: list[VerificationResult]
    ) -> float:
        """Compute the Temporal Consistency Score (TCS).

        The TCS is defined as the fraction of verified claims that are
        temporally consistent (SUPPORTED or PARTIALLY_SUPPORTED).

        Parameters
        ----------
        verification_results :
            Per-claim verification outcomes.

        Returns
        -------
        float
            The TCS in ``[0.0, 1.0]``, or ``1.0`` if no claims exist.
        """
        if not verification_results:
            return 1.0

        consistent_statuses = {
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
        }
        consistent_count = sum(
            1 for vr in verification_results
            if vr.status in consistent_statuses
        )
        return consistent_count / len(verification_results)

    def _annotate_answer(
        self,
        answer: str,
        verification_results: list[VerificationResult],
        hallucination_report: dict[str, Any],
    ) -> str:
        """Annotate the answer with verification metadata.

        Appends a structured verification summary to the original answer
        so that downstream consumers can see which claims were verified,
        which failed, and the overall consistency score.

        Parameters
        ----------
        answer :
            The raw LLM-generated answer.
        verification_results :
            Per-claim verification outcomes.
        hallucination_report :
            The hallucination classification report.

        Returns
        -------
        str
            The answer with verification annotations appended.
        """
        if not verification_results:
            return answer

        annotations: list[str] = ["\n\n--- Verification Summary ---"]

        for i, vr in enumerate(verification_results, 1):
            status_label = vr.status.value.replace("_", " ").title()
            annotations.append(
                f"[Claim {i}] {status_label} "
                f"(confidence: {vr.confidence:.2f})"
            )

        is_hallucinated = hallucination_report.get(
            "is_hallucinated", False
        )
        tcs = self._compute_tcs(verification_results)
        annotations.append(f"\nTemporal Consistency Score: {tcs:.4f}")
        if is_hallucinated:
            annotations.append(
                "WARNING: Potential temporal hallucination detected."
            )

        return answer + "\n".join(annotations)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_to_dict(fact: Any) -> dict[str, Any]:
        """Convert a temporal fact to a plain dictionary.

        Parameters
        ----------
        fact :
            A ``TemporalFact`` dataclass instance.

        Returns
        -------
        dict[str, Any]
            The fact as a JSON-serialisable dictionary.
        """
        try:
            return asdict(fact)
        except TypeError:
            # Fallback for non-dataclass fact representations.
            return {
                "subject": getattr(fact, "subject", str(fact)),
                "predicate": getattr(fact, "predicate", ""),
                "object": getattr(fact, "object", ""),
                "timestamp": getattr(fact, "timestamp", ""),
            }

    @staticmethod
    def _claim_to_dict(claim: Any) -> dict[str, Any]:
        """Convert a temporal claim to a plain dictionary.

        Parameters
        ----------
        claim :
            A ``TemporalClaim`` dataclass instance.

        Returns
        -------
        dict[str, Any]
            The claim as a JSON-serialisable dictionary.
        """
        try:
            return asdict(claim)
        except TypeError:
            return {
                "subject": getattr(claim, "subject", str(claim)),
                "predicate": getattr(claim, "predicate", ""),
                "object": getattr(claim, "object", ""),
                "timestamp": getattr(claim, "timestamp", ""),
            }

    @staticmethod
    def _vr_to_dict(vr: VerificationResult) -> dict[str, Any]:
        """Convert a verification result to a plain dictionary.

        Parameters
        ----------
        vr :
            A ``VerificationResult`` dataclass instance.

        Returns
        -------
        dict[str, Any]
            The result as a JSON-serialisable dictionary, with nested
            enums converted to their string values.
        """
        try:
            d = asdict(vr)
            # The VerificationStatus enum does not serialise cleanly
            # via asdict -- replace with its string value.
            d["status"] = vr.status.value
            return d
        except TypeError:
            return {
                "status": getattr(vr, "status", VerificationStatus.CANNOT_VERIFY).value,
                "confidence": getattr(vr, "confidence", 0.0),
                "explanation": getattr(vr, "explanation", ""),
            }
