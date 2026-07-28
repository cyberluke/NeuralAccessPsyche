"""
Document Innovation Workflow — P2 fix from docs/nram-fix-spec-v1.md.

This workflow transforms source documents into breakthrough innovation outputs
through a multi-stage pipeline that prevents copy-paste paraphrasing.

Pipeline stages:
1. Evidence Extractor — Extract facts, citations, capabilities from source
2. Fact Ledger — Store structured facts WITHOUT original prose
3. Contrarian Deconstructor — Challenge assumptions, reveal blind spots
4. 6 Expert Personas — Parallel analysis from different perspectives:
   - Future Anthropologist
   - Interface Radical
   - Product Minimalist
   - Economic Architect
   - Adversarial CTO
   - Science-fiction Prototyper
5. Concept Tournament — Select best concepts from expert analyses
6. Visionary Composer — Synthesize using ONLY facts/citations/capabilities/contradictions/concepts
   (NOT original paragraphs — this is the critical anti-copy mechanism)
7. Anti-copy Gate — Reject output if too similar to source document

The Composer receives a structured brief, not the original text. This forces
genuine synthesis rather than paraphrasing.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.agentic.contracts import (
    EvidenceItem,
    EvidenceSourceType,
    InnovationWorkflowState,
    WorkflowEvent,
    WorkflowStatus,
)
from core.agentic.evidence_ledger import EvidenceLedger
from core.agentic.personas.base import BasePersona

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Innovation Workflow schemas
# ---------------------------------------------------------------------------

class DocumentStage(str, Enum):
    """Stages in the Document Innovation Workflow."""
    CREATED = "created"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    FACT_LEDGER = "fact_ledger"
    CONTRARIAN_DECONSTRUCTION = "contrarian_deconstruction"
    EXPERT_ANALYSIS = "expert_analysis"
    CONCEPT_TOURNAMENT = "concept_tournament"
    VISIONARY_COMPOSITION = "visionary_composition"
    ANTI_COPY_GATE = "anti_copy_gate"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED_SIMILAR = "rejected_similar"


class ExtractedFact(BaseModel):
    """A fact extracted from the source document."""
    id: str
    claim: str
    citation: Optional[str] = None
    source_section: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    fact_type: str = "claim"  # claim, capability, contradiction, concept


class ContrarianChallenge(BaseModel):
    """A challenge to an assumption from the source document."""
    id: str
    assumption_challenged: str
    why_it_may_be_wrong: str
    alternative_frame: str
    evidence_required: List[str] = Field(default_factory=list)


class ExpertAnalysis(BaseModel):
    """Analysis from one of the 6 expert personas."""
    id: str
    expert_role: str
    key_insights: List[str] = Field(default_factory=list)
    novel_concepts: List[str] = Field(default_factory=list)
    contradictions_found: List[str] = Field(default_factory=list)
    recommended_directions: List[str] = Field(default_factory=list)


class ConceptTournamentResult(BaseModel):
    """Result of the concept tournament — selected winning concepts."""
    id: str
    winning_concepts: List[str] = Field(default_factory=list)
    rejected_concepts: List[str] = Field(default_factory=list)
    synthesis_direction: str = ""


class ComposerBrief(BaseModel):
    """
    The brief given to the Visionary Composer.
    
    CRITICAL: This contains ONLY structured data, NOT original paragraphs.
    This is the anti-copy mechanism — the Composer cannot paraphrase what it
    never sees.
    """
    facts: List[ExtractedFact] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    novel_concepts: List[str] = Field(default_factory=list)
    expert_insights: List[str] = Field(default_factory=list)
    synthesis_direction: str = ""


class DocumentInnovationOutput(BaseModel):
    """Final output from the Document Innovation Workflow."""
    workflow_id: str
    source_document_hash: str
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    contrarian_challenges: List[ContrarianChallenge] = Field(default_factory=list)
    expert_analyses: List[ExpertAnalysis] = Field(default_factory=list)
    tournament_result: Optional[ConceptTournamentResult] = None
    final_output: str = ""
    similarity_score: float = 0.0  # 0.0 = completely original, 1.0 = identical to source
    status: DocumentStage = DocumentStage.CREATED


# ---------------------------------------------------------------------------
# Expert Persona definitions
# ---------------------------------------------------------------------------

EXPERT_ROLES = [
    {
        "role": "Future Anthropologist",
        "profile": "threshold",
        "temperature": 0.85,
        "focus": "Analyze the document as an artifact from 2050. What does it reveal about the assumptions, blind spots, and cultural constraints of its era? What would a future historian find naive or prescient?",
    },
    {
        "role": "Interface Radical",
        "profile": "psychedelic",
        "temperature": 0.95,
        "focus": "Reimagine the core concept as a radically different interface or interaction model. What if the entire paradigm were inverted? What new human-computer relationship does this suggest?",
    },
    {
        "role": "Product Minimalist",
        "profile": "normal",
        "temperature": 0.4,
        "focus": "Strip the concept to its absolute essence. What is the ONE thing this must do brilliantly? What can be removed without losing the core value? Apply ruthless product discipline.",
    },
    {
        "role": "Economic Architect",
        "profile": "threshold",
        "temperature": 0.7,
        "focus": "Design the economic model around this concept. How does it create value? Who captures that value? What are the network effects, moats, and potential disruptions to existing markets?",
    },
    {
        "role": "Adversarial CTO",
        "profile": "normal",
        "temperature": 0.3,
        "focus": "Attack this concept technically. What are the hidden complexities, scaling bottlenecks, security risks, and technical debt traps? What would make this fail in production?",
    },
    {
        "role": "Science-fiction Prototyper",
        "profile": "psychedelic",
        "temperature": 1.0,
        "focus": "Write a prototype of this concept as if it already exists in 2035. Describe the user experience, the technology stack, the societal impact. Make it concrete and vivid.",
    },
]


# ---------------------------------------------------------------------------
# Document Innovation Workflow orchestrator
# ---------------------------------------------------------------------------

class DocumentInnovationWorkflow:
    """
    Orchestrates the Document Innovation Workflow.
    
    This workflow prevents copy-paste paraphrasing by:
    1. Extracting structured facts (not prose)
    2. Passing only facts/citations/capabilities to the Composer
    3. Rejecting output that is too similar to the source
    """

    def __init__(
        self,
        nram_api_base: str = "http://localhost:8000/v1",
        similarity_threshold: float = 0.35,  # Reject if >35% similar to source
    ) -> None:
        self._nram_api_base = nram_api_base
        self._similarity_threshold = similarity_threshold
        self._evidence_ledger = EvidenceLedger()

    async def run(
        self,
        source_document: str,
        user_goal: str,
        workflow_id: Optional[str] = None,
    ) -> DocumentInnovationOutput:
        """
        Run the full Document Innovation Workflow.
        
        Args:
            source_document: The original document text
            user_goal: What the user wants to achieve with this document
            workflow_id: Optional workflow ID (generated if not provided)
        
        Returns:
            DocumentInnovationOutput with final synthesized document
        
        Raises:
            ValueError: If output is too similar to source (anti-copy gate)
        """
        workflow_id = workflow_id or f"doc-innov-{uuid.uuid4().hex[:12]}"
        source_hash = hashlib.sha256(source_document.encode()).hexdigest()[:16]

        output = DocumentInnovationOutput(
            workflow_id=workflow_id,
            source_document_hash=source_hash,
        )

        try:
            # Stage 1: Evidence Extraction
            logger.info(f"[{workflow_id}] Stage 1: Evidence Extraction")
            output.extracted_facts = await self._extract_evidence(source_document, user_goal)

            # Stage 2: Fact Ledger (already stored in extracted_facts)
            logger.info(f"[{workflow_id}] Stage 2: Fact Ledger — {len(output.extracted_facts)} facts stored")

            # Stage 3: Contrarian Deconstruction
            logger.info(f"[{workflow_id}] Stage 3: Contrarian Deconstruction")
            output.contrarian_challenges = await self._deconstruct_assumptions(
                output.extracted_facts, user_goal
            )

            # Stage 4: Expert Analysis (6 personas in parallel)
            logger.info(f"[{workflow_id}] Stage 4: Expert Analysis (6 personas)")
            output.expert_analyses = await self._run_expert_analysis(
                output.extracted_facts,
                output.contrarian_challenges,
                user_goal,
            )

            # Stage 5: Concept Tournament
            logger.info(f"[{workflow_id}] Stage 5: Concept Tournament")
            output.tournament_result = await self._run_concept_tournament(output.expert_analyses)

            # Stage 6: Visionary Composition
            logger.info(f"[{workflow_id}] Stage 6: Visionary Composition")
            brief = self._build_composer_brief(output)
            output.final_output = await self._compose_visionary(brief, user_goal)

            # Stage 7: Anti-copy Gate
            logger.info(f"[{workflow_id}] Stage 7: Anti-copy Gate")
            output.similarity_score = self._compute_similarity(source_document, output.final_output)

            if output.similarity_score > self._similarity_threshold:
                output.status = DocumentStage.REJECTED_SIMILAR
                raise ValueError(
                    f"Output rejected: similarity score {output.similarity_score:.2f} "
                    f"exceeds threshold {self._similarity_threshold:.2f}. "
                    "The output is too similar to the source document."
                )

            output.status = DocumentStage.COMPLETED
            logger.info(
                f"[{workflow_id}] Completed. Similarity score: {output.similarity_score:.2f}"
            )
            return output

        except Exception as e:
            output.status = DocumentStage.FAILED
            logger.error(f"[{workflow_id}] Workflow failed: {e}", exc_info=True)
            raise

    async def _extract_evidence(
        self, source_document: str, user_goal: str
    ) -> List[ExtractedFact]:
        """
        Stage 1: Extract structured facts from the source document.
        
        Uses NRAM with normal profile for precise extraction.
        """
        persona = _EvidenceExtractorPersona(nram_api_base=self._nram_api_base)
        return await persona.run(source_document=source_document, user_goal=user_goal)

    async def _deconstruct_assumptions(
        self, facts: List[ExtractedFact], user_goal: str
    ) -> List[ContrarianChallenge]:
        """
        Stage 3: Challenge assumptions in the extracted facts.
        
        Uses NRAM with threshold profile for contrarian thinking.
        """
        persona = _ContrarianDeconstructorPersona(nram_api_base=self._nram_api_base)
        return await persona.run(facts=facts, user_goal=user_goal)

    async def _run_expert_analysis(
        self,
        facts: List[ExtractedFact],
        challenges: List[ContrarianChallenge],
        user_goal: str,
    ) -> List[ExpertAnalysis]:
        """
        Stage 4: Run 6 expert personas in parallel.
        
        Each expert analyzes the facts from their unique perspective.
        """
        tasks = []
        for expert_config in EXPERT_ROLES:
            persona = _ExpertPersona(
                nram_api_base=self._nram_api_base,
                expert_config=expert_config,
            )
            tasks.append(persona.run(facts=facts, challenges=challenges, user_goal=user_goal))

        return list(await asyncio.gather(*tasks))

    async def _run_concept_tournament(
        self, expert_analyses: List[ExpertAnalysis]
    ) -> ConceptTournamentResult:
        """
        Stage 5: Select winning concepts from expert analyses.
        
        Uses NRAM with threshold profile for evaluative judgment.
        """
        persona = _ConceptTournamentPersona(nram_api_base=self._nram_api_base)
        return await persona.run(expert_analyses=expert_analyses)

    def _build_composer_brief(self, output: DocumentInnovationOutput) -> ComposerBrief:
        """
        Build the brief for the Visionary Composer.
        
        CRITICAL: This brief contains ONLY structured data:
        - facts (ExtractedFact objects)
        - citations
        - capabilities
        - contradictions
        - novel concepts
        - expert insights
        
        It does NOT contain original paragraphs from the source document.
        This is the anti-copy mechanism.
        """
        facts = output.extracted_facts
        citations = [f.citation for f in facts if f.citation]
        capabilities = [f.claim for f in facts if f.fact_type == "capability"]
        contradictions = [f.claim for f in facts if f.fact_type == "contradiction"]
        novel_concepts = []
        expert_insights = []

        for analysis in output.expert_analyses:
            novel_concepts.extend(analysis.novel_concepts)
            expert_insights.extend(analysis.key_insights)

        if output.tournament_result:
            synthesis_direction = output.tournament_result.synthesis_direction
        else:
            synthesis_direction = ""

        return ComposerBrief(
            facts=facts,
            citations=citations,
            capabilities=capabilities,
            contradictions=contradictions,
            novel_concepts=novel_concepts,
            expert_insights=expert_insights,
            synthesis_direction=synthesis_direction,
        )

    async def _compose_visionary(
        self, brief: ComposerBrief, user_goal: str
    ) -> str:
        """
        Stage 6: Compose the final visionary output.
        
        Uses NRAM with visionary-peak profile for breakthrough synthesis.
        The Composer receives ONLY the brief, not the original document.
        """
        persona = _VisionaryComposerPersona(nram_api_base=self._nram_api_base)
        return await persona.run(brief=brief, user_goal=user_goal)

    def _compute_similarity(self, source: str, output: str) -> float:
        """
        Compute similarity between source and output using SequenceMatcher.
        
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        # Normalize whitespace
        source_normalized = " ".join(source.split())
        output_normalized = " ".join(output.split())

        matcher = SequenceMatcher(None, source_normalized, output_normalized)
        return matcher.ratio()


# ---------------------------------------------------------------------------
# Persona implementations for each stage
# ---------------------------------------------------------------------------

class _EvidenceExtractorPersona(BasePersona):
    """Stage 1: Extract structured facts from source document."""

    name = "evidence_extractor"

    def __init__(self, nram_api_base: str) -> None:
        super().__init__(
            nram_api_base=nram_api_base,
            evidence_ledger=None,
            workflow_seed=271,
        )
        self._nram_config = {
            "profile": "normal",
            "intensity": 0.2,
            "temperature": 0.3,
            "coherence_floor": 0.95,
        }

    async def run(self, source_document: str, user_goal: str) -> List[ExtractedFact]:  # type: ignore[override]
        system_prompt = (
            "You are a precision evidence extractor. Your task is to extract structured facts "
            "from documents. Extract ONLY factual claims, capabilities, and contradictions. "
            "Do NOT paraphrase or summarize — extract discrete, verifiable facts.\n\n"
            "For each fact, provide:\n"
            "- claim: The factual claim (one sentence)\n"
            "- citation: Where in the document this appears (section/paragraph)\n"
            "- fact_type: One of 'claim', 'capability', 'contradiction', 'concept'\n"
            "- confidence: Your confidence this is a real fact (0.0-1.0)\n\n"
            "Return a JSON array of facts."
        )

        user_prompt = (
            f"USER GOAL: {user_goal}\n\n"
            f"SOURCE DOCUMENT:\n{source_document[:8000]}\n\n"
            "Extract all factual claims, capabilities, and contradictions from this document. "
            "Return as JSON array."
        )

        messages = self.build_messages(system_prompt, user_prompt)
        raw_output = await self.invoke_model(messages, max_tokens=4096)

        if raw_output is None:
            logger.warning("Evidence extraction failed, returning empty list")
            return []

        try:
            facts_data = self.validate_list_output(raw_output, ExtractedFact)
            # Assign IDs
            for i, fact in enumerate(facts_data):
                fact.id = f"fact-{i:03d}"
            return facts_data
        except Exception as e:
            logger.error(f"Failed to parse evidence extraction output: {e}")
            return []


class _ContrarianDeconstructorPersona(BasePersona):
    """Stage 3: Challenge assumptions in extracted facts."""

    name = "contrarian_deconstructor"

    def __init__(self, nram_api_base: str) -> None:
        super().__init__(
            nram_api_base=nram_api_base,
            evidence_ledger=None,
            workflow_seed=271,
        )
        self._nram_config = {
            "profile": "threshold",
            "intensity": 0.6,
            "temperature": 0.75,
            "contrarian_force": 0.9,
            "coherence_floor": 0.85,
        }

    async def run(  # type: ignore[override]
        self, facts: List[ExtractedFact], user_goal: str
    ) -> List[ContrarianChallenge]:
        facts_text = "\n".join(f"- {f.claim}" for f in facts[:50])

        system_prompt = (
            "You are a contrarian thinker. Your task is to challenge assumptions and reveal blind spots.\n\n"
            "For each major assumption in the facts, explain:\n"
            "- assumption_challenged: What assumption are you challenging?\n"
            "- why_it_may_be_wrong: Why might this assumption be incorrect?\n"
            "- alternative_frame: What is an alternative way to frame this?\n"
            "- evidence_required: What evidence would be needed to test this?\n\n"
            "Return a JSON array of challenges."
        )

        user_prompt = (
            f"USER GOAL: {user_goal}\n\n"
            f"EXTRACTED FACTS:\n{facts_text}\n\n"
            "Challenge the top 5-10 assumptions. Return as JSON array."
        )

        messages = self.build_messages(system_prompt, user_prompt)
        raw_output = await self.invoke_model(messages, max_tokens=4096)

        if raw_output is None:
            return []

        try:
            challenges = self.validate_list_output(raw_output, ContrarianChallenge)
            for i, challenge in enumerate(challenges):
                challenge.id = f"challenge-{i:03d}"
            return challenges
        except Exception as e:
            logger.error(f"Failed to parse contrarian output: {e}")
            return []


class _ExpertPersona(BasePersona):
    """Stage 4: Expert analysis from a specific perspective."""

    def __init__(self, nram_api_base: str, expert_config: Dict[str, Any]) -> None:
        super().__init__(
            nram_api_base=nram_api_base,
            evidence_ledger=None,
            workflow_seed=271,
        )
        self._expert_config = expert_config
        self.name = expert_config["role"].lower().replace(" ", "_")
        self._nram_config = {
            "profile": expert_config["profile"],
            "intensity": 0.7,
            "temperature": expert_config["temperature"],
        }

    async def run(  # type: ignore[override]
        self,
        facts: List[ExtractedFact],
        challenges: List[ContrarianChallenge],
        user_goal: str,
    ) -> ExpertAnalysis:
        facts_text = "\n".join(f"- {f.claim}" for f in facts[:30])
        challenges_text = "\n".join(
            f"- {c.assumption_challenged}: {c.why_it_may_be_wrong}" for c in challenges[:10]
        )

        system_prompt = (
            f"You are the {self._expert_config['role']}.\n\n"
            f"YOUR FOCUS: {self._expert_config['focus']}\n\n"
            "Analyze the provided facts and challenges from your unique perspective. "
            "Provide:\n"
            "- key_insights: 3-5 key insights from your perspective\n"
            "- novel_concepts: 2-3 novel concepts or frameworks you propose\n"
            "- contradictions_found: Any contradictions or tensions you see\n"
            "- recommended_directions: 2-3 recommended directions based on your analysis\n\n"
            "Return as JSON object."
        )

        user_prompt = (
            f"USER GOAL: {user_goal}\n\n"
            f"FACTS:\n{facts_text}\n\n"
            f"CONTRARIAN CHALLENGES:\n{challenges_text}\n\n"
            "Provide your expert analysis. Return as JSON."
        )

        messages = self.build_messages(system_prompt, user_prompt)
        raw_output = await self.invoke_model(messages, max_tokens=4096)

        if raw_output is None:
            return ExpertAnalysis(
                id=f"expert-{uuid.uuid4().hex[:8]}",
                expert_role=self._expert_config["role"],
            )

        try:
            data = self.validate_output(raw_output, ExpertAnalysis)
            data.id = f"expert-{uuid.uuid4().hex[:8]}"
            data.expert_role = self._expert_config["role"]
            return data
        except Exception as e:
            logger.error(f"Failed to parse expert output: {e}")
            return ExpertAnalysis(
                id=f"expert-{uuid.uuid4().hex[:8]}",
                expert_role=self._expert_config["role"],
            )


class _ConceptTournamentPersona(BasePersona):
    """Stage 5: Select winning concepts from expert analyses."""

    name = "concept_tournament"

    def __init__(self, nram_api_base: str) -> None:
        super().__init__(
            nram_api_base=nram_api_base,
            evidence_ledger=None,
            workflow_seed=271,
        )
        self._nram_config = {
            "profile": "threshold",
            "intensity": 0.5,
            "temperature": 0.6,
            "coherence_floor": 0.9,
        }

    async def run(self, expert_analyses: List[ExpertAnalysis]) -> ConceptTournamentResult:  # type: ignore[override]
        analyses_text = []
        for analysis in expert_analyses:
            analyses_text.append(
                f"\n[{analysis.expert_role}]\n"
                f"Insights: {', '.join(analysis.key_insights[:3])}\n"
                f"Concepts: {', '.join(analysis.novel_concepts[:3])}\n"
                f"Directions: {', '.join(analysis.recommended_directions[:2])}"
            )

        system_prompt = (
            "You are a concept tournament judge. Your task is to select the winning concepts "
            "from multiple expert analyses.\n\n"
            "Evaluate all concepts and insights, then select:\n"
            "- winning_concepts: The 3-5 strongest concepts across all analyses\n"
            "- rejected_concepts: Concepts that were interesting but not viable\n"
            "- synthesis_direction: A one-paragraph direction for how to synthesize the winners\n\n"
            "Return as JSON object."
        )

        user_prompt = (
            f"EXPERT ANALYSES:{''.join(analyses_text)}\n\n"
            "Select the winning concepts and provide synthesis direction. Return as JSON."
        )

        messages = self.build_messages(system_prompt, user_prompt)
        raw_output = await self.invoke_model(messages, max_tokens=4096)

        if raw_output is None:
            return ConceptTournamentResult(id="tournament-failed")

        try:
            result = self.validate_output(raw_output, ConceptTournamentResult)
            result.id = f"tournament-{uuid.uuid4().hex[:8]}"
            return result
        except Exception as e:
            logger.error(f"Failed to parse tournament output: {e}")
            return ConceptTournamentResult(id="tournament-failed")


class _VisionaryComposerPersona(BasePersona):
    """
    Stage 6: Compose the final visionary output.
    
    CRITICAL: This persona receives ONLY the ComposerBrief (structured facts,
    citations, capabilities, contradictions, concepts), NOT the original document.
    This prevents paraphrasing and forces genuine synthesis.
    """

    name = "visionary_composer"

    def __init__(self, nram_api_base: str) -> None:
        super().__init__(
            nram_api_base=nram_api_base,
            evidence_ledger=None,
            workflow_seed=271,
        )
        self._nram_config = {
            "profile": "visionary-peak",  # Use the new visionary-peak profile
            "intensity": 0.9,
            "temperature": 0.85,
            "visionary_intensity": 0.95,
            "associative_distance": 0.85,
            "coherence_floor": 0.88,
            "novelty_target": 0.9,
        }

    async def run(self, brief: ComposerBrief, user_goal: str) -> str:  # type: ignore[override]
        # Build structured brief text — NO original paragraphs
        facts_text = "\n".join(f"- [{f.fact_type}] {f.claim}" for f in brief.facts[:50])
        citations_text = "\n".join(f"- {c}" for c in brief.citations[:20])
        capabilities_text = "\n".join(f"- {c}" for c in brief.capabilities[:20])
        contradictions_text = "\n".join(f"- {c}" for c in brief.contradictions[:10])
        concepts_text = "\n".join(f"- {c}" for c in brief.novel_concepts[:20])
        insights_text = "\n".join(f"- {i}" for i in brief.expert_insights[:30])

        system_prompt = (
            "You are a visionary composer. Your task is to synthesize breakthrough insights "
            "from structured data.\n\n"
            "CRITICAL RULES:\n"
            "1. You will receive ONLY structured facts, citations, capabilities, contradictions, "
            "and novel concepts. You will NOT receive the original document.\n"
            "2. You must synthesize these into a coherent, visionary output.\n"
            "3. Do NOT paraphrase or quote the facts directly — synthesize them into new insights.\n"
            "4. Use the novel concepts and expert insights as building blocks for breakthrough ideas.\n"
            "5. Maintain high coherence while pushing for novelty.\n\n"
            "Your output should be a visionary synthesis that could not have been produced by "
            "simply paraphrasing the source material."
        )

        user_prompt = (
            f"USER GOAL: {user_goal}\n\n"
            f"SYNTHESIS DIRECTION: {brief.synthesis_direction}\n\n"
            f"EXTRACTED FACTS ({len(brief.facts)} total):\n{facts_text}\n\n"
            f"CITATIONS:\n{citations_text}\n\n"
            f"CAPABILITIES:\n{capabilities_text}\n\n"
            f"CONTRADICTIONS:\n{contradictions_text}\n\n"
            f"NOVEL CONCEPTS:\n{concepts_text}\n\n"
            f"EXPERT INSIGHTS:\n{insights_text}\n\n"
            "Synthesize these into a visionary output. Do NOT paraphrase the facts — "
            "create new insights from them."
        )

        messages = self.build_messages(system_prompt, user_prompt)
        raw_output = await self.invoke_model(messages, max_tokens=8192)

        if raw_output is None:
            return ""

        return raw_output.strip()
