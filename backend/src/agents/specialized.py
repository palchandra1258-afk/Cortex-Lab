"""
Specialized Agents for Cortex Lab Agentic RAG
Each agent leverages fine-tuned model capabilities:
- TimelineAgent: temporal/chronological queries + faithful generation (Stage 1)
- CausalAgent: cause-effect reasoning via causal_reason (Stage 3)
- ReflectionAgent: belief evolution via detect_belief_change (Stage 5)
- PlanningAgent: complex multi-step decomposition + RAFT (Stage 12)
- ArbitrationAgent: conflict resolution with faithful citations
"""

import time
from typing import Dict, List, Optional

from src.models import (
    AgentResponse, CausalMemoryObject, MemoryQuery, QueryIntent,
    RetrievalResult, RetrievalQuality
)
from src.llm import LocalLLM
from src.retrieval.hybrid_retriever import HybridRetriever


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, llm: LocalLLM, retriever: HybridRetriever):
        self.name = name
        self.llm = llm
        self.retriever = retriever

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        raise NotImplementedError

    def _format_evidence(self, results: List[RetrievalResult], max_items: int = 5) -> str:
        """Format retrieval results into a text block for LLM context."""
        parts = []
        for i, r in enumerate(results[:max_items]):
            ts = r.memory.timestamp.strftime("%Y-%m-%d %H:%M") if r.memory.timestamp else "Unknown"
            parts.append(f"[{i+1}] ({ts}, {r.memory.memory_type.value}, score: {r.score:.2f})\n{r.memory.content}")
        return "\n\n".join(parts) if parts else "No relevant memories found."

    def _evidence_texts(self, results: List[RetrievalResult], max_items: int = 5) -> List[str]:
        """Extract plain text snippets for LLM methods."""
        return [r.memory.content[:300] for r in results[:max_items]]


class TimelineAgent(BaseAgent):
    """Handles temporal queries using faithful generation (Stage 1 fine-tuning)."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("timeline", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        # Retrieve with temporal focus
        results = await self.retriever.retrieve(query, top_k=15)

        # Sort results by timestamp
        results.sort(key=lambda r: r.memory.timestamp)

        evidence = self._evidence_texts(results, max_items=8)

        # Use faithful generation (Stage 1) for grounded timeline narrative
        if evidence and evidence[0] != "No relevant memories found.":
            answer = self.llm.generate_faithful(
                query.raw_query, evidence,
                session_context="Focus on the timeline, sequence of events, and temporal patterns."
            )
        else:
            answer = self.llm.generate(
                f"""<|im_start|>system
You are Cortex Lab, an AI memory assistant. The user asked about a timeline
but no relevant memories were found. Say so honestly.
<|im_end|>
<|im_start|>user
{query.raw_query}
<|im_end|>
<|im_start|>assistant
""",
                max_tokens=200, temperature=0.3
            )

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.5 + len(results) * 0.05, 0.95),
            reasoning_trace=f"Timeline agent: retrieved {len(results)} memories, sorted chronologically, faithful generation",
            processing_time_ms=elapsed,
        )


class CausalAgent(BaseAgent):
    """Handles causal queries using causal_reason (Stage 3 fine-tuning)."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("causal", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=15)

        evidence = self._evidence_texts(results, max_items=8)

        # Use fine-tuned causal reasoning (Stage 3)
        if evidence:
            answer = self.llm.causal_reason(query.raw_query, evidence)
        else:
            answer = "I don't have enough stored memories to trace a causal chain for this question."

        # If causal reasoning is thin, supplement with faithful generation
        if len(answer.strip()) < 50 and evidence:
            supplement = self.llm.generate_faithful(
                query.raw_query, evidence,
                session_context="Identify cause-and-effect relationships."
            )
            if len(supplement.strip()) > len(answer.strip()):
                answer = supplement

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.06, 0.90),
            reasoning_trace=f"Causal agent: analyzed {len(results)} memories with causal_reason (Stage 3)",
            processing_time_ms=elapsed,
        )


class ReflectionAgent(BaseAgent):
    """Handles reflective queries using belief change detection (Stage 5 fine-tuning)."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("reflection", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=20)

        # Sort by time to see evolution
        results.sort(key=lambda r: r.memory.timestamp)

        evidence = self._evidence_texts(results, max_items=10)

        # Try belief change detection if we have enough temporal spread
        belief_analysis = ""
        if len(results) >= 2:
            earliest = results[0].memory.content[:200]
            latest = results[-1].memory.content[:200]
            topic = query.topics[0] if query.topics else query.raw_query[:50]

            try:
                delta = self.llm.detect_belief_change(earliest, latest, topic)
                change_type = delta.get("change_type", "unknown")
                explanation = delta.get("explanation", "")
                if explanation:
                    belief_analysis = f"\n\n**Belief Evolution ({change_type}):** {explanation}"
            except Exception:
                pass

        # Generate reflection with faithful grounding
        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query, evidence,
                session_context="Analyze patterns, evolution, and turning points in these memories."
            )
            answer += belief_analysis
        else:
            answer = "I don't have enough stored memories to identify patterns or changes for this query."

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.04, 0.85),
            reasoning_trace=f"Reflection agent: analyzed {len(results)} memories + belief change detection (Stage 5)",
            processing_time_ms=elapsed,
        )


class PlanningAgent(BaseAgent):
    """Handles complex multi-step queries with RAFT distractor-awareness (Stage 12)."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("planning", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        # Execute sub-queries if available
        all_results = []
        sub_answers = []

        sub_queries = query.sub_queries if query.sub_queries else [query.raw_query]

        for sq in sub_queries:
            sub_query = MemoryQuery(
                raw_query=sq,
                intent=query.intent,
                complexity=0.4,
                embedding=self.retriever.embeddings.embed(sq).tolist(),
            )
            results = await self.retriever.retrieve(sub_query, top_k=8)
            all_results.extend(results)

            if results:
                evidence = self._evidence_texts(results, max_items=3)
                sub_answer = self.llm.generate_faithful(sq, evidence)
                sub_answers.append(f"Q: {sq}\nA: {sub_answer}")

        # Deduplicate results
        seen = set()
        unique_results = []
        for r in all_results:
            if r.memory.id not in seen:
                seen.add(r.memory.id)
                unique_results.append(r)

        # RAFT-aware synthesis: separate oracle docs from distractors
        if len(unique_results) >= 3:
            # Top-scored are oracle, lower-scored are potential distractors
            sorted_results = sorted(unique_results, key=lambda r: r.score, reverse=True)
            oracle_docs = [r.memory.content[:250] for r in sorted_results[:3]]
            distractor_docs = [r.memory.content[:250] for r in sorted_results[3:6]]

            final_answer = self.llm.raft_generate(
                query.raw_query, oracle_docs, distractor_docs
            )
        elif sub_answers:
            # Synthesize from sub-answers with faithful generation
            combined_context = "\n\n".join(sub_answers)
            evidence = self._evidence_texts(unique_results, max_items=5)
            final_answer = self.llm.generate_faithful(
                query.raw_query, evidence,
                session_context=combined_context
            )
        else:
            final_answer = "I don't have enough stored memories to answer this complex question."

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=final_answer,
            evidence=unique_results[:5],
            confidence=min(0.5 + len(unique_results) * 0.03, 0.90),
            reasoning_trace=f"Planning agent: decomposed into {len(sub_queries)} sub-queries, {len(unique_results)} unique memories, RAFT synthesis",
            sub_queries_used=sub_queries,
            processing_time_ms=elapsed,
        )


class ArbitrationAgent(BaseAgent):
    """Handles conflicting information with faithful citation-based resolution."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("arbitration", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=15)

        evidence = self._evidence_texts(results, max_items=8)

        # Use faithful generation with explicit conflict-resolution framing
        if evidence:
            answer = self.llm.generate_faithful(
                query.raw_query, evidence,
                session_context=(
                    "Identify contradictions in the evidence. Determine which is most likely "
                    "correct based on recency, confidence, and context. Explain the evolution "
                    "from old belief to new belief. Cite evidence with [1], [2], etc."
                )
            )
        else:
            answer = "I don't have enough stored memories to compare or resolve conflicts for this query."

        # Also run belief change detection if we have temporal spread
        if len(results) >= 2:
            results_sorted = sorted(results, key=lambda r: r.memory.timestamp)
            earliest = results_sorted[0].memory.content[:200]
            latest = results_sorted[-1].memory.content[:200]
            topic = query.topics[0] if query.topics else query.raw_query[:50]
            try:
                delta = self.llm.detect_belief_change(earliest, latest, topic)
                change_type = delta.get("change_type", "")
                explanation = delta.get("explanation", "")
                if explanation and change_type in ("contradiction", "refinement"):
                    answer += f"\n\n**Belief Change ({change_type}):** {explanation}"
            except Exception:
                pass

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=0.7,
            reasoning_trace=f"Arbitration agent: analyzed {len(results)} memories with faithful generation + belief detection",
            processing_time_ms=elapsed,
        )
