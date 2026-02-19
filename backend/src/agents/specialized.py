"""
Specialized Agents for Cortex Lab Agentic RAG
Each agent handles a specific type of reasoning:
- TimelineAgent: temporal/chronological queries
- CausalAgent: why/cause-effect reasoning
- ReflectionAgent: belief evolution and patterns
- PlanningAgent: complex multi-step decomposition
- ArbitrationAgent: conflict resolution
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
            parts.append(f"[Memory {i+1}] ({ts}, {r.memory.memory_type.value}, score: {r.score:.2f})\n{r.memory.content}")
        return "\n\n".join(parts) if parts else "No relevant memories found."


class TimelineAgent(BaseAgent):
    """Handles temporal queries: 'When did X happen?', 'What was I doing in June?'"""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("timeline", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        # Retrieve with temporal focus
        results = await self.retriever.retrieve(query, top_k=15)

        # Sort results by timestamp
        results.sort(key=lambda r: r.memory.timestamp)

        evidence_text = self._format_evidence(results, max_items=8)

        # Generate timeline narrative
        prompt = f"""You are an AI memory assistant. Based on the user's memories below, 
provide a chronological narrative answering the question.

Focus on the timeline, sequence of events, and temporal patterns.
If there's not enough context, say so honestly.

Question: {query.raw_query}

Memories (chronological):
{evidence_text}

Answer (narrative format):"""

        answer = self.llm.generate(prompt, max_tokens=400, temperature=0.3)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.5 + len(results) * 0.05, 0.95),
            reasoning_trace=f"Timeline agent: retrieved {len(results)} memories, sorted chronologically",
            processing_time_ms=elapsed,
        )


class CausalAgent(BaseAgent):
    """Handles causal queries: 'Why did I do X?', 'What led to Y?'"""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("causal", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=15)

        evidence_text = self._format_evidence(results, max_items=8)

        prompt = f"""You are an AI memory assistant specialized in causal reasoning.
Analyze the user's memories to find cause-and-effect relationships.

For the question below, identify:
1. The main event/decision
2. What led to it (causes, influences, preceding events)
3. What resulted from it (consequences, effects)

If you can trace a causal chain, present it step by step.
If there's insufficient evidence, say so honestly.

Question: {query.raw_query}

Relevant Memories:
{evidence_text}

Causal Analysis:"""

        answer = self.llm.generate(prompt, max_tokens=500, temperature=0.3)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.06, 0.90),
            reasoning_trace=f"Causal agent: analyzed {len(results)} memories for cause-effect chains",
            processing_time_ms=elapsed,
        )


class ReflectionAgent(BaseAgent):
    """Handles reflective queries: 'How did my thinking change?', 'What patterns do I see?'"""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("reflection", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=20)

        # Sort by time to see evolution
        results.sort(key=lambda r: r.memory.timestamp)

        evidence_text = self._format_evidence(results, max_items=10)

        prompt = f"""You are an AI memory assistant specializing in self-reflection and pattern analysis.
Analyze the user's memories to identify:

1. How their thinking/beliefs evolved over time
2. Recurring patterns or themes
3. Key turning points or realizations
4. Contradictions or changes in stance

Question: {query.raw_query}

Memories (chronological):
{evidence_text}

Reflection Analysis:"""

        answer = self.llm.generate(prompt, max_tokens=500, temperature=0.3)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=min(0.4 + len(results) * 0.04, 0.85),
            reasoning_trace=f"Reflection agent: analyzed {len(results)} memories for patterns and evolution",
            processing_time_ms=elapsed,
        )


class PlanningAgent(BaseAgent):
    """Handles complex multi-step queries requiring decomposition."""

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
                evidence = self._format_evidence(results, max_items=3)
                sub_answer = self.llm.generate(
                    f"Based on these memories, briefly answer: {sq}\n\nMemories:\n{evidence}\n\nAnswer:",
                    max_tokens=150, temperature=0.3
                )
                sub_answers.append(f"Q: {sq}\nA: {sub_answer}")

        # Synthesize final answer
        combined_context = "\n\n".join(sub_answers) if sub_answers else "No sub-answers generated."

        prompt = f"""You are an AI memory assistant. The following sub-questions were answered 
from the user's memories. Synthesize a comprehensive final answer.

Original Question: {query.raw_query}

Sub-Question Answers:
{combined_context}

Synthesized Answer:"""

        answer = self.llm.generate(prompt, max_tokens=500, temperature=0.3)

        # Deduplicate results
        seen = set()
        unique_results = []
        for r in all_results:
            if r.memory.id not in seen:
                seen.add(r.memory.id)
                unique_results.append(r)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=unique_results[:5],
            confidence=min(0.5 + len(unique_results) * 0.03, 0.90),
            reasoning_trace=f"Planning agent: decomposed into {len(sub_queries)} sub-queries, retrieved {len(unique_results)} unique memories",
            sub_queries_used=sub_queries,
            processing_time_ms=elapsed,
        )


class ArbitrationAgent(BaseAgent):
    """Handles conflicting information and belief resolution."""

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever):
        super().__init__("arbitration", llm, retriever)

    async def execute(self, query: MemoryQuery, context: str = "") -> AgentResponse:
        t0 = time.time()

        results = await self.retriever.retrieve(query, top_k=15)

        evidence_text = self._format_evidence(results, max_items=8)

        prompt = f"""You are an AI memory assistant specializing in resolving conflicts 
and contradictions in the user's memories.

Analyze the memories below and:
1. Identify any contradicting information
2. Determine which is most likely correct (based on recency, confidence, context)
3. Explain the evolution from old belief to new belief
4. If no contradiction exists, explain the consistent thread

Question: {query.raw_query}

Memories:
{evidence_text}

Analysis:"""

        answer = self.llm.generate(prompt, max_tokens=400, temperature=0.3)

        elapsed = (time.time() - t0) * 1000
        return AgentResponse(
            agent_name=self.name,
            answer=answer,
            evidence=results[:5],
            confidence=0.7,
            reasoning_trace=f"Arbitration agent: analyzed {len(results)} memories for conflicts",
            processing_time_ms=elapsed,
        )
