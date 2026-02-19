"""
Agent Orchestrator for Cortex Lab
Routes queries to specialized agents based on intent and complexity.
Implements Adaptive-RAG routing + CRAG quality evaluation + Self-RAG reflection.
"""

import asyncio
import time
from typing import Dict, List, Optional

from src.models import (
    AgentResponse, MemoryQuery, OrchestratorResponse, QueryIntent,
    RetrievalQuality, RetrievalResult, RoutingStrategy
)
from src.llm import LocalLLM
from src.retrieval.query_engine import QueryAnalyzer, QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.specialized import (
    TimelineAgent, CausalAgent, ReflectionAgent, PlanningAgent, ArbitrationAgent
)


class AgentOrchestrator:
    """
    Central orchestrator implementing Adaptive-RAG:
    - Simple queries → No retrieval, direct LLM answer
    - Moderate queries → Single agent
    - Complex queries → Multi-agent with synthesis
    
    Post-retrieval: CRAG quality check + Self-RAG reflection loop.
    """

    def __init__(self, llm: LocalLLM, retriever: HybridRetriever,
                 analyzer: QueryAnalyzer, transformer: QueryTransformer):
        self.llm = llm
        self.retriever = retriever
        self.analyzer = analyzer
        self.transformer = transformer

        # Initialize specialized agents
        self.agents = {
            "timeline": TimelineAgent(llm, retriever),
            "causal": CausalAgent(llm, retriever),
            "reflection": ReflectionAgent(llm, retriever),
            "planning": PlanningAgent(llm, retriever),
            "arbitration": ArbitrationAgent(llm, retriever),
        }

        # Intent → Agent mapping
        self.intent_to_agent = {
            QueryIntent.TEMPORAL: "timeline",
            QueryIntent.CAUSAL: "causal",
            QueryIntent.REFLECTIVE: "reflection",
            QueryIntent.COMPARATIVE: "arbitration",
            QueryIntent.FACTUAL: "planning",
            QueryIntent.PROCEDURAL: "planning",
            QueryIntent.EXPLORATORY: "planning",
        }

    async def process(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """
        Full orchestration pipeline:
        1. Analyze query (intent, complexity, routing)
        2. Transform query (multi-query, HyDE, step-back)
        3. Route to agent(s)
        4. CRAG quality evaluation
        5. Self-RAG reflection loop
        6. Return final response
        """
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  🧠 Orchestrator: Processing query")
        print(f"  📝 Query: {raw_query[:80]}...")
        print(f"{'='*60}")

        # 1. Analyze query
        query = self.analyzer.analyze(raw_query)

        # 2. Transform query (add multi-query, HyDE, etc.)
        query = self.transformer.transform(query)

        # 3. Route based on complexity
        if query.routing == RoutingStrategy.NO_RETRIEVAL:
            response = await self._handle_no_retrieval(query)
        elif query.routing == RoutingStrategy.SINGLE_STEP:
            response = await self._handle_single_step(query)
        else:
            response = await self._handle_multi_step(query)

        # 4. CRAG quality evaluation
        response = await self._crag_evaluate(query, response)

        # 5. Self-RAG reflection (if quality is not CORRECT)
        if response.confidence < 0.7 and response.evidence:
            response = await self._self_rag_reflect(query, response)

        response.query_analysis = query
        response.processing_time_ms = (time.time() - t0) * 1000

        print(f"\n  ✅ Response ready: confidence={response.confidence:.2f}, "
              f"agents={response.agents_used}, time={response.processing_time_ms:.0f}ms\n")

        return response

    async def _handle_no_retrieval(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle simple queries that don't need memory retrieval."""
        print("  ⚡ Routing: NO_RETRIEVAL (simple query)")

        answer = self.llm.generate(
            f"Answer the following question concisely:\n\n{query.raw_query}\n\nAnswer:",
            max_tokens=300, temperature=0.3
        )

        return OrchestratorResponse(
            answer=answer,
            thinking="This is a simple query that doesn't require searching through memories.",
            agents_used=["direct"],
            confidence=0.8,
            reasoning_trace="Direct LLM answer (no retrieval needed)",
        )

    async def _handle_single_step(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle moderate queries with a single agent."""
        agent_name = self.intent_to_agent.get(query.intent, "planning")
        agent = self.agents.get(agent_name, self.agents["planning"])

        print(f"  🔀 Routing: SINGLE_STEP → {agent_name} agent")

        agent_response = await agent.execute(query)

        thinking = (
            f"Intent: {query.intent.value} (complexity: {query.complexity:.2f})\n"
            f"Agent: {agent_name}\n"
            f"Evidence: {len(agent_response.evidence)} memories retrieved\n"
            f"Reasoning: {agent_response.reasoning_trace}"
        )

        return OrchestratorResponse(
            answer=agent_response.answer,
            thinking=thinking,
            evidence=agent_response.evidence,
            agents_used=[agent_name],
            confidence=agent_response.confidence,
            reasoning_trace=agent_response.reasoning_trace,
        )

    async def _handle_multi_step(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle complex queries with multiple agents."""
        # Determine which agents to use
        primary_agent_name = self.intent_to_agent.get(query.intent, "planning")
        agent_names = [primary_agent_name]

        # Add secondary agents based on intent
        if query.intent == QueryIntent.CAUSAL:
            agent_names.append("timeline")
        elif query.intent == QueryIntent.REFLECTIVE:
            agent_names.append("causal")
        elif query.intent == QueryIntent.TEMPORAL:
            agent_names.append("reflection")

        # Always include planning for decomposed queries
        if "planning" not in agent_names and query.sub_queries:
            agent_names.append("planning")

        print(f"  🔀 Routing: MULTI_STEP → {agent_names}")

        # Execute agents in parallel
        tasks = []
        for name in agent_names:
            agent = self.agents.get(name, self.agents["planning"])
            tasks.append(agent.execute(query))

        agent_responses = await asyncio.gather(*tasks)

        # Synthesize responses
        combined_answers = []
        all_evidence = []
        all_traces = []

        for name, resp in zip(agent_names, agent_responses):
            combined_answers.append(f"[{name.title()} Agent]: {resp.answer}")
            all_evidence.extend(resp.evidence)
            all_traces.append(f"{name}: {resp.reasoning_trace}")

        # LLM synthesis of multi-agent output
        synthesis_prompt = f"""You are an AI assistant synthesizing information from multiple 
analysis perspectives about the user's memories.

Question: {query.raw_query}

Agent Analyses:
{chr(10).join(combined_answers)}

Provide a unified, comprehensive answer that combines insights from all analyses.
Be concise but thorough.

Synthesized Answer:"""

        final_answer = self.llm.generate(synthesis_prompt, max_tokens=500, temperature=0.3)

        # Deduplicate evidence
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            if e.memory.id not in seen:
                seen.add(e.memory.id)
                unique_evidence.append(e)

        avg_confidence = sum(r.confidence for r in agent_responses) / len(agent_responses)

        thinking = (
            f"Intent: {query.intent.value} (complexity: {query.complexity:.2f})\n"
            f"Agents: {', '.join(agent_names)}\n"
            f"Total evidence: {len(unique_evidence)} unique memories\n"
            f"Traces:\n" + "\n".join(f"  - {t}" for t in all_traces)
        )

        return OrchestratorResponse(
            answer=final_answer,
            thinking=thinking,
            evidence=unique_evidence[:10],
            agents_used=agent_names,
            confidence=avg_confidence,
            reasoning_trace=f"Multi-agent synthesis from {len(agent_names)} agents",
        )

    async def _crag_evaluate(self, query: MemoryQuery, response: OrchestratorResponse) -> OrchestratorResponse:
        """
        CRAG (Corrective RAG): Evaluate retrieval quality.
        → CORRECT: Use as is
        → AMBIGUOUS: Supplement with more retrieval
        → INCORRECT: Refine query and re-retrieve
        """
        if not response.evidence:
            return response  # Nothing to evaluate

        # Simple heuristic evaluation (avoid extra LLM call for speed)
        avg_score = sum(r.score for r in response.evidence) / len(response.evidence)

        if avg_score > 0.6:
            # CORRECT: Good retrieval quality
            return response
        elif avg_score > 0.3:
            # AMBIGUOUS: Supplement
            response.reasoning_trace += " | CRAG: AMBIGUOUS — retrieval quality moderate"
            response.confidence *= 0.85
        else:
            # INCORRECT: Low quality, caveat the answer
            response.reasoning_trace += " | CRAG: INCORRECT — retrieval quality low"
            response.confidence *= 0.6
            response.answer = (
                "⚠️ *Note: Limited relevant memories found. The following answer is based on "
                "partial information:*\n\n" + response.answer
            )

        return response

    async def _self_rag_reflect(self, query: MemoryQuery, response: OrchestratorResponse) -> OrchestratorResponse:
        """
        Self-RAG: Generate → Critique → Revise loop.
        Max 2 iterations for latency budget.
        """
        if self.llm.model is None:
            return response

        critique_prompt = f"""Evaluate this answer about someone's personal memories.

Question: {query.raw_query}
Answer: {response.answer[:300]}

Rate on a scale of 1-10:
1. Relevance (does it answer the question?)
2. Faithfulness (is it grounded in the evidence?)
3. Completeness (does it cover all aspects?)

Overall score (just the number):"""

        try:
            score_text = self.llm.generate(critique_prompt, max_tokens=10, temperature=0.1)
            # Parse score
            import re
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = int(numbers[0])
                if score >= 7:
                    response.confidence = min(response.confidence + 0.1, 0.95)
                    response.reasoning_trace += f" | Self-RAG: score={score}/10 (accepted)"
                else:
                    response.confidence = max(response.confidence - 0.1, 0.3)
                    response.reasoning_trace += f" | Self-RAG: score={score}/10 (low confidence)"
        except Exception:
            pass

        return response

    def process_sync(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(raw_query, session_context))
        finally:
            loop.close()
