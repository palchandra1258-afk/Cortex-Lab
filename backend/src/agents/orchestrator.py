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
        CRAG (Corrective RAG): Evaluate retrieval quality with multi-signal assessment.
        → CORRECT: Use as is
        → AMBIGUOUS: Supplement with more retrieval
        → INCORRECT: Refine query and re-retrieve
        """
        if not response.evidence:
            return response

        # Multi-signal quality evaluation
        avg_score = sum(r.score for r in response.evidence) / len(response.evidence)
        max_score = max(r.score for r in response.evidence)
        evidence_count = len(response.evidence)

        # Check entity coverage (do retrieved memories mention query entities?)
        entity_coverage = 0.0
        if query.entities:
            matched = 0
            for ent in query.entities:
                for r in response.evidence:
                    if ent.lower() in r.memory.content.lower():
                        matched += 1
                        break
            entity_coverage = matched / len(query.entities)

        # Combined quality score
        quality_score = (
            0.40 * avg_score +
            0.20 * max_score +
            0.20 * min(evidence_count / 5.0, 1.0) +
            0.20 * entity_coverage
        )

        if quality_score > 0.55:
            # CORRECT: Good retrieval quality
            response.reasoning_trace += f" | CRAG: CORRECT (quality={quality_score:.2f})"
            return response
        elif quality_score > 0.30:
            # AMBIGUOUS: Try to supplement with additional retrieval
            response.reasoning_trace += f" | CRAG: AMBIGUOUS (quality={quality_score:.2f})"
            response.confidence *= 0.85

            # Attempt supplementary retrieval with step-back query
            if query.step_back_query:
                try:
                    from src.models import MemoryQuery as MQ
                    sb_query = MQ(
                        raw_query=query.step_back_query,
                        intent=query.intent,
                        complexity=0.4,
                        embedding=self.retriever.embeddings.embed(query.step_back_query).tolist(),
                    )
                    extra_results = await self.retriever.retrieve(sb_query, top_k=5)
                    # Add non-duplicate results
                    existing_ids = {r.memory.id for r in response.evidence}
                    for r in extra_results:
                        if r.memory.id not in existing_ids:
                            response.evidence.append(r)
                            existing_ids.add(r.memory.id)
                    response.reasoning_trace += f" → supplemented +{len(extra_results)} from step-back"
                except Exception:
                    pass
        else:
            # INCORRECT: Low quality, caveat the answer
            response.reasoning_trace += f" | CRAG: INCORRECT (quality={quality_score:.2f})"
            response.confidence *= 0.55
            response.answer = (
                "⚠️ *Note: Limited relevant memories found. The following answer is based on "
                "partial information:*\n\n" + response.answer
            )

        return response

    async def _self_rag_reflect(self, query: MemoryQuery, response: OrchestratorResponse) -> OrchestratorResponse:
        """
        Self-RAG: Generate → Critique → Revise loop.
        Max 2 iterations for latency budget.
        Evaluates: relevance, faithfulness, completeness.
        """
        if self.llm.model is None:
            return response

        # Format evidence for critique
        evidence_summary = ""
        for i, r in enumerate(response.evidence[:3]):
            evidence_summary += f"[{i+1}] {r.memory.content[:150]}\n"

        critique_prompt = f"""Evaluate this answer about someone's personal memories.

Question: {query.raw_query}
Answer: {response.answer[:400]}

Supporting evidence:
{evidence_summary}

Rate each criterion 1-10:
1. RELEVANCE: Does the answer address the question?
2. FAITHFULNESS: Is the answer grounded in the evidence?
3. COMPLETENESS: Does it cover the key aspects?

Respond with three numbers separated by commas (e.g., 8,7,6):"""

        try:
            score_text = self.llm.generate(critique_prompt, max_tokens=20, temperature=0.1)
            import re
            numbers = re.findall(r'\d+', score_text)

            if len(numbers) >= 3:
                relevance = min(int(numbers[0]), 10)
                faithfulness = min(int(numbers[1]), 10)
                completeness = min(int(numbers[2]), 10)
                avg_score = (relevance + faithfulness + completeness) / 3.0

                if avg_score >= 7:
                    response.confidence = min(response.confidence + 0.1, 0.95)
                    response.reasoning_trace += f" | Self-RAG: {relevance}/{faithfulness}/{completeness} (accepted)"
                elif avg_score >= 5:
                    # Attempt revision with explicit instruction on weak areas
                    weak_area = "relevance" if relevance < faithfulness and relevance < completeness else (
                        "faithfulness" if faithfulness < completeness else "completeness"
                    )
                    revision_prompt = f"""Revise this answer to improve {weak_area}.

Question: {query.raw_query}
Original answer: {response.answer[:300]}

Evidence:
{evidence_summary}

Improved answer (focus on {weak_area}):"""
                    revised = self.llm.generate(revision_prompt, max_tokens=400, temperature=0.3)
                    if len(revised.strip()) > 20:
                        response.answer = revised.strip()
                        response.reasoning_trace += f" | Self-RAG: {relevance}/{faithfulness}/{completeness} → revised ({weak_area})"
                        response.confidence = min(response.confidence + 0.05, 0.85)
                    else:
                        response.reasoning_trace += f" | Self-RAG: {relevance}/{faithfulness}/{completeness} (revision failed)"
                else:
                    response.confidence = max(response.confidence - 0.15, 0.25)
                    response.reasoning_trace += f" | Self-RAG: {relevance}/{faithfulness}/{completeness} (low quality)"
            elif numbers:
                score = int(numbers[0])
                if score >= 7:
                    response.confidence = min(response.confidence + 0.1, 0.95)
                else:
                    response.confidence = max(response.confidence - 0.1, 0.3)
                response.reasoning_trace += f" | Self-RAG: score={score}/10"
        except Exception as e:
            response.reasoning_trace += f" | Self-RAG: error ({str(e)[:50]})"

        return response

    def process_sync(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(raw_query, session_context))
        finally:
            loop.close()
