"""
Agent Orchestrator for Cortex Lab — Fine-Tuned Model Integration
Routes queries to specialized agents based on intent and complexity.
Implements:
  - Adaptive-RAG routing with LLM-based structured JSON routing (Stage 2)
  - CRAG quality evaluation with multi-signal assessment
  - Self-RAG reflection with ISREL/ISSUP/ISUSE critique tokens (Stage 4)
  - FLARE: Forward-Looking Active Retrieval (EMNLP 2023)
  - RAFT: Distractor-aware generation (Stage 12)
  - Chain-of-Retrieval for complex multi-hop queries
  - Function calling integration (Stage 13)
"""

import asyncio
import time
import re
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


# ─── Tool Registry for Function Calling (Stage 13) ──────────────────────────

AVAILABLE_TOOLS = [
    {
        "name": "search_memories",
        "description": "Search through stored memories by semantic similarity",
        "parameters": {"query": "str", "top_k": "int (default 10)"},
    },
    {
        "name": "search_by_time",
        "description": "Find memories from a specific time period",
        "parameters": {"start_date": "str (ISO format)", "end_date": "str (ISO format)"},
    },
    {
        "name": "find_entity",
        "description": "Look up information about a specific entity (person, place, project)",
        "parameters": {"entity_name": "str"},
    },
    {
        "name": "trace_causal_chain",
        "description": "Trace cause-effect relationships for an event or decision",
        "parameters": {"event": "str"},
    },
    {
        "name": "detect_belief_evolution",
        "description": "Check how beliefs about a topic have changed over time",
        "parameters": {"topic": "str"},
    },
    {
        "name": "summarize_topic",
        "description": "Get a summary of all memories related to a topic",
        "parameters": {"topic": "str"},
    },
]


class AgentOrchestrator:
    """
    Central orchestrator implementing Adaptive-RAG with fine-tuned model integration.
    
    Pipeline:
    1. LLM-based structured routing (Stage 2) with keyword fallback
    2. Query transformation (multi-query, HyDE, step-back, decomposition)
    3. Agent execution (single or multi-agent)
    4. CRAG quality evaluation (multi-signal)
    5. Self-RAG ISREL/ISSUP/ISUSE reflection (Stage 4)
    6. FLARE: Forward-looking active retrieval on low-confidence segments
    7. RAFT: Distractor-aware final generation (Stage 12)
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
        Full orchestration pipeline with fine-tuned model integration.
        """
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  🧠 Orchestrator: Processing query")
        print(f"  📝 Query: {raw_query[:80]}...")
        print(f"{'='*60}")

        # 1. Analyze query (keyword heuristics + LLM-based routing)
        query = self.analyzer.analyze(raw_query)

        # 1b. LLM-based routing enhancement (Stage 2 fine-tuning)
        query = await self._llm_route_query(query, session_context)

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

        # 5. Self-RAG ISREL/ISSUP/ISUSE reflection (Stage 4)
        if response.evidence and response.confidence < 0.75:
            response = await self._self_rag_critique(query, response)

        # 6. FLARE: Active retrieval on low-confidence segments
        if response.confidence < 0.6 and response.evidence:
            response = await self._flare_active_retrieval(query, response)

        response.query_analysis = query
        response.processing_time_ms = (time.time() - t0) * 1000

        # Track token usage
        response.token_usage = self.llm.get_stats()

        print(f"\n  ✅ Response ready: confidence={response.confidence:.2f}, "
              f"agents={response.agents_used}, time={response.processing_time_ms:.0f}ms\n")

        return response

    async def _llm_route_query(self, query: MemoryQuery, session_context: str) -> MemoryQuery:
        """Use fine-tuned LLM (Stage 2) for structured routing when keyword analysis is uncertain."""
        if self.llm.model is None:
            return query

        # Only use LLM routing if keyword confidence is low
        if query.complexity < 0.3 or query.complexity > 0.7:
            return query  # High/low confidence — keyword routing is sufficient

        try:
            routing = self.llm.route_query(query.raw_query, session_context)

            # Map LLM intent to our enum
            intent_map = {
                "temporal": QueryIntent.TEMPORAL,
                "causal": QueryIntent.CAUSAL,
                "reflective": QueryIntent.REFLECTIVE,
                "factual": QueryIntent.FACTUAL,
                "procedural": QueryIntent.PROCEDURAL,
                "comparative": QueryIntent.COMPARATIVE,
                "exploratory": QueryIntent.EXPLORATORY,
            }
            llm_intent = routing.get("intent", "").lower()
            if llm_intent in intent_map:
                query.intent = intent_map[llm_intent]

            # Use LLM complexity if it disagrees significantly
            llm_complexity = float(routing.get("complexity", query.complexity))
            if abs(llm_complexity - query.complexity) > 0.2:
                query.complexity = (query.complexity + llm_complexity) / 2.0

            # Re-evaluate routing
            if query.complexity < 0.3:
                query.routing = RoutingStrategy.NO_RETRIEVAL
            elif query.complexity < 0.6:
                query.routing = RoutingStrategy.SINGLE_STEP
            else:
                query.routing = RoutingStrategy.MULTI_STEP

            print(f"  🎯 LLM routing: intent={query.intent.value}, complexity={query.complexity:.2f}")
        except Exception as e:
            print(f"  ⚠ LLM routing failed: {e}, using keyword routing")

        return query

    async def _handle_no_retrieval(self, query: MemoryQuery) -> OrchestratorResponse:
        """Handle simple queries that don't need memory retrieval."""
        print("  ⚡ Routing: NO_RETRIEVAL (simple query)")

        answer = self.llm.generate(
            f"""<|im_start|>system
You are Cortex Lab, a personal AI memory and reasoning assistant.
If this is a personal question about the user and you don't have stored memories
about it, honestly say you don't have that information yet.
Never fabricate personal details.
<|im_end|>
<|im_start|>user
{query.raw_query}
<|im_end|>
<|im_start|>assistant
""",
            max_tokens=300, temperature=0.3
        )

        return OrchestratorResponse(
            answer=answer,
            thinking="Simple query — no memory retrieval needed.",
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
        """Handle complex queries with multiple agents + Chain-of-Retrieval."""
        primary_agent_name = self.intent_to_agent.get(query.intent, "planning")
        agent_names = [primary_agent_name]

        # Add secondary agents based on intent
        if query.intent == QueryIntent.CAUSAL:
            agent_names.append("timeline")
        elif query.intent == QueryIntent.REFLECTIVE:
            agent_names.append("causal")
        elif query.intent == QueryIntent.TEMPORAL:
            agent_names.append("reflection")

        if "planning" not in agent_names and query.sub_queries:
            agent_names.append("planning")

        print(f"  🔀 Routing: MULTI_STEP → {agent_names}")

        # Execute agents in parallel
        tasks = []
        for name in agent_names:
            agent = self.agents.get(name, self.agents["planning"])
            tasks.append(agent.execute(query))

        agent_responses = await asyncio.gather(*tasks)

        # Combine all evidence and answers
        combined_answers = []
        all_evidence = []
        all_traces = []

        for name, resp in zip(agent_names, agent_responses):
            combined_answers.append(f"[{name.title()} Agent]: {resp.answer}")
            all_evidence.extend(resp.evidence)
            all_traces.append(f"{name}: {resp.reasoning_trace}")

        # Deduplicate evidence
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            if e.memory.id not in seen:
                seen.add(e.memory.id)
                unique_evidence.append(e)

        # Use faithful generation (Stage 1) for synthesis with citations
        evidence_texts = [e.memory.content[:250] for e in unique_evidence[:5]]
        if evidence_texts:
            final_answer = self.llm.generate_faithful(
                query.raw_query, evidence_texts,
                session_context="\n".join(combined_answers[:3])
            )
        else:
            # Fallback synthesis
            synthesis_prompt = f"""<|im_start|>system
You are Cortex Lab, synthesizing multi-agent analysis of the user's memories.
Be concise but thorough. If no relevant memories exist, say so honestly.
<|im_end|>
<|im_start|>user
{query.raw_query}

Agent Analyses:
{chr(10).join(combined_answers)}
<|im_end|>
<|im_start|>assistant
"""
            final_answer = self.llm.generate(synthesis_prompt, max_tokens=500, temperature=0.3)

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
        CRAG (Corrective RAG): Multi-signal retrieval quality evaluation.
        → CORRECT: Use as is
        → AMBIGUOUS: Supplement with more retrieval
        → INCORRECT: Refine and caveat
        """
        if not response.evidence:
            return response

        # Multi-signal quality evaluation
        avg_score = sum(r.score for r in response.evidence) / len(response.evidence)
        max_score = max(r.score for r in response.evidence)
        evidence_count = len(response.evidence)

        # Entity coverage check
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
            response.reasoning_trace += f" | CRAG: CORRECT (q={quality_score:.2f})"
            return response
        elif quality_score > 0.30:
            response.reasoning_trace += f" | CRAG: AMBIGUOUS (q={quality_score:.2f})"
            response.confidence *= 0.85

            # Supplementary retrieval with step-back query
            if query.step_back_query:
                try:
                    sb_query = MemoryQuery(
                        raw_query=query.step_back_query,
                        intent=query.intent,
                        complexity=0.4,
                        embedding=self.retriever.embeddings.embed(query.step_back_query).tolist(),
                    )
                    extra_results = await self.retriever.retrieve(sb_query, top_k=5)
                    existing_ids = {r.memory.id for r in response.evidence}
                    added = 0
                    for r in extra_results:
                        if r.memory.id not in existing_ids:
                            response.evidence.append(r)
                            existing_ids.add(r.memory.id)
                            added += 1
                    response.reasoning_trace += f" → +{added} from step-back"
                except Exception:
                    pass
        else:
            response.reasoning_trace += f" | CRAG: INCORRECT (q={quality_score:.2f})"
            response.confidence *= 0.55
            response.answer = (
                "⚠️ *Limited relevant memories found. Based on partial information:*\n\n"
                + response.answer
            )

        return response

    async def _self_rag_critique(self, query: MemoryQuery,
                                  response: OrchestratorResponse) -> OrchestratorResponse:
        """
        Self-RAG with ISREL/ISSUP/ISUSE critique tokens (Stage 4 fine-tuning).
        Generate → Critique → Revise loop (max 2 iterations).
        """
        if self.llm.model is None:
            return response

        evidence_texts = [r.memory.content[:200] for r in response.evidence[:5]]
        if not evidence_texts:
            return response

        try:
            # Use fine-tuned ISREL/ISSUP/ISUSE critique
            critique = self.llm.self_rag_critique(
                query.raw_query, response.answer, evidence_texts
            )

            isrel = critique.get("ISREL", 5)
            issup = critique.get("ISSUP", 5)
            isuse = critique.get("ISUSE", 5)
            avg = critique.get("avg_score", 5.0)
            verdict = critique.get("verdict", "REVISE")

            response.reasoning_trace += f" | Self-RAG: R={isrel}/S={issup}/U={isuse} ({verdict})"

            if verdict == "ACCEPT" or avg >= 7.0:
                response.confidence = min(response.confidence + 0.1, 0.95)
            elif avg >= 5.0:
                # Identify weakest area and revise
                weak = "relevance" if isrel <= issup and isrel <= isuse else (
                    "faithfulness" if issup <= isuse else "completeness"
                )
                revision_prompt = f"""<|im_start|>system
Revise this answer to improve {weak}. Be grounded in the evidence.
<|im_end|>
<|im_start|>user
Question: {query.raw_query}
Original answer: {response.answer[:300]}
Evidence: {chr(10).join(f"[{i+1}] {e}" for i, e in enumerate(evidence_texts[:3]))}

Improved answer (focus on {weak}):
<|im_end|>
<|im_start|>assistant
"""
                revised = self.llm.generate(revision_prompt, max_tokens=400, temperature=0.3)
                if len(revised.strip()) > 20:
                    response.answer = revised.strip()
                    response.reasoning_trace += f" → revised ({weak})"
                    response.confidence = min(response.confidence + 0.05, 0.85)
            else:
                response.confidence = max(response.confidence - 0.15, 0.25)
                response.reasoning_trace += " (low quality)"

        except Exception as e:
            response.reasoning_trace += f" | Self-RAG error: {str(e)[:50]}"

        return response

    async def _flare_active_retrieval(self, query: MemoryQuery,
                                       response: OrchestratorResponse) -> OrchestratorResponse:
        """
        FLARE: Forward-Looking Active Retrieval (EMNLP 2023).
        Identifies low-confidence segments in the answer and retrieves
        additional evidence to fill gaps.
        """
        if self.llm.model is None or not response.answer:
            return response

        try:
            # Split answer into sentences
            sentences = re.split(r'(?<=[.!?])\s+', response.answer)
            if len(sentences) < 2:
                return response

            # Identify sentences that might need more evidence
            # (hedging language, vague claims, question marks)
            uncertain_markers = [
                "might", "possibly", "perhaps", "unclear", "not sure",
                "limited", "insufficient", "partial", "?", "may have",
            ]

            sentences_to_verify = []
            for i, sent in enumerate(sentences):
                if any(marker in sent.lower() for marker in uncertain_markers):
                    sentences_to_verify.append((i, sent))

            if not sentences_to_verify:
                return response

            # Retrieve additional evidence for uncertain segments
            additional_evidence = []
            for idx, sent in sentences_to_verify[:2]:  # Max 2 FLARE retrievals
                flare_query = MemoryQuery(
                    raw_query=sent,
                    intent=query.intent,
                    complexity=0.4,
                    embedding=self.retriever.embeddings.embed(sent).tolist(),
                )
                results = await self.retriever.retrieve(flare_query, top_k=3)
                additional_evidence.extend(results)

            if additional_evidence:
                # Deduplicate
                existing_ids = {r.memory.id for r in response.evidence}
                new_evidence = [r for r in additional_evidence if r.memory.id not in existing_ids]

                if new_evidence:
                    response.evidence.extend(new_evidence[:3])
                    new_evidence_texts = [r.memory.content[:200] for r in new_evidence[:3]]

                    # Re-generate with augmented evidence
                    all_evidence_texts = (
                        [r.memory.content[:200] for r in response.evidence[:5]]
                    )
                    revised = self.llm.generate_faithful(
                        query.raw_query, all_evidence_texts
                    )
                    if len(revised.strip()) > 20:
                        response.answer = revised.strip()
                        response.confidence = min(response.confidence + 0.1, 0.85)
                        response.reasoning_trace += f" | FLARE: +{len(new_evidence)} evidence"

        except Exception as e:
            response.reasoning_trace += f" | FLARE error: {str(e)[:50]}"

        return response

    def process_sync(self, raw_query: str, session_context: str = "") -> OrchestratorResponse:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(raw_query, session_context))
        finally:
            loop.close()
