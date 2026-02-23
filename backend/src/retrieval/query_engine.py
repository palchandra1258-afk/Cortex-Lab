"""
Query Intelligence Layer for Cortex Lab
- Intent Detection (keyword heuristics + LLM fallback via route_query Stage 2)
- Complexity Scoring
- Adaptive Routing
- Multi-Query Generation (RAG-Fusion)
- HyDE (Hypothetical Document Embedding)
- Step-Back Prompting
- Query Decomposition
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.models import MemoryQuery, QueryIntent, RoutingStrategy
from src.models.embeddings import EmbeddingModel
from src.llm import LocalLLM


class QueryAnalyzer:
    """
    Analyzes queries to determine intent, complexity, and routing.
    Uses keyword heuristics first, with LLM fallback for ambiguous cases.
    The LLM fallback leverages route_query (Stage 2 fine-tuning) but is
    deferred to the orchestrator for async execution.
    """

    # Intent keyword mappings
    INTENT_KEYWORDS = {
        QueryIntent.TEMPORAL: [
            "when", "what time", "how long ago", "last week", "yesterday",
            "last month", "in january", "in february", "in march", "in april",
            "in may", "in june", "in july", "in august", "in september",
            "in october", "in november", "in december", "timeline", "chronolog",
            "sequence", "before", "after", "during",
        ],
        QueryIntent.CAUSAL: [
            "why", "because", "caused", "led to", "reason", "result of",
            "consequence", "what made me", "what caused", "how come",
            "factor", "influence",
        ],
        QueryIntent.REFLECTIVE: [
            "how did my", "changed", "evolved", "pattern", "realized",
            "over time", "growth", "progress", "trend", "shift in",
            "belief", "opinion changed",
        ],
        QueryIntent.FACTUAL: [
            "what is", "what are", "who is", "define", "explain",
            "tell me about", "describe", "what did I learn",
        ],
        QueryIntent.PROCEDURAL: [
            "how do", "how to", "steps", "process", "method",
            "procedure", "workflow", "guide",
        ],
        QueryIntent.COMPARATIVE: [
            "compare", "difference", "similar", "versus", "vs",
            "better", "worse", "prefer",
        ],
        QueryIntent.EXPLORATORY: [
            "tell me", "what about", "anything about", "related to",
            "show me", "find", "search",
        ],
    }

    # Complexity indicators
    COMPLEXITY_BOOSTERS = [
        "why", "how did", "evolution", "over time", "relationship between",
        "compare", "analyze", "pattern", "all the times", "chain of events",
        "led to", "caused", "history of", "trace",
    ]

    def analyze(self, query: str) -> MemoryQuery:
        """Full query analysis: intent + complexity + routing + temporal extraction."""
        t0 = time.time()
        query_lower = query.lower().strip()

        # 1. Detect intent
        intent = self._detect_intent(query_lower)

        # 2. Score complexity
        complexity = self._score_complexity(query_lower)

        # 3. Determine routing
        routing = self._determine_routing(complexity)

        # 4. Extract temporal constraints
        time_start, time_end = self._extract_temporal(query_lower)

        # 5. Extract entities from query
        entities = self._extract_query_entities(query)

        # 6. Extract topics
        topics = self._extract_query_topics(query_lower)

        result = MemoryQuery(
            raw_query=query,
            intent=intent,
            complexity=complexity,
            routing=routing,
            time_start=time_start,
            time_end=time_end,
            entities=entities,
            topics=topics,
            confidence=0.8,
        )

        elapsed = (time.time() - t0) * 1000
        print(f"  🔍 Query analyzed: intent={intent.value}, complexity={complexity:.2f}, routing={routing.value} ({elapsed:.0f}ms)")

        return result

    def _detect_intent(self, query: str) -> QueryIntent:
        """Keyword-based intent detection."""
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query)
            if score > 0:
                scores[intent] = score

        if scores:
            return max(scores, key=scores.get)
        return QueryIntent.EXPLORATORY

    def _score_complexity(self, query: str) -> float:
        """Score query complexity 0.0-1.0."""
        score = 0.3  # baseline

        # Word count
        words = len(query.split())
        if words > 15:
            score += 0.1
        if words > 25:
            score += 0.1

        # Complexity indicators
        for booster in self.COMPLEXITY_BOOSTERS:
            if booster in query:
                score += 0.1

        # Multiple questions
        if query.count("?") > 1:
            score += 0.15

        # Conjunctions suggesting multi-part
        if any(w in query for w in ["and", "also", "additionally", "then"]):
            score += 0.05

        return min(score, 1.0)

    def _determine_routing(self, complexity: float) -> RoutingStrategy:
        """Determine routing strategy based on complexity."""
        if complexity < 0.3:
            return RoutingStrategy.NO_RETRIEVAL
        elif complexity < 0.6:
            return RoutingStrategy.SINGLE_STEP
        else:
            return RoutingStrategy.MULTI_STEP

    def _extract_temporal(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract time range from query."""
        now = datetime.now()
        start = None
        end = None

        # Relative time patterns
        if "yesterday" in query:
            start = now - timedelta(days=1)
            end = now
        elif "last week" in query:
            start = now - timedelta(weeks=1)
            end = now
        elif "last month" in query:
            start = now - timedelta(days=30)
            end = now
        elif "last year" in query:
            start = now - timedelta(days=365)
            end = now
        elif "today" in query:
            start = now.replace(hour=0, minute=0, second=0)
            end = now

        # Month name patterns
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        for month_name, month_num in months.items():
            if month_name in query:
                year = now.year
                # If month is in the future, use last year
                if month_num > now.month:
                    year -= 1
                start = datetime(year, month_num, 1)
                if month_num == 12:
                    end = datetime(year + 1, 1, 1)
                else:
                    end = datetime(year, month_num + 1, 1)
                break

        return start, end

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity references from query."""
        entities = []
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)
        return entities

    def _extract_query_topics(self, query: str) -> List[str]:
        """Extract topics from query."""
        topic_map = {
            "work": ["work", "job", "career", "office", "project"],
            "health": ["health", "exercise", "gym", "doctor", "sleep"],
            "relationships": ["friend", "family", "partner", "relationship"],
            "learning": ["learn", "study", "course", "book", "research"],
            "technology": ["code", "programming", "AI", "machine learning"],
            "finance": ["money", "budget", "invest", "salary"],
        }
        topics = []
        for topic, keywords in topic_map.items():
            if any(kw in query for kw in keywords):
                topics.append(topic)
        return topics


class QueryTransformer:
    """
    Transforms queries for improved retrieval coverage.
    - Multi-Query Generation (RAG-Fusion)
    - HyDE (Hypothetical Document Embedding)
    - Step-Back Prompting
    - Query Decomposition
    """

    def __init__(self, llm: LocalLLM, embedding_model: EmbeddingModel):
        self.llm = llm
        self.embeddings = embedding_model

    def transform(self, query: MemoryQuery) -> MemoryQuery:
        """Apply all relevant transformations based on routing strategy."""
        t0 = time.time()

        if query.routing == RoutingStrategy.NO_RETRIEVAL:
            return query  # Skip transformations for simple queries

        # Multi-query generation (always for SINGLE_STEP and MULTI_STEP)
        query.multi_queries = self._generate_multi_queries(query.raw_query)

        # HyDE for factual/exploratory queries
        if query.intent in (QueryIntent.FACTUAL, QueryIntent.EXPLORATORY, QueryIntent.PROCEDURAL):
            query.hyde_answer = self._generate_hyde(query.raw_query)

        # Step-back for causal/reflective queries
        if query.intent in (QueryIntent.CAUSAL, QueryIntent.REFLECTIVE) and query.complexity > 0.5:
            query.step_back_query = self._generate_step_back(query.raw_query)

        # Decomposition for complex multi-step queries
        if query.routing == RoutingStrategy.MULTI_STEP:
            query.sub_queries = self._decompose_query(query.raw_query)

        # Generate query embedding
        query.embedding = self.embeddings.embed(query.raw_query).tolist()

        elapsed = (time.time() - t0) * 1000
        variants = len(query.multi_queries) + (1 if query.hyde_answer else 0) + (1 if query.step_back_query else 0) + len(query.sub_queries)
        print(f"  🔄 Query transformed: {variants} variants ({elapsed:.0f}ms)")

        return query

    def _generate_multi_queries(self, query: str) -> List[str]:
        """Generate query variants for RAG-Fusion."""
        if self.llm.model is None:
            return [query]  # Return original if no LLM

        prompt = f"""<|im_start|>system
Generate 3 different versions of the following question.
Each version should preserve the meaning but use different wording.
Output one version per line, numbered 1-3.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
1."""

        result = self.llm.generate(prompt, max_tokens=150, temperature=0.5)
        lines = [l.strip() for l in result.split("\n") if l.strip()]

        variants = []
        for line in lines:
            # Clean up numbering
            clean = re.sub(r'^(Version\s*)?\d+[:.]\s*', '', line).strip()
            if clean and len(clean) > 5 and clean != query:
                variants.append(clean)

        return variants[:3] if variants else [query]

    def _generate_hyde(self, query: str) -> str:
        """Generate hypothetical answer for HyDE."""
        if self.llm.model is None:
            return ""

        prompt = f"""<|im_start|>system
Write a brief hypothetical answer (2-3 sentences) to this question,
as if answering from personal memories.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

        return self.llm.generate(prompt, max_tokens=100, temperature=0.4).strip()

    def _generate_step_back(self, query: str) -> str:
        """Generate a step-back (more abstract) question."""
        if self.llm.model is None:
            return ""

        prompt = f"""<|im_start|>system
Given this specific question, generate ONE more general question
that would provide useful background context.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

        return self.llm.generate(prompt, max_tokens=50, temperature=0.3).strip()

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        if self.llm.model is None:
            return [query]

        prompt = f"""<|im_start|>system
Break this complex question into 2-3 simpler sub-questions
that can each be answered independently.
Output one sub-question per line, numbered 1-3.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
1."""

        result = self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        lines = [l.strip() for l in result.split("\n") if l.strip()]

        sub_queries = []
        for line in lines:
            clean = re.sub(r'^\d+[:.]\s*', '', line).strip()
            if clean and len(clean) > 5:
                sub_queries.append(clean)

        return sub_queries[:3] if sub_queries else [query]
