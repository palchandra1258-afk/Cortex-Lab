"""
Memory Ingestion Pipeline for Cortex Lab
Processes raw text into rich CausalMemoryObjects with:
- Memory type classification
- Emotion detection
- Entity extraction
- Topic extraction
- Proposition decomposition
- Contextual chunking
- Embedding generation
"""

import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models import (
    CausalMemoryObject, MemoryType, EmotionLabel, EntityNode, GraphEdge
)
from src.models.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph


class MemoryIngestionPipeline:
    """
    Full ingestion pipeline: raw text → enriched CausalMemoryObject → stored.
    
    Pipeline stages:
    1. Text preprocessing
    2. Memory type classification
    3. Emotion detection
    4. Entity extraction
    5. Topic extraction
    6. Importance scoring
    7. Proposition decomposition
    8. Contextual prefix generation
    9. Embedding generation
    10. Storage (vector + metadata + graph)
    """

    def __init__(self, llm: LocalLLM, embedding_model: EmbeddingModel,
                 vector_store: VectorStore, metadata_store: MetadataStore,
                 knowledge_graph: KnowledgeGraph):
        self.llm = llm
        self.embeddings = embedding_model
        self.vectors = vector_store
        self.metadata = metadata_store
        self.graph = knowledge_graph

        # Keyword-based classifiers (fast, no LLM needed for ~85% of cases)
        self._emotion_keywords = {
            EmotionLabel.HAPPY: ["happy", "joy", "excited", "great", "wonderful", "love", "amazing", "glad", "pleased", "delighted"],
            EmotionLabel.SAD: ["sad", "depressed", "down", "unhappy", "miserable", "grief", "loss", "miss"],
            EmotionLabel.ANGRY: ["angry", "furious", "mad", "irritated", "annoyed", "frustrated", "rage"],
            EmotionLabel.ANXIOUS: ["anxious", "worried", "nervous", "stress", "panic", "fear", "uncertain"],
            EmotionLabel.EXCITED: ["excited", "thrilled", "eager", "pumped", "can't wait", "stoked"],
            EmotionLabel.CONFUSED: ["confused", "puzzled", "unsure", "don't understand", "lost", "bewildered"],
            EmotionLabel.HOPEFUL: ["hopeful", "optimistic", "promising", "looking forward", "positive"],
            EmotionLabel.FRUSTRATED: ["frustrated", "stuck", "annoyed", "struggling", "difficult"],
        }

        self._memory_type_keywords = {
            MemoryType.EPISODIC: ["went to", "met with", "had", "visited", "talked to", "saw", "did"],
            MemoryType.SEMANTIC: ["learned", "understood", "concept", "means", "defined as", "is a", "theory"],
            MemoryType.PROCEDURAL: ["how to", "process", "steps", "method", "procedure", "workflow"],
            MemoryType.REFLECTIVE: ["realized", "think", "feel", "believe", "changed my mind", "pattern", "noticed"],
        }

    async def ingest(self, content: str, session_id: str = "",
                     source: str = "chat", session_context: str = "") -> CausalMemoryObject:
        """
        Full ingestion pipeline. Returns enriched memory object.
        """
        t0 = time.time()

        # 1. Create base memory
        memory = CausalMemoryObject(
            content=content.strip(),
            session_id=session_id,
            source=source,
            timestamp=datetime.now(),
        )

        # 2. Classify memory type (keyword-first, LLM fallback)
        memory.memory_type = self._classify_memory_type(content)

        # 3. Detect emotion
        memory.emotion, memory.emotion_confidence = self._detect_emotion(content)

        # 4. Extract entities
        memory.entities = self._extract_entities(content)

        # 5. Extract topics
        memory.topics = self._extract_topics(content)

        # 6. Score importance
        memory.importance = self._score_importance(content, memory)

        # 7. Decompose into propositions
        memory.propositions = self._extract_propositions(content)

        # 8. Generate contextual prefix
        if session_context:
            memory.context_prefix = self._generate_context_prefix(content, session_context)

        # 9. Generate embedding (on contextual content)
        embed_text = memory.context_prefix + " " + content if memory.context_prefix else content
        embedding = self.embeddings.embed(embed_text)
        memory.embedding = embedding.tolist()

        # 10. Store everything
        self.vectors.add(memory.id, embedding, memory.timestamp)
        self.metadata.store_memory(memory)

        # 11. Update knowledge graph with entities
        self._update_graph(memory)

        elapsed_ms = (time.time() - t0) * 1000
        print(f"  📝 Memory ingested: [{memory.memory_type.value}] {content[:60]}... ({elapsed_ms:.0f}ms)")

        return memory

    def _classify_memory_type(self, text: str) -> MemoryType:
        """Classify memory type using keyword matching."""
        text_lower = text.lower()
        scores = {}
        for mtype, keywords in self._memory_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[mtype] = score

        if scores:
            return max(scores, key=scores.get)

        # LLM fallback for ambiguous cases
        if self.llm.model is not None:
            result = self.llm.classify(
                f"Classify this memory:\n\"{text[:200]}\"\n\nTypes: episodic (events/activities), semantic (facts/knowledge), procedural (processes/how-to), reflective (thoughts/realizations)",
                ["episodic", "semantic", "procedural", "reflective"],
                default="episodic"
            )
            try:
                return MemoryType(result.lower().strip())
            except ValueError:
                pass

        return MemoryType.EPISODIC

    def _detect_emotion(self, text: str) -> Tuple[EmotionLabel, float]:
        """Detect emotion using keyword scoring."""
        text_lower = text.lower()
        scores = {}
        for emotion, keywords in self._emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            best = max(scores, key=scores.get)
            confidence = min(scores[best] / 3.0, 1.0)
            return best, confidence

        return EmotionLabel.NEUTRAL, 0.5

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities using pattern matching and capitalization heuristics."""
        entities = []

        # Find capitalized words (likely proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and i > 0 and len(clean) > 1:
                # Not at start of sentence (likely proper noun)
                entities.append(clean)

        # Find quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted)

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                unique.append(e)

        return unique[:10]  # Cap at 10 entities

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using simple keyword/category matching."""
        topic_keywords = {
            "work": ["work", "job", "office", "meeting", "project", "deadline", "colleague", "boss", "career"],
            "health": ["health", "exercise", "gym", "doctor", "sleep", "diet", "sick", "medicine", "workout"],
            "relationships": ["friend", "family", "partner", "relationship", "love", "date", "social"],
            "learning": ["learn", "study", "course", "book", "read", "understand", "tutorial", "research"],
            "technology": ["code", "programming", "software", "computer", "AI", "machine learning", "model", "algorithm"],
            "finance": ["money", "budget", "invest", "salary", "expense", "save", "cost", "price"],
            "personal": ["feel", "think", "believe", "want", "goal", "dream", "plan", "decide"],
            "creative": ["write", "design", "art", "music", "create", "build", "idea", "project"],
        }

        text_lower = text.lower()
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics[:5]

    def _score_importance(self, text: str, memory: CausalMemoryObject) -> float:
        """Score importance 0.0-1.0 based on heuristics."""
        score = 0.5

        # Longer content tends to be more important
        word_count = len(text.split())
        if word_count > 50:
            score += 0.1
        if word_count > 100:
            score += 0.1

        # Emotional content is more important
        if memory.emotion != EmotionLabel.NEUTRAL:
            score += 0.1
        if memory.emotion_confidence > 0.7:
            score += 0.1

        # Reflective memories are higher importance
        if memory.memory_type == MemoryType.REFLECTIVE:
            score += 0.15

        # Entities increase importance
        if len(memory.entities) > 2:
            score += 0.1

        # Decision keywords
        decision_words = ["decided", "chose", "will", "plan to", "going to", "committed"]
        if any(w in text.lower() for w in decision_words):
            score += 0.15

        return min(max(score, 0.0), 1.0)

    def _extract_propositions(self, text: str) -> List[str]:
        """
        Extract atomic propositions using LLM-based decomposition (EMNLP 2024).
        Falls back to enhanced sentence splitting if LLM unavailable.
        """
        # Try LLM-based atomic fact decomposition first
        if self.llm.model is not None and len(text) > 30:
            try:
                result = self.llm.generate(
                    f"""Decompose the following text into independent atomic facts.
Each fact should be a single, self-contained statement that can be understood without context.
Return one fact per line, no numbering.

Text: "{text[:500]}"

Atomic facts:""",
                    max_tokens=300,
                    temperature=0.1,
                )
                props = [p.strip().lstrip("- •·") for p in result.strip().split("\n")]
                props = [p for p in props if len(p) > 10 and not p.startswith("Atomic")]
                if props:
                    return props[:12]
            except Exception:
                pass

        # Enhanced fallback: split by clauses and sentences
        propositions = []
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            # Further split compound sentences on conjunctions
            clauses = re.split(r'\s+(?:and|but|however|although|because|so|then)\s+', sent, flags=re.IGNORECASE)
            for clause in clauses:
                clause = clause.strip().rstrip('.!?')
                if len(clause) > 10:
                    propositions.append(clause)
        return propositions[:12]

    def _generate_context_prefix(self, content: str, session_context: str) -> str:
        """Generate contextual prefix using session context (Anthropic-style)."""
        if self.llm.model is not None and session_context:
            prefix = self.llm.generate(
                f"""Given this conversation session context:
{session_context[:500]}

And this specific memory:
{content[:300]}

Write a SHORT context (1-2 sentences) to situate this memory. Include who/what/when if relevant.
Context:""",
                max_tokens=60,
                temperature=0.2,
            )
            return prefix.strip()
        return ""

    def _update_graph(self, memory: CausalMemoryObject):
        """Update knowledge graph with extracted entities and relationships."""
        for entity_name in memory.entities:
            # Check if entity already exists
            existing_id = self.graph.find_entity_by_name(entity_name)
            if existing_id:
                # Update existing entity's last_seen and memory_ids
                if self.graph.graph is not None and existing_id in self.graph.graph:
                    node = self.graph.graph.nodes[existing_id]
                    node["last_seen"] = memory.timestamp.isoformat()
                    existing_mids = node.get("memory_ids", [])
                    if memory.id not in existing_mids:
                        existing_mids.append(memory.id)
                        node["memory_ids"] = existing_mids
            else:
                # Create new entity
                entity = EntityNode(
                    canonical_name=entity_name,
                    entity_type=self._infer_entity_type(entity_name, memory.content),
                    first_seen=memory.timestamp,
                    last_seen=memory.timestamp,
                    memory_ids=[memory.id],
                )
                self.graph.add_entity(entity)

        # Create edges between co-occurring entities
        for i, ent1 in enumerate(memory.entities):
            for ent2 in memory.entities[i + 1:]:
                id1 = self.graph.find_entity_by_name(ent1)
                id2 = self.graph.find_entity_by_name(ent2)
                if id1 and id2:
                    edge = GraphEdge(
                        source_id=id1,
                        target_id=id2,
                        relation=self._infer_relation(ent1, ent2, memory.content),
                        weight=1.0,
                        memory_ids=[memory.id],
                        timestamp=memory.timestamp,
                    )
                    self.graph.add_edge(edge)

        # Detect and store belief evolution
        self._detect_belief_evolution(memory)

    def _infer_entity_type(self, entity_name: str, context: str) -> str:
        """Infer entity type from context clues."""
        context_lower = context.lower()
        name_lower = entity_name.lower()

        # Person indicators
        person_words = ["met", "talked to", "said", "told me", "friend", "colleague", "manager", "boss", "partner"]
        if any(w in context_lower for w in person_words):
            return "person"

        # Place indicators
        place_words = ["went to", "visited", "at the", "in the", "location", "city", "country", "office"]
        if any(w in context_lower for w in place_words):
            return "place"

        # Project indicators
        project_words = ["project", "codebase", "repo", "app", "system", "framework", "tool"]
        if any(w in context_lower for w in project_words):
            return "project"

        # Concept/topic indicators
        concept_words = ["concept", "theory", "idea", "principle", "method", "approach"]
        if any(w in context_lower for w in concept_words):
            return "concept"

        return "unknown"

    def _infer_relation(self, entity1: str, entity2: str, context: str) -> str:
        """Infer the relation between two co-occurring entities."""
        context_lower = context.lower()
        causal_words = ["because", "caused", "led to", "resulted in", "due to"]
        if any(w in context_lower for w in causal_words):
            return "caused"
        collab_words = ["worked with", "collaborated", "together", "team"]
        if any(w in context_lower for w in collab_words):
            return "works_with"
        discuss_words = ["discussed", "talked about", "mentioned", "about"]
        if any(w in context_lower for w in discuss_words):
            return "discussed"
        return "co_mentioned"

    def _detect_belief_evolution(self, memory: CausalMemoryObject):
        """
        Detect belief contradictions/evolution when a new memory is ingested.
        Multi-stage pipeline per RAG-Architecture §9:
        1. Semantic similarity: find memories about same topic (>0.85)
        2. Stance detection: classify stance change
        3. Temporal context: weight recency
        4. Classification: CONTRADICTION / REFINEMENT / EXPANSION / ABANDONMENT
        5. Storage as BeliefDelta
        """
        from src.models import BeliefDelta, BeliefChangeType

        if not memory.embedding or memory.memory_type not in (MemoryType.REFLECTIVE, MemoryType.SEMANTIC):
            return

        # Stage 1: Find semantically similar past memories (same topic)
        query_emb = self.embeddings.embed(memory.content)
        similar = self.vectors.search(query_emb, top_k=10)

        for mem_id, sim_score in similar:
            if sim_score < 0.75 or mem_id == memory.id:
                continue

            old_memory = self.metadata.get_memory(mem_id)
            if not old_memory:
                continue

            # Skip if same session (likely continuation, not contradiction)
            if old_memory.session_id == memory.session_id and memory.session_id:
                continue

            # Stage 2: Stance detection via keyword heuristics
            stance = self._detect_stance(old_memory.content, memory.content)
            if stance == "agree":
                continue  # No belief change

            # Stage 3: Temporal context — require minimum time gap (1 day)
            time_gap = abs((memory.timestamp - old_memory.timestamp).total_seconds())
            if time_gap < 86400:  # Less than 1 day apart
                continue

            # Stage 4: Classify change type
            if stance == "disagree":
                change_type = BeliefChangeType.CONTRADICTION
                confidence = min(sim_score + 0.1, 1.0)
            elif stance == "expand":
                change_type = BeliefChangeType.REFINEMENT
                confidence = sim_score * 0.8
            else:
                continue

            # Stage 5: Store BeliefDelta
            topic = memory.topics[0] if memory.topics else "general"
            delta = BeliefDelta(
                topic=topic,
                old_belief_id=old_memory.id,
                new_belief_id=memory.id,
                old_belief_text=old_memory.content[:200],
                new_belief_text=memory.content[:200],
                change_type=change_type,
                confidence=confidence,
                detected_at=datetime.now(),
                evidence_chain=[old_memory.id, memory.id],
            )
            self.metadata.store_belief_delta(delta)
            print(f"  🔄 Belief evolution detected: {change_type.value} on '{topic}' (conf: {confidence:.2f})")
            break  # Only detect the most significant change per ingestion

    def _detect_stance(self, old_text: str, new_text: str) -> str:
        """Detect stance between two texts: agree, disagree, expand, neutral."""
        old_lower = old_text.lower()
        new_lower = new_text.lower()

        # Strong disagreement indicators
        negation_pairs = [
            ("love", "hate"), ("like", "dislike"), ("good", "bad"), ("great", "terrible"),
            ("agree", "disagree"), ("support", "oppose"), ("yes", "no"),
            ("happy", "unhappy"), ("positive", "negative"), ("enjoy", "dread"),
        ]
        for pos, neg in negation_pairs:
            if (pos in old_lower and neg in new_lower) or (neg in old_lower and pos in new_lower):
                return "disagree"

        # Explicit contradiction words in new text
        contradiction_words = ["actually", "i was wrong", "changed my mind", "no longer",
                                "not anymore", "contrary to", "opposite", "however"]
        if any(w in new_lower for w in contradiction_words):
            return "disagree"

        # Expansion indicators
        expand_words = ["also", "additionally", "moreover", "learned that", "realized",
                        "in addition", "furthermore", "building on"]
        if any(w in new_lower for w in expand_words):
            return "expand"

        return "neutral"
