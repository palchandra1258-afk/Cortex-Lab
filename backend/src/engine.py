"""
Cortex Lab RAG Engine — Central Integration Point
Ties together all components: LLM, Embeddings, Storage, Agents, Cache, Ingestion.
Provides a single interface for the FastAPI server.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional

from src.models import (
    CausalMemoryObject, MemoryQuery, OrchestratorResponse,
    RetrievalResult, BeliefDelta
)
from src.models.embeddings import EmbeddingModel
from src.llm import LocalLLM
from src.storage.vector_store import VectorStore
from src.storage.metadata_store import MetadataStore
from src.storage.knowledge_graph import KnowledgeGraph
from src.retrieval.query_engine import QueryAnalyzer, QueryTransformer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agents.orchestrator import AgentOrchestrator
from src.ingestion import MemoryIngestionPipeline
from src.cache import MultiLevelCache


class CortexRAGEngine:
    """
    The central Agentic RAG engine for Cortex Lab.
    
    This is the single entry point that the FastAPI server uses.
    It initializes and wires together all subsystems:
    - EmbeddingModel (sentence-transformers)
    - VectorStore (FAISS)
    - MetadataStore (DuckDB)
    - KnowledgeGraph (NetworkX)
    - LocalLLM (DeepSeek-R1-1.5B interface)
    - QueryAnalyzer + QueryTransformer
    - HybridRetriever (5-channel)
    - AgentOrchestrator (5 specialized agents)
    - MemoryIngestionPipeline
    - MultiLevelCache
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.initialized = False

        # Components (initialized in init())
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_store: Optional[VectorStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.llm: Optional[LocalLLM] = None
        self.query_analyzer: Optional[QueryAnalyzer] = None
        self.query_transformer: Optional[QueryTransformer] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.ingestion: Optional[MemoryIngestionPipeline] = None
        self.cache: Optional[MultiLevelCache] = None

        # Session tracking
        self._current_session_id = ""
        self._session_context = ""

    def init(self, model=None, tokenizer=None):
        """
        Initialize all RAG components.
        Called during server startup after the LLM model loads.
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("  🧠 Initializing Cortex Lab RAG Engine")
        print("=" * 60)

        # 1. Embedding Model
        print("\n[1/9] Embedding Model...")
        self.embedding_model = EmbeddingModel(device="cpu")

        # 2. Vector Store
        print("[2/9] Vector Store...")
        self.vector_store = VectorStore(
            dimension=self.embedding_model.dimension,
            data_dir=f"{self.data_dir}/vectors"
        )

        # 3. Metadata Store
        print("[3/9] Metadata Store...")
        self.metadata_store = MetadataStore(db_path=f"{self.data_dir}/cortex.duckdb")

        # 4. Knowledge Graph
        print("[4/9] Knowledge Graph...")
        self.knowledge_graph = KnowledgeGraph(data_dir=f"{self.data_dir}/graph")

        # 5. LLM Interface
        print("[5/9] LLM Interface...")
        self.llm = LocalLLM(model=model, tokenizer=tokenizer)

        # 6. Query Engine
        print("[6/9] Query Engine...")
        self.query_analyzer = QueryAnalyzer()
        self.query_transformer = QueryTransformer(self.llm, self.embedding_model)

        # 7. Hybrid Retriever
        print("[7/9] Hybrid Retriever...")
        self.hybrid_retriever = HybridRetriever(
            self.embedding_model, self.vector_store,
            self.metadata_store, self.knowledge_graph
        )

        # 8. Agent Orchestrator
        print("[8/9] Agent Orchestrator...")
        self.orchestrator = AgentOrchestrator(
            self.llm, self.hybrid_retriever,
            self.query_analyzer, self.query_transformer
        )

        # 9. Ingestion Pipeline
        print("[9/9] Ingestion Pipeline + Cache...")
        self.ingestion = MemoryIngestionPipeline(
            self.llm, self.embedding_model,
            self.vector_store, self.metadata_store,
            self.knowledge_graph
        )
        self.cache = MultiLevelCache(self.embedding_model)

        # Run tier migration on startup
        try:
            self.vector_store.migrate_tiers()
        except Exception as e:
            print(f"  ⚠ Tier migration skipped: {e}")

        self.initialized = True
        elapsed = time.time() - t0
        print(f"\n  ✅ RAG Engine ready in {elapsed:.1f}s")
        print(f"  📊 Memories: {self.metadata_store.count_memories()} | "
              f"Vectors: {self.vector_store.count()} | "
              f"Graph: {self.knowledge_graph.get_stats()}")
        print("=" * 60 + "\n")

    def set_model(self, model, tokenizer):
        """Update LLM reference (called when model finishes loading)."""
        if self.llm:
            self.llm.set_model(model, tokenizer)

    # ─── RAG-Enhanced Chat ───────────────────────────────────────────────

    async def rag_chat(self, user_message: str, session_id: str = "",
                        conversation_history: List[Dict] = None) -> Dict:
        """
        Main RAG-enhanced chat endpoint.
        1. Ingest user message as memory
        2. Check cache
        3. Run agentic RAG pipeline
        4. Return enhanced response with evidence
        """
        if not self.initialized:
            return {"answer": "RAG system is still initializing...", "evidence": []}

        t0 = time.time()

        # Set session
        if not session_id:
            session_id = f"session-{int(time.time())}"
        self._current_session_id = session_id

        # Build session context from history
        session_context = ""
        if conversation_history:
            recent = conversation_history[-6:]  # Last 3 exchanges
            session_context = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')[:200]}"
                for m in recent
            )

        # 1. Ingest user message as memory
        memory = await self.ingestion.ingest(
            content=user_message,
            session_id=session_id,
            source="chat",
            session_context=session_context,
        )

        # Invalidate retriever caches after new ingestion
        self.hybrid_retriever.invalidate_caches()

        # 2. Check cache
        cached, cache_level = self.cache.get(user_message)
        if cached:
            print(f"  ⚡ Cache hit ({cache_level})")
            cached["cache_hit"] = True
            cached["cache_level"] = cache_level
            return cached

        # 3. Run orchestrator
        response = await self.orchestrator.process(user_message, session_context)

        # 4. Format result
        result = {
            "answer": response.answer,
            "thinking": response.thinking,
            "evidence": [
                {
                    "content": e.memory.content[:300],
                    "score": round(e.score, 3),
                    "channel": e.channel,
                    "timestamp": e.memory.timestamp.isoformat(),
                    "memory_type": e.memory.memory_type.value,
                    "emotion": e.memory.emotion.value,
                    "entities": e.memory.entities[:5],
                }
                for e in response.evidence[:5]
            ],
            "agents_used": response.agents_used,
            "confidence": round(response.confidence, 3),
            "reasoning_trace": response.reasoning_trace,
            "query_analysis": {
                "intent": response.query_analysis.intent.value if response.query_analysis else "unknown",
                "complexity": round(response.query_analysis.complexity, 2) if response.query_analysis else 0,
                "routing": response.query_analysis.routing.value if response.query_analysis else "unknown",
            },
            "processing_time_ms": round(response.processing_time_ms, 1),
            "cache_hit": False,
        }

        # 5. Cache result
        self.cache.set(user_message, result)

        # 6. Store conversation turn
        self.metadata_store.store_conversation_turn(
            session_id=session_id,
            role="user",
            content=user_message,
            memory_id=memory.id,
        )
        self.metadata_store.store_conversation_turn(
            session_id=session_id,
            role="assistant",
            content=response.answer,
            thinking=response.thinking,
        )

        return result

    # ─── Memory Management ───────────────────────────────────────────────

    async def ingest_memory(self, content: str, source: str = "manual",
                             session_id: str = "") -> Dict:
        """Manually ingest a memory."""
        if not self.initialized:
            return {"error": "RAG system not initialized"}

        memory = await self.ingestion.ingest(
            content=content, session_id=session_id, source=source
        )

        # Invalidate caches (new memory might change future answers)
        self.cache.invalidate_topic(memory.topics[0] if memory.topics else "")
        self.hybrid_retriever.invalidate_caches()

        return memory.to_dict()

    def get_memories(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get all memories with pagination."""
        if not self.initialized:
            return []
        memories = self.metadata_store.get_all_memories(limit=limit, offset=offset)
        return [m.to_dict() for m in memories]

    def search_memories(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search memories by content similarity."""
        if not self.initialized:
            return []
        embedding = self.embedding_model.embed(query)
        results = self.vector_store.search(embedding, top_k=top_k)

        memories = []
        for mid, score in results:
            mem = self.metadata_store.get_memory(mid)
            if mem:
                d = mem.to_dict()
                d["score"] = round(score, 3)
                memories.append(d)
        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self.initialized:
            return False
        self.metadata_store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)
        return True

    # ─── Knowledge Graph ─────────────────────────────────────────────────

    def get_graph_data(self) -> Dict:
        """Get graph data for visualization."""
        if not self.initialized:
            return {"nodes": [], "edges": []}
        return self.knowledge_graph.get_graph_data()

    def get_entities(self, limit: int = 100) -> List[Dict]:
        """Get all entities."""
        if not self.initialized:
            return []
        return self.metadata_store.get_entities(limit=limit)

    def get_belief_deltas(self, limit: int = 50) -> List[Dict]:
        """Get detected belief evolution events."""
        if not self.initialized:
            return []
        try:
            return self.metadata_store.get_belief_deltas(limit=limit)
        except Exception:
            return []

    def get_community_summaries(self) -> List[Dict]:
        """Get GraphRAG community summaries."""
        if not self.initialized:
            return []
        return self.knowledge_graph.get_community_summaries()

    # ─── System Stats ────────────────────────────────────────────────────

    def get_rag_stats(self) -> Dict:
        """Get comprehensive RAG system statistics."""
        if not self.initialized:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "memories": self.metadata_store.get_stats(),
            "vectors": self.vector_store.get_stats(),
            "graph": self.knowledge_graph.get_stats(),
            "cache": self.cache.get_stats(),
            "llm": self.llm.get_stats(),
        }

    # ─── Persistence ─────────────────────────────────────────────────────

    def save_all(self):
        """Persist all data to disk."""
        if not self.initialized:
            return
        print("\n💾 Saving all data...")
        self.vector_store.save()
        self.knowledge_graph.save()
        print("  ✅ All data saved\n")

    def shutdown(self):
        """Graceful shutdown."""
        self.save_all()
        if self.metadata_store:
            self.metadata_store.close()


# Singleton instance
rag_engine = CortexRAGEngine()
