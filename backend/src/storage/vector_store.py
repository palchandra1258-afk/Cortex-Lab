"""
Vector Store for Cortex Lab
FAISS-based vector storage with tiered indexing (Hot/Warm/Cold).
"""

import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class VectorStore:
    """
    FAISS-based vector store with hot/warm/cold tiering.
    - Hot (HNSW): Recent memories (<30 days), ~5ms, ~98% recall
    - Warm (IVFFlat): 30 days-1 year, ~15ms, ~95% recall
    - Cold (IVFFlat): >1 year, ~25ms, ~90% recall
    
    Falls back to brute-force flat index if FAISS is unavailable.
    """

    def __init__(self, dimension: int = 384, data_dir: str = "data/vectors"):
        self.dimension = dimension
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.faiss = None
        self._use_faiss = False
        self._load_faiss()

        # In-memory storage
        self.hot_index = None   # HNSW for recent
        self.warm_index = None  # IVFFlat for medium
        self.cold_index = None  # IVFFlat for old

        # ID mappings: faiss_idx -> memory_id
        self.hot_ids: List[str] = []
        self.warm_ids: List[str] = []
        self.cold_ids: List[str] = []

        # Simple flat store as fallback / primary
        self.vectors: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, datetime] = {}

        self._init_indices()
        self._load_state()

    def _load_faiss(self):
        try:
            import faiss
            self.faiss = faiss
            self._use_faiss = True
            print("  ✓ FAISS loaded successfully")
        except ImportError:
            print("  ⚠ FAISS not available, using NumPy fallback")
            self._use_faiss = False

    def _init_indices(self):
        if self._use_faiss:
            # Hot tier: HNSW (fast, high recall for recent memories)
            self.hot_index = self.faiss.IndexHNSWFlat(self.dimension, 32)
            self.hot_index.hnsw.efSearch = 64
            self.hot_index.hnsw.efConstruction = 128

            # Warm tier: IVFFlat with scalar quantizer (4x compression)
            # Will be trained once we have enough warm vectors
            self.warm_quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.warm_index = None  # Initialized when first warm vectors added

            # Cold tier: IVFFlat with PQ (8-16x compression)
            self.cold_quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.cold_index = None  # Initialized when first cold vectors added

    def _ensure_warm_index(self, training_vectors: np.ndarray = None):
        """Initialize warm tier IVF-SQ8 index if not already created."""
        if self._use_faiss and self.warm_index is None:
            nlist = max(4, min(int(len(self.warm_ids) ** 0.5), 64))
            # IVF with scalar quantizer (SQ8) — 4x compression
            self.warm_index = self.faiss.IndexIVFScalarQuantizer(
                self.warm_quantizer, self.dimension, nlist,
                self.faiss.ScalarQuantizer.QT_8bit
            )
            if training_vectors is not None and len(training_vectors) >= nlist:
                self.warm_index.train(training_vectors)
                self.warm_index.nprobe = max(2, nlist // 4)

    def _ensure_cold_index(self, training_vectors: np.ndarray = None):
        """Initialize cold tier IVF-PQ index if not already created."""
        if self._use_faiss and self.cold_index is None:
            nlist = max(4, min(int(len(self.cold_ids) ** 0.5), 32))
            # IVF with product quantizer (PQ) — 8-16x compression
            m = min(48, self.dimension)  # Number of sub-quantizers (must divide dimension)
            while self.dimension % m != 0 and m > 1:
                m -= 1
            self.cold_index = self.faiss.IndexIVFPQ(
                self.cold_quantizer, self.dimension, nlist, m, 8
            )
            if training_vectors is not None and len(training_vectors) >= nlist:
                self.cold_index.train(training_vectors)
                self.cold_index.nprobe = max(2, nlist // 4)

    def migrate_tiers(self):
        """
        Migrate vectors between hot/warm/cold tiers based on age.
        - Hot: < 30 days
        - Warm: 30 days - 1 year
        - Cold: > 1 year
        Called periodically (e.g., daily or on startup).
        """
        if not self._use_faiss:
            return

        now = datetime.now()
        hot_cutoff = now - timedelta(days=30)
        warm_cutoff = now - timedelta(days=365)

        to_warm = []
        to_cold = []

        for mid, ts in list(self.timestamps.items()):
            if ts < warm_cutoff and mid not in self.cold_ids:
                to_cold.append(mid)
            elif ts < hot_cutoff and mid not in self.warm_ids and mid not in self.cold_ids:
                to_warm.append(mid)

        if to_warm:
            warm_vecs = np.array([self.vectors[mid] for mid in to_warm if mid in self.vectors], dtype=np.float32)
            if len(warm_vecs) >= 4:
                self._ensure_warm_index(warm_vecs)
                if self.warm_index is not None and self.warm_index.is_trained:
                    self.warm_index.add(warm_vecs)
                    self.warm_ids.extend(to_warm[:len(warm_vecs)])
                    print(f"  📦 Migrated {len(warm_vecs)} vectors to warm tier")

        if to_cold:
            cold_vecs = np.array([self.vectors[mid] for mid in to_cold if mid in self.vectors], dtype=np.float32)
            if len(cold_vecs) >= 4:
                self._ensure_cold_index(cold_vecs)
                if self.cold_index is not None and self.cold_index.is_trained:
                    self.cold_index.add(cold_vecs)
                    self.cold_ids.extend(to_cold[:len(cold_vecs)])
                    print(f"  🧊 Migrated {len(cold_vecs)} vectors to cold tier")

    def add(self, memory_id: str, embedding: np.ndarray, timestamp: Optional[datetime] = None):
        """Add a vector to the store."""
        if timestamp is None:
            timestamp = datetime.now()

        embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        self.vectors[memory_id] = embedding.flatten()
        self.timestamps[memory_id] = timestamp

        if self._use_faiss and self.hot_index is not None:
            self.hot_index.add(embedding)
            self.hot_ids.append(memory_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 20,
               time_start: Optional[datetime] = None,
               time_end: Optional[datetime] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors across all tiers. Returns list of (memory_id, score)."""
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        results = []

        if self._use_faiss:
            # Search hot tier (HNSW — highest recall, lowest latency)
            if self.hot_index is not None and self.hot_index.ntotal > 0:
                k = min(top_k * 2, self.hot_index.ntotal)
                distances, indices = self.hot_index.search(query_embedding, k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.hot_ids):
                        continue
                    mid = self.hot_ids[idx]
                    score = 1.0 / (1.0 + float(dist))
                    results.append((mid, score))

            # Search warm tier (IVF-SQ8 — moderate recall)
            if self.warm_index is not None and self.warm_index.ntotal > 0:
                k_warm = min(top_k, self.warm_index.ntotal)
                try:
                    distances, indices = self.warm_index.search(query_embedding, k_warm)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(self.warm_ids):
                            continue
                        mid = self.warm_ids[idx]
                        score = 1.0 / (1.0 + float(dist)) * 0.95  # Slight discount for warm
                        results.append((mid, score))
                except Exception:
                    pass

            # Search cold tier (IVF-PQ — archive recall)
            if self.cold_index is not None and self.cold_index.ntotal > 0:
                k_cold = min(top_k // 2, self.cold_index.ntotal)
                try:
                    distances, indices = self.cold_index.search(query_embedding, k_cold)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(self.cold_ids):
                            continue
                        mid = self.cold_ids[idx]
                        score = 1.0 / (1.0 + float(dist)) * 0.90  # Slight discount for cold
                        results.append((mid, score))
                except Exception:
                    pass
        else:
            # NumPy fallback: brute-force cosine similarity
            for mid, vec in self.vectors.items():
                sim = float(np.dot(query_embedding.flatten(), vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vec) + 1e-10
                ))
                results.append((mid, sim))

        # Time filtering
        if time_start or time_end:
            filtered = []
            for mid, score in results:
                ts = self.timestamps.get(mid)
                if ts:
                    if time_start and ts < time_start:
                        continue
                    if time_end and ts > time_end:
                        continue
                filtered.append((mid, score))
            results = filtered

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, memory_id: str):
        """Remove a vector (marks as deleted, rebuilt on next compact)."""
        self.vectors.pop(memory_id, None)
        self.timestamps.pop(memory_id, None)

    def count(self) -> int:
        return len(self.vectors)

    def save(self):
        """Persist state to disk."""
        state = {
            "hot_ids": self.hot_ids,
            "warm_ids": self.warm_ids,
            "cold_ids": self.cold_ids,
            "timestamps": {k: v.isoformat() for k, v in self.timestamps.items()},
        }
        with open(os.path.join(self.data_dir, "vector_state.json"), "w") as f:
            json.dump(state, f)

        # Save vectors as numpy
        if self.vectors:
            ids = list(self.vectors.keys())
            vecs = np.array([self.vectors[i] for i in ids], dtype=np.float32)
            np.save(os.path.join(self.data_dir, "vectors.npy"), vecs)
            with open(os.path.join(self.data_dir, "vector_ids.json"), "w") as f:
                json.dump(ids, f)

        # Save FAISS index
        if self._use_faiss and self.hot_index is not None and self.hot_index.ntotal > 0:
            self.faiss.write_index(self.hot_index, os.path.join(self.data_dir, "hot.index"))

        print(f"  ✓ Vector store saved ({self.count()} vectors)")

    def _load_state(self):
        """Load state from disk."""
        state_path = os.path.join(self.data_dir, "vector_state.json")
        ids_path = os.path.join(self.data_dir, "vector_ids.json")
        vecs_path = os.path.join(self.data_dir, "vectors.npy")

        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.hot_ids = state.get("hot_ids", [])
            self.warm_ids = state.get("warm_ids", [])
            self.cold_ids = state.get("cold_ids", [])
            self.timestamps = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("timestamps", {}).items()
            }

        if os.path.exists(ids_path) and os.path.exists(vecs_path):
            with open(ids_path) as f:
                ids = json.load(f)
            vecs = np.load(vecs_path)
            for i, mid in enumerate(ids):
                self.vectors[mid] = vecs[i]

            # Rebuild FAISS index
            if self._use_faiss and self.hot_index is not None and len(vecs) > 0:
                self.hot_index.add(vecs)
                self.hot_ids = ids[:]

            print(f"  ✓ Vector store loaded ({len(ids)} vectors)")

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.count(),
            "hot_count": len(self.hot_ids),
            "warm_count": len(self.warm_ids),
            "cold_count": len(self.cold_ids),
            "using_faiss": self._use_faiss,
            "dimension": self.dimension,
        }
