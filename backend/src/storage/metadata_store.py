"""
DuckDB Metadata Store for Cortex Lab
Relational storage for structured memory metadata, temporal queries, and SQL filtering.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from src.models import CausalMemoryObject, MemoryType, EmotionLabel, BeliefDelta, EntityNode, GraphEdge


class MetadataStore:
    """
    DuckDB-backed metadata storage for memory objects.
    Stores all structured metadata (timestamps, types, entities, etc.)
    Falls back to in-memory dict store if DuckDB is unavailable.
    """

    def __init__(self, db_path: str = "data/cortex.duckdb"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        self.conn = None
        self._use_duckdb = False
        self._fallback: Dict[str, Dict] = {}  # memory_id -> dict

        if HAS_DUCKDB:
            try:
                self.conn = duckdb.connect(db_path)
                self._use_duckdb = True
                self._init_tables()
                print(f"  ✓ DuckDB metadata store: {db_path}")
            except Exception as e:
                print(f"  ⚠ DuckDB failed ({e}), using in-memory fallback")
                self._use_duckdb = False
        else:
            print("  ⚠ DuckDB not installed, using in-memory fallback")

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type VARCHAR DEFAULT 'episodic',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emotion VARCHAR DEFAULT 'neutral',
                emotion_confidence FLOAT DEFAULT 0.0,
                importance FLOAT DEFAULT 0.5,
                topics JSON DEFAULT '[]',
                entities JSON DEFAULT '[]',
                entity_ids JSON DEFAULT '[]',
                causes JSON DEFAULT '[]',
                effects JSON DEFAULT '[]',
                causal_description TEXT DEFAULT '',
                context_prefix TEXT DEFAULT '',
                propositions JSON DEFAULT '[]',
                raptor_level INTEGER DEFAULT 0,
                raptor_children JSON DEFAULT '[]',
                session_id VARCHAR DEFAULT '',
                source VARCHAR DEFAULT 'chat',
                metadata JSON DEFAULT '{}'
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS belief_deltas (
                id VARCHAR PRIMARY KEY,
                topic VARCHAR NOT NULL,
                old_belief_id VARCHAR,
                new_belief_id VARCHAR,
                old_belief_text TEXT,
                new_belief_text TEXT,
                change_type VARCHAR DEFAULT 'new_belief',
                confidence FLOAT DEFAULT 0.0,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evidence_chain JSON DEFAULT '[]'
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id VARCHAR PRIMARY KEY,
                canonical_name VARCHAR NOT NULL,
                aliases JSON DEFAULT '[]',
                entity_type VARCHAR DEFAULT 'unknown',
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                memory_ids JSON DEFAULT '[]',
                attributes JSON DEFAULT '{}'
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id VARCHAR,
                target_id VARCHAR,
                relation VARCHAR,
                weight FLOAT DEFAULT 1.0,
                memory_ids JSON DEFAULT '[]',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id, relation)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                content TEXT NOT NULL,
                thinking TEXT DEFAULT '',
                memory_id VARCHAR DEFAULT '',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON DEFAULT '{}'
            )
        """)

    # ─── Memory CRUD ─────────────────────────────────────────────────────

    def store_memory(self, memory: CausalMemoryObject):
        """Store a memory object."""
        if self._use_duckdb:
            self.conn.execute("""
                INSERT OR REPLACE INTO memories VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.timestamp,
                memory.emotion.value,
                memory.emotion_confidence,
                memory.importance,
                json.dumps(memory.topics),
                json.dumps(memory.entities),
                json.dumps(memory.entity_ids),
                json.dumps(memory.causes),
                json.dumps(memory.effects),
                memory.causal_description,
                memory.context_prefix,
                json.dumps(memory.propositions),
                memory.raptor_level,
                json.dumps(memory.raptor_children),
                memory.session_id,
                memory.source,
                json.dumps(memory.metadata),
            ])
        else:
            self._fallback[memory.id] = memory.to_dict()

    def get_memory(self, memory_id: str) -> Optional[CausalMemoryObject]:
        """Retrieve a memory by ID."""
        if self._use_duckdb:
            result = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?", [memory_id]
            ).fetchone()
            if result:
                return self._row_to_memory(result)
            return None
        else:
            data = self._fallback.get(memory_id)
            return CausalMemoryObject.from_dict(data) if data else None

    def get_memories(self, memory_ids: List[str]) -> List[CausalMemoryObject]:
        """Retrieve multiple memories by IDs."""
        if not memory_ids:
            return []
        results = []
        for mid in memory_ids:
            mem = self.get_memory(mid)
            if mem:
                results.append(mem)
        return results

    def search_by_time(self, start: Optional[datetime] = None,
                       end: Optional[datetime] = None,
                       limit: int = 50) -> List[CausalMemoryObject]:
        """Search memories by time range."""
        if self._use_duckdb:
            conditions = []
            params = []
            if start:
                conditions.append("timestamp >= ?")
                params.append(start)
            if end:
                conditions.append("timestamp <= ?")
                params.append(end)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = self.conn.execute(
                f"SELECT * FROM memories {where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        else:
            memories = [CausalMemoryObject.from_dict(d) for d in self._fallback.values()]
            if start:
                memories = [m for m in memories if m.timestamp >= start]
            if end:
                memories = [m for m in memories if m.timestamp <= end]
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories[:limit]

    def search_by_topic(self, topic: str, limit: int = 20) -> List[CausalMemoryObject]:
        """Search memories by topic."""
        if self._use_duckdb:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE topics LIKE ? ORDER BY timestamp DESC LIMIT ?",
                [f'%"{topic}"%', limit]
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        else:
            results = []
            for d in self._fallback.values():
                if topic.lower() in str(d.get("topics", [])).lower():
                    results.append(CausalMemoryObject.from_dict(d))
            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit]

    def search_by_entity(self, entity: str, limit: int = 20) -> List[CausalMemoryObject]:
        """Search memories mentioning an entity."""
        if self._use_duckdb:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE entities LIKE ? ORDER BY timestamp DESC LIMIT ?",
                [f'%"{entity}"%', limit]
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        else:
            results = []
            for d in self._fallback.values():
                if entity.lower() in str(d.get("entities", [])).lower():
                    results.append(CausalMemoryObject.from_dict(d))
            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit]

    def get_all_memories(self, limit: int = 100, offset: int = 0) -> List[CausalMemoryObject]:
        """Get all memories with pagination."""
        if self._use_duckdb:
            rows = self.conn.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                [limit, offset]
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]
        else:
            memories = [CausalMemoryObject.from_dict(d) for d in self._fallback.values()]
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories[offset:offset + limit]

    def count_memories(self) -> int:
        if self._use_duckdb:
            return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return len(self._fallback)

    def delete_memory(self, memory_id: str):
        if self._use_duckdb:
            self.conn.execute("DELETE FROM memories WHERE id = ?", [memory_id])
        else:
            self._fallback.pop(memory_id, None)

    # ─── Conversation Storage ────────────────────────────────────────────

    def store_conversation_turn(self, session_id: str, role: str, content: str,
                                 thinking: str = "", memory_id: str = ""):
        """Store a conversation turn."""
        turn_id = f"{session_id}-{role}-{datetime.now().timestamp()}"
        if self._use_duckdb:
            self.conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [turn_id, session_id, role, content, thinking, memory_id,
                 datetime.now(), json.dumps({})]
            )

    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get all turns in a conversation."""
        if self._use_duckdb:
            rows = self.conn.execute(
                "SELECT role, content, thinking, timestamp FROM conversations WHERE session_id = ? ORDER BY timestamp",
                [session_id]
            ).fetchall()
            return [{"role": r[0], "content": r[1], "thinking": r[2], "timestamp": str(r[3])} for r in rows]
        return []

    # ─── Entity Storage ──────────────────────────────────────────────────

    def store_entity(self, entity: EntityNode):
        if self._use_duckdb:
            self.conn.execute("""
                INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                entity.id, entity.canonical_name, json.dumps(entity.aliases),
                entity.entity_type, entity.first_seen, entity.last_seen,
                json.dumps(entity.memory_ids), json.dumps(entity.attributes)
            ])

    def get_entities(self, limit: int = 100) -> List[Dict]:
        if self._use_duckdb:
            rows = self.conn.execute(
                "SELECT * FROM entities ORDER BY last_seen DESC LIMIT ?", [limit]
            ).fetchall()
            return [{
                "id": r[0], "canonical_name": r[1],
                "aliases": json.loads(r[2]) if isinstance(r[2], str) else r[2],
                "entity_type": r[3],
                "first_seen": str(r[4]), "last_seen": str(r[5]),
                "memory_ids": json.loads(r[6]) if isinstance(r[6], str) else r[6],
            } for r in rows]
        return []

    # ─── Belief Deltas ───────────────────────────────────────────────────

    def store_belief_delta(self, delta: BeliefDelta):
        if self._use_duckdb:
            self.conn.execute("""
                INSERT OR REPLACE INTO belief_deltas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                delta.id, delta.topic, delta.old_belief_id, delta.new_belief_id,
                delta.old_belief_text, delta.new_belief_text,
                delta.change_type.value, delta.confidence,
                delta.detected_at, json.dumps(delta.evidence_chain)
            ])

    def get_belief_deltas(self, topic: Optional[str] = None, limit: int = 50) -> List[Dict]:
        if self._use_duckdb:
            if topic:
                rows = self.conn.execute(
                    "SELECT * FROM belief_deltas WHERE topic LIKE ? ORDER BY detected_at DESC LIMIT ?",
                    [f"%{topic}%", limit]
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM belief_deltas ORDER BY detected_at DESC LIMIT ?", [limit]
                ).fetchall()
            return [{
                "id": r[0], "topic": r[1], "old_belief_text": r[4],
                "new_belief_text": r[5], "change_type": r[6],
                "confidence": r[7], "detected_at": str(r[8])
            } for r in rows]
        return []

    # ─── Graph Edges ─────────────────────────────────────────────────────

    def store_edge(self, edge: GraphEdge):
        if self._use_duckdb:
            self.conn.execute("""
                INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)
            """, [
                edge.source_id, edge.target_id, edge.relation,
                edge.weight, json.dumps(edge.memory_ids), edge.timestamp
            ])

    def get_edges(self, entity_id: Optional[str] = None) -> List[Dict]:
        if self._use_duckdb:
            if entity_id:
                rows = self.conn.execute(
                    "SELECT * FROM graph_edges WHERE source_id = ? OR target_id = ?",
                    [entity_id, entity_id]
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM graph_edges").fetchall()
            return [{
                "source_id": r[0], "target_id": r[1], "relation": r[2],
                "weight": r[3], "memory_ids": json.loads(r[4]) if isinstance(r[4], str) else r[4],
            } for r in rows]
        return []

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        if self._use_duckdb:
            mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            entity_count = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            edge_count = self.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
            belief_count = self.conn.execute("SELECT COUNT(*) FROM belief_deltas").fetchone()[0]
            conv_count = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            return {
                "memories": mem_count,
                "entities": entity_count,
                "edges": edge_count,
                "belief_deltas": belief_count,
                "conversations": conv_count,
                "backend": "duckdb",
            }
        return {
            "memories": len(self._fallback),
            "backend": "in-memory",
        }

    # ─── Internal ────────────────────────────────────────────────────────

    def _row_to_memory(self, row) -> CausalMemoryObject:
        """Convert a DuckDB row tuple to CausalMemoryObject."""
        return CausalMemoryObject(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]) if row[2] else MemoryType.EPISODIC,
            timestamp=row[3] if isinstance(row[3], datetime) else datetime.fromisoformat(str(row[3])),
            emotion=EmotionLabel(row[4]) if row[4] else EmotionLabel.NEUTRAL,
            emotion_confidence=float(row[5] or 0),
            importance=float(row[6] or 0.5),
            topics=json.loads(row[7]) if isinstance(row[7], str) else (row[7] or []),
            entities=json.loads(row[8]) if isinstance(row[8], str) else (row[8] or []),
            entity_ids=json.loads(row[9]) if isinstance(row[9], str) else (row[9] or []),
            causes=json.loads(row[10]) if isinstance(row[10], str) else (row[10] or []),
            effects=json.loads(row[11]) if isinstance(row[11], str) else (row[11] or []),
            causal_description=row[12] or "",
            context_prefix=row[13] or "",
            propositions=json.loads(row[14]) if isinstance(row[14], str) else (row[14] or []),
            raptor_level=int(row[15] or 0),
            raptor_children=json.loads(row[16]) if isinstance(row[16], str) else (row[16] or []),
            session_id=row[17] or "",
            source=row[18] or "chat",
            metadata=json.loads(row[19]) if isinstance(row[19], str) else (row[19] or {}),
        )

    def close(self):
        if self.conn:
            self.conn.close()
