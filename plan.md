# EdgeMemory: Comprehensive Implementation Plan

## Executive Summary

EdgeMemory is a **laptop-buildable, fully open-source personal causal intelligence system** that treats your entire life stream as training data for a personal brain model. Unlike shallow chatbot memories, this system builds a **structured, time-aware, causal knowledge graph** of your decisions, emotions, and reflections—all running locally on your device.

---

## Part 1: Core Vision & Philosophy

### 1.1 The Fundamental Problem

Current AI assistants suffer from four critical limitations:

| Limitation | Impact | EdgeMemory Solution |
|------------|--------|---------------------|
| **No deep personal grounding** | LLMs know fragments of you from short chats; your beliefs, doubts, and goals never become stable memory | Continuous collection into structured, persistent memory |
| **Context-window forgetting** | Important discussions are pushed out; past reasoning is lost | Infinite temporal memory with compression |
| **Weak non-temporal memory** | Current "memory" = unstructured notes, no understanding of *when* or *how thinking evolved* | Time-aware sequence modeling with LSTM/GRU + attention |
| **Cloud-first, privacy-weak** | Server-side logs expose personal data | 100% on-device with quantization, pruning, ONNX |

### 1.2 The Core Insight

> **Memory ≠ Logging. Memory = Intentional Brain Duplication.**

You feed your internal state into a personal model, then use it to:
- Analyze your own limitations and blind spots
- Track belief evolution over decades
- Surface causal patterns you can't consciously track
- Answer queries like *"What was my thought process about dropping out of college in my 5th semester?"*

### 1.3 The Non-Negotiable Design Principle

```
┌─────────────────────────────────────────────────────────────────┐
│  LLM is NOT the brain.                                          │
│  LLM is a REASONING LENS over structured, evolving memory.      │
└─────────────────────────────────────────────────────────────────┘
```

**Constraints:**
- Memory must be **explicit, inspectable, and causal**
- Reasoning must be **local and controllable**
- Retrieval must be **agentic and multi-step**, not top-K similarity
- The system must explain **why**, not just **what**

---

## Part 2: System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EDGEMEMORY SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   INPUT     │     │              INGESTION PIPELINE                 │   │
│  │  ─────────  │     │  ┌─────────┐  ┌──────────┐  ┌───────────────┐  │   │
│  │  Voice      │────▶│  │ Whisper │─▶│ Memory   │─▶│ Embedding +   │  │   │
│  │  Text       │     │  │  (ASR)  │  │ Builder  │  │ Classification│  │   │
│  │  Reflection │     │  └─────────┘  └──────────┘  └───────────────┘  │   │
│  └─────────────┘     └──────────────────────┬──────────────────────────┘   │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        STORAGE LAYER                                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │  │
│  │  │ Vector DB  │  │ Relational │  │  Causal    │  │   Knowledge    │  │  │
│  │  │  (FAISS/   │  │    DB      │  │   Graph    │  │     Graph      │  │  │
│  │  │  Qdrant)   │  │ (DuckDB)   │  │   Links    │  │   (Triples)    │  │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MULTI-CHANNEL RETRIEVAL                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │  │
│  │  │ Dense   │  │ Sparse  │  │  Graph  │  │   SQL   │  │ Re-ranker │  │  │
│  │  │ (bge/e5)│  │ (BM25)  │  │ Traverse│  │ Filter  │  │ (Fusion)  │  │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         AGENT LAYER                                   │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │  │
│  │  │ Timeline │ │ Cause-   │ │Reflection│ │ Planning │ │ Arbitration│ │  │
│  │  │  Agent   │ │ Effect   │ │  Agent   │ │  Agent   │ │   Agent    │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                             │                               │
│                                             ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    LOCAL LLM (Reasoning Only)                         │  │
│  │            Mistral 7B / Qwen2.5 7B / Phi-3 Mini                       │  │
│  │                    via Ollama / llama.cpp                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Overview

**Ingestion Loop (Memory Creation):**
```
INPUT (voice/text) → Whisper ASR → Memory Event Builder → Embedding Generation
                                                              ↓
                     Memory Evolution Check ← Storage (Vector + SQL + Graph)
                                                              ↓
                                              Store + Link + Update
```

**Query Loop (Reasoning):**
```
Query → Intent Detection → Multi-Channel Retrieval → Fusion + Re-ranking
                                                          ↓
             Answer + Evidence ← LLM Reasoning ← Temporal Abstraction
                                                          ↑
                                              Conflict Resolution
```

---

## Part 3: Detailed Implementation Plan

### Phase 1: Foundation Infrastructure (Week 1-2)

#### Step 1.1: Project Setup & Environment

**Objective:** Create the development environment with all dependencies.

```
DL_Final_Project/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── asr.py              # Whisper integration
│   │   ├── memory_builder.py   # Memory event construction
│   │   ├── classifier.py       # Lightweight fine-tuned classifier (NEW)
│   │   ├── entity_extractor.py # Entity extraction
│   │   └── entity_resolver.py  # Entity resolution & linking (NEW)
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py     # FAISS/Qdrant wrapper
│   │   ├── relational_db.py    # DuckDB/SQLite wrapper
│   │   ├── knowledge_graph.py  # Triple storage
│   │   ├── causal_graph.py     # Cause-effect links
│   │   ├── belief_tracker.py   # Belief evolution tracking
│   │   └── memory_consolidator.py  # Hierarchical summarization (NEW)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense_retriever.py  # Embedding-based
│   │   ├── sparse_retriever.py # BM25
│   │   ├── graph_retriever.py  # Graph traversal
│   │   ├── sql_retriever.py    # Exact filters
│   │   └── fusion.py           # Re-ranking & fusion
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Agent interface
│   │   ├── orchestrator.py     # Agent orchestration & chaining (NEW)
│   │   ├── timeline_agent.py
│   │   ├── causal_agent.py
│   │   ├── reflection_agent.py
│   │   ├── planning_agent.py
│   │   └── arbitration_agent.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── local_llm.py        # Ollama/llama.cpp interface
│   │   └── prompts.py          # Prompt templates
│   ├── models/
│   │   ├── __init__.py
│   │   ├── memory_types.py     # Memory data models
│   │   ├── embeddings.py       # Embedding model wrapper
│   │   └── classifiers/        # Fine-tuned classifier models (NEW)
│   │       ├── memory_type_classifier.py
│   │       ├── emotion_classifier.py
│   │       └── importance_scorer.py
│   ├── evaluation/             # Evaluation framework (NEW)
│   │   ├── __init__.py
│   │   ├── synthetic_dataset.py
│   │   ├── metrics.py
│   │   └── regression_tests.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
├── data/
│   ├── memories.db             # DuckDB database
│   ├── vectors/                # FAISS indices
│   ├── audio/                  # Raw audio storage (optional)
│   ├── summaries/              # Hierarchical summaries (NEW)
│   └── evaluation/             # Evaluation datasets (NEW)
├── models/                     # Fine-tuned classifier weights (NEW)
│   ├── memory_type_classifier.pt
│   ├── emotion_classifier.pt
│   └── importance_scorer.pt
├── tests/
├── notebooks/                  # Experimentation
├── configs/
│   └── config.yaml
├── requirements.txt
├── plan.md
└── README.md
```

**Dependencies (requirements.txt):**
```
# Core
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0

# ASR
openai-whisper>=20230918
faster-whisper>=0.9.0

# Embeddings
faiss-cpu>=1.7.4
qdrant-client>=1.7.0

# Database
duckdb>=0.9.0
sqlalchemy>=2.0.0

# Sparse Retrieval
rank-bm25>=0.2.2
whoosh>=2.7.4

# Knowledge Graph
networkx>=3.0

# LLM Interface
ollama>=0.1.0

# Lightweight Classifiers (NEW - for classification bottleneck fix)
setfit>=1.0.0                   # Few-shot fine-tuning
scikit-learn>=1.3.0             # Traditional ML classifiers
joblib>=1.3.0                   # Model serialization

# Entity Resolution (NEW)
rapidfuzz>=3.0.0                # Fuzzy string matching
spacy>=3.7.0                    # NER fallback

# Compression/Summarization (NEW)
tiktoken>=0.5.0                 # Token counting
langchain-text-splitters>=0.0.1 # Chunking utilities

# Evaluation (NEW)
pytest>=7.4.0
pytest-benchmark>=4.0.0

# Utilities
pydantic>=2.0.0
pyyaml>=6.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.66.0
```

**Tasks:**
- [ ] Initialize Git repository
- [ ] Create virtual environment
- [ ] Install all dependencies
- [ ] Set up configuration management (YAML-based)
- [ ] Create logging infrastructure
- [ ] Write basic test scaffolding

---

#### Step 1.2: Define Core Data Models

**Objective:** Create the foundational data structures for memories.

**File: `src/models/memory_types.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from uuid import uuid4

class CausalMemoryObject(BaseModel):
    """The core primitive for all memories."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    content: str
    type: Literal["episodic", "semantic", "procedural", "reflective"]
    
    # Classification metadata
    topic: Optional[str] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion: Optional[str] = None
    
    # Causal links
    causes: List[str] = Field(default_factory=list)  # event_ids
    effects: List[str] = Field(default_factory=list)  # event_ids
    
    # Evolution tracking
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    supersedes: Optional[str] = None  # event_id of older version
    superseded_by: Optional[str] = None  # event_id of newer version
    
    # Entity extraction
    entities: List[str] = Field(default_factory=list)  # people, projects, topics
    
    # Embedding (stored separately in vector DB)
    embedding_id: Optional[str] = None

class MemoryQuery(BaseModel):
    """Structured query for memory retrieval."""
    raw_query: str
    intent: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    memory_types: List[str] = Field(default_factory=list)
    abstraction_level: Literal["specific", "summary", "pattern"] = "specific"

class BeliefDelta(BaseModel):
    """Tracks how a belief/understanding changed over time."""
    topic: str
    old_belief_id: str
    new_belief_id: str
    change_timestamp: datetime
    cause_event_ids: List[str] = Field(default_factory=list)
    change_type: Literal["contradiction", "refinement", "expansion", "abandonment"]
```

**Memory Types Explained:**

| Type | Description | Example | Decay Rate |
|------|-------------|---------|------------|
| **Episodic** | What happened (events, conversations) | "Had a meeting with John about Project X" | Medium |
| **Semantic** | What you learned (facts, insights) | "Transformers use self-attention mechanism" | Low |
| **Procedural** | How you do things (habits, methods) | "I always review code before committing" | Very Low |
| **Reflective** | Why you changed (meta-cognition) | "I realized I was avoiding hard problems" | Low |

---

### Phase 2: Ingestion Pipeline (Week 2-3)

#### Step 2.1: Speech-to-Text Module

**Objective:** Convert voice input to text using local Whisper.

**File: `src/ingestion/asr.py`**

```python
from faster_whisper import WhisperModel
from pathlib import Path
from typing import Optional
import logging

class LocalASR:
    """Local speech-to-text using faster-whisper."""
    
    def __init__(
        self, 
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize Whisper model.
        
        Args:
            model_size: tiny, base, small, medium, large-v2
            device: cpu or cuda
            compute_type: int8, float16, float32
        """
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        self.logger = logging.getLogger(__name__)
    
    def transcribe(
        self, 
        audio_path: str,
        language: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio file to text.
        
        Returns:
            {
                "text": str,
                "segments": List[{start, end, text}],
                "language": str
            }
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5
        )
        
        segments_list = []
        full_text = []
        
        for segment in segments:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())
        
        return {
            "text": " ".join(full_text),
            "segments": segments_list,
            "language": info.language
        }
    
    def transcribe_stream(self, audio_stream):
        """For real-time transcription (future implementation)."""
        raise NotImplementedError("Streaming ASR not yet implemented")
```

**Tasks:**
- [ ] Implement `LocalASR` class
- [ ] Test with sample audio files
- [ ] Benchmark transcription speed on target hardware
- [ ] Add support for streaming/real-time (stretch goal)

---

#### Step 2.2: Lightweight Classifiers (Critical Fix: Classification Bottleneck)

**Problem:** Using LLM for every memory classification is:
- Slow (2-5 seconds per classification on CPU)
- Energy-draining on mobile
- A single point of failure

**Solution:** Train small fine-tuned classifiers (~50MB total) for type/emotion/importance detection. Use LLM only for complex edge cases.

**File: `src/models/classifiers/memory_type_classifier.py`**

```python
from setfit import SetFitModel
from typing import List, Tuple
import torch
import joblib
from pathlib import Path

class MemoryTypeClassifier:
    """
    Lightweight classifier for memory type detection.
    Uses SetFit (few-shot fine-tuning) for ~50MB model size.
    
    Latency: ~50ms vs ~3000ms for LLM
    """
    
    MEMORY_TYPES = ["episodic", "semantic", "procedural", "reflective"]
    CONFIDENCE_THRESHOLD = 0.7  # Below this, escalate to LLM
    
    def __init__(self, model_path: str = "models/memory_type_classifier"):
        self.model_path = Path(model_path)
        if self.model_path.exists():
            self.model = SetFitModel.from_pretrained(str(self.model_path))
        else:
            # Initialize with base model for fine-tuning
            self.model = SetFitModel.from_pretrained(
                "sentence-transformers/paraphrase-MiniLM-L3-v2"
            )
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict memory type with confidence score.
        
        Returns:
            (predicted_type, confidence)
        """
        # Get prediction probabilities
        probs = self.model.predict_proba([text])[0]
        predicted_idx = probs.argmax()
        confidence = float(probs[predicted_idx])
        
        return self.MEMORY_TYPES[predicted_idx], confidence
    
    def needs_llm_fallback(self, confidence: float) -> bool:
        """Check if confidence is too low for reliable classification."""
        return confidence < self.CONFIDENCE_THRESHOLD
    
    def fine_tune(self, texts: List[str], labels: List[str], epochs: int = 1):
        """
        Fine-tune on user's data for personalization.
        Can be run periodically with corrected labels.
        """
        from setfit import SetFitTrainer
        
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset={"text": texts, "label": labels},
            num_iterations=20,
            num_epochs=epochs
        )
        trainer.train()
        self.model.save_pretrained(str(self.model_path))
    
    @classmethod
    def create_training_data(cls) -> Tuple[List[str], List[str]]:
        """Generate synthetic training data for initial model."""
        examples = {
            "episodic": [
                "I had a meeting with John today about the project",
                "Went to the coffee shop and ran into an old friend",
                "The presentation went well, got positive feedback",
                "Had dinner with family, discussed vacation plans",
            ],
            "semantic": [
                "Transformers use self-attention mechanisms",
                "The capital of France is Paris",
                "Neural networks learn through backpropagation",
                "REST APIs use HTTP methods for communication",
            ],
            "procedural": [
                "I always review my code before committing",
                "My morning routine starts with meditation",
                "When debugging, I first check the logs",
                "I handle stress by taking short walks",
            ],
            "reflective": [
                "I realized I was avoiding difficult conversations",
                "Looking back, I should have started earlier",
                "I've noticed I learn better with hands-on practice",
                "My fear of failure was holding me back",
            ],
        }
        texts, labels = [], []
        for label, examples_list in examples.items():
            texts.extend(examples_list)
            labels.extend([label] * len(examples_list))
        return texts, labels


class EmotionClassifier:
    """Lightweight emotion classifier using DistilBERT."""
    
    EMOTIONS = ["neutral", "happy", "sad", "anxious", "excited", "frustrated", "angry", "confused"]
    
    def __init__(self, model_path: str = "models/emotion_classifier"):
        from transformers import pipeline
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict emotion with confidence."""
        results = self.classifier(text[:512])[0]  # Truncate for model
        top_result = max(results, key=lambda x: x['score'])
        return top_result['label'].lower(), top_result['score']


class ImportanceScorer:
    """
    Rule-based + ML hybrid for importance scoring.
    Fast heuristics with ML refinement.
    """
    
    IMPORTANCE_KEYWORDS = {
        "high": ["decided", "realized", "important", "critical", "breakthrough", 
                 "finally", "major", "significant", "promise", "commit"],
        "medium": ["learned", "discussed", "meeting", "worked", "progress"],
        "low": ["random", "casual", "just", "quick", "minor"]
    }
    
    def __init__(self):
        self.keyword_weights = {"high": 0.8, "medium": 0.5, "low": 0.2}
    
    def score(self, text: str) -> float:
        """
        Score importance from 0.0 to 1.0.
        Uses keyword matching + text length heuristics.
        """
        text_lower = text.lower()
        
        # Keyword-based scoring
        score = 0.5  # Default
        for level, keywords in self.IMPORTANCE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                score = self.keyword_weights[level]
                break
        
        # Length adjustment (longer = potentially more important)
        word_count = len(text.split())
        if word_count > 100:
            score = min(1.0, score + 0.1)
        elif word_count < 10:
            score = max(0.1, score - 0.1)
        
        return round(score, 2)
```

---

#### Step 2.3: Memory Event Builder (Optimized)

**Objective:** Convert raw text into structured memory objects using lightweight classifiers with LLM fallback.

**File: `src/ingestion/memory_builder.py`**

```python
from datetime import datetime
from typing import Optional, List
from src.models.memory_types import CausalMemoryObject
from src.models.classifiers.memory_type_classifier import (
    MemoryTypeClassifier, EmotionClassifier, ImportanceScorer
)
from src.llm.local_llm import LocalLLM
import json
import logging

class MemoryEventBuilder:
    """
    Build structured memory objects from raw text.
    
    OPTIMIZATION: Uses lightweight classifiers (~50ms) for routine classification.
    Falls back to LLM (~3s) only for edge cases with low confidence.
    """
    
    def __init__(
        self, 
        llm: LocalLLM,
        use_lightweight_classifiers: bool = True
    ):
        self.llm = llm
        self.use_lightweight = use_lightweight_classifiers
        self.logger = logging.getLogger(__name__)
        
        # Initialize lightweight classifiers
        if use_lightweight_classifiers:
            self.type_classifier = MemoryTypeClassifier()
            self.emotion_classifier = EmotionClassifier()
            self.importance_scorer = ImportanceScorer()
        
        # LLM fallback prompt for complex cases
        self.llm_classification_prompt = """
        Analyze the following text and extract structured information.
        The lightweight classifier was uncertain. Please provide accurate classification.
        
        Text: {text}
        
        Return a JSON object with:
        {{
            "type": "episodic|semantic|procedural|reflective",
            "topic": "main topic/theme",
            "importance": 0.0-1.0 (how significant is this),
            "emotion": "neutral|happy|sad|anxious|excited|frustrated|etc",
            "entities": ["person names", "project names", "places", "concepts"],
            "intent": "what is the purpose/intent of this memory"
        }}
        
        Only respond with valid JSON.
        """
    
    def build(
        self, 
        raw_text: str, 
        timestamp: Optional[datetime] = None,
        source: str = "voice"
    ) -> CausalMemoryObject:
        """
        Build a memory object from raw text.
        
        Args:
            raw_text: The transcribed or typed text
            timestamp: When this memory occurred (defaults to now)
            source: voice, text, reflection
        
        Returns:
            CausalMemoryObject with classified metadata
        """
        timestamp = timestamp or datetime.now()
        use_llm_fallback = False
        
        if self.use_lightweight:
            # FAST PATH: Use lightweight classifiers (~50ms total)
            memory_type, type_confidence = self.type_classifier.predict(raw_text)
            emotion, emotion_confidence = self.emotion_classifier.predict(raw_text)
            importance = self.importance_scorer.score(raw_text)
            
            # Check if any classifier is uncertain
            if self.type_classifier.needs_llm_fallback(type_confidence):
                self.logger.info(f"Low confidence ({type_confidence:.2f}), using LLM fallback")
                use_llm_fallback = True
            
            if not use_llm_fallback:
                return CausalMemoryObject(
                    timestamp=timestamp,
                    content=raw_text,
                    type=memory_type,
                    topic=None,  # Topic extraction still uses LLM
                    importance=importance,
                    emotion=emotion,
                    entities=[]  # Entity extraction handled separately
                )
        
        # SLOW PATH: LLM fallback for complex cases
        prompt = self.llm_classification_prompt.format(text=raw_text)
        response = self.llm.generate(prompt)
        
        try:
            metadata = json.loads(response)
        except json.JSONDecodeError:
            metadata = {
                "type": "episodic",
                "topic": None,
                "importance": 0.5,
                "emotion": "neutral",
                "entities": [],
                "intent": None
            }
        
        return CausalMemoryObject(
            timestamp=timestamp,
            content=raw_text,
            type=metadata.get("type", "episodic"),
            topic=metadata.get("topic"),
            importance=metadata.get("importance", 0.5),
            emotion=metadata.get("emotion"),
            entities=metadata.get("entities", [])
        )
    
    def build_batch(
        self, 
        texts: List[str], 
        timestamps: Optional[List[datetime]] = None
    ) -> List[CausalMemoryObject]:
        """Build multiple memory objects efficiently."""
        timestamps = timestamps or [datetime.now()] * len(texts)
        return [
            self.build(text, ts) 
            for text, ts in zip(texts, timestamps)
        ]
    
    def get_classification_stats(self) -> dict:
        """Return statistics about classifier usage for monitoring."""
        return {
            "lightweight_calls": getattr(self, '_lightweight_count', 0),
            "llm_fallback_calls": getattr(self, '_llm_fallback_count', 0),
            "avg_lightweight_latency_ms": getattr(self, '_avg_lightweight_latency', 0),
        }
```

**Tasks:**
- [ ] Implement lightweight classifiers (SetFit-based)
- [ ] Fine-tune on synthetic data (50-100 examples per class)
- [ ] Add monitoring for fallback rate (target: <10% LLM usage)
- [ ] Test classification accuracy vs pure-LLM baseline

---

#### Step 2.4: Entity Extraction & Linking

**Objective:** Extract entities and build knowledge graph connections.

**File: `src/ingestion/entity_extractor.py`**

```python
from typing import List, Dict, Tuple
from src.llm.local_llm import LocalLLM
import re

class EntityExtractor:
    """Extract and categorize entities from memory content."""
    
    ENTITY_TYPES = ["person", "project", "topic", "place", "event", "artifact"]
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
        self.extraction_prompt = """
        Extract all named entities from the following text.
        
        Text: {text}
        
        For each entity, provide:
        - name: the entity name
        - type: person|project|topic|place|event|artifact
        - relations: any relationships mentioned (e.g., "works_on", "located_at")
        
        Return as JSON array.
        """
    
    def extract(self, text: str) -> List[Dict]:
        """
        Extract entities from text.
        
        Returns:
            List of {name, type, relations} dicts
        """
        prompt = self.extraction_prompt.format(text=text)
        response = self.llm.generate(prompt)
        
        try:
            entities = json.loads(response)
            return entities
        except:
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> List[Dict]:
        """Simple regex-based fallback for entity extraction."""
        entities = []
        
        # Simple proper noun detection
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for noun in set(proper_nouns):
            entities.append({
                "name": noun,
                "type": "unknown",
                "relations": []
            })
        
        return entities
    
    def build_relations(
        self, 
        memory_id: str,
        entities: List[Dict]
    ) -> List[Tuple[str, str, str]]:
        """
        Build (subject, predicate, object) triples for knowledge graph.
        
        Returns:
            List of (memory_id, relation, entity_name) triples
        """
        triples = []
        for entity in entities:
            # Memory -> mentions -> Entity
            triples.append((memory_id, "mentions", entity["name"]))
            
            # Add any explicit relations
            for rel in entity.get("relations", []):
                triples.append((entity["name"], rel.get("type"), rel.get("target")))
        
        return triples
```

**Tasks:**
- [ ] Implement entity extraction with LLM + spaCy fallback
- [ ] Build relation triple generator
- [ ] Test extraction accuracy

---

#### Step 2.5: Entity Resolution & Linking

**Objective:** Resolve entity mentions to canonical entities, handling coreference and name variations.

**Critical Fix:** The original plan assumed extracted entities could be directly linked. In practice, "Bob", "Robert", "my manager", and "Robert Chen" might all refer to the same person.

**File: `src/ingestion/entity_resolver.py`**

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from rapidfuzz import fuzz, process
import spacy
from datetime import datetime
import json

@dataclass
class CanonicalEntity:
    """A canonical entity in the knowledge base."""
    canonical_id: str
    canonical_name: str
    entity_type: str  # person, project, topic, place, event, artifact
    aliases: List[str] = field(default_factory=list)
    first_mention: Optional[datetime] = None
    mention_count: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def add_alias(self, alias: str):
        """Add a new alias if not already present."""
        normalized = alias.lower().strip()
        if normalized not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.mention_count += 1

class EntityResolver:
    """
    Resolve entity mentions to canonical entities.
    
    Handles:
    - Coreference resolution ("my manager" -> "Robert Chen")
    - Name variations ("Bob" -> "Robert Chen")
    - Fuzzy matching for typos
    - Entity merging when same entity discovered
    """
    
    def __init__(
        self, 
        fuzzy_threshold: float = 85.0,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Args:
            fuzzy_threshold: Minimum fuzzy match score (0-100) to consider a match
            spacy_model: spaCy model for NER and coreference
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.canonical_entities: Dict[str, CanonicalEntity] = {}
        self.alias_index: Dict[str, str] = {}  # alias -> canonical_id
        
        # Load spaCy for coreference resolution
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
    
    def resolve(
        self, 
        extracted_entities: List[Dict],
        context: Optional[str] = None
    ) -> List[Dict]:
        """
        Resolve extracted entities to canonical entities.
        
        Args:
            extracted_entities: List of {name, type, relations} from EntityExtractor
            context: Full text context for coreference resolution
        
        Returns:
            List of resolved entities with canonical_id added
        """
        resolved = []
        
        for entity in extracted_entities:
            mention = entity["name"]
            entity_type = entity.get("type", "unknown")
            
            # Step 1: Check exact alias match
            canonical_id = self._exact_match(mention)
            
            # Step 2: Try fuzzy matching
            if not canonical_id:
                canonical_id, confidence = self._fuzzy_match(mention, entity_type)
                if confidence < self.fuzzy_threshold:
                    canonical_id = None
            
            # Step 3: Create new canonical entity if no match
            if not canonical_id:
                canonical_id = self._create_canonical(mention, entity_type)
            else:
                # Add this mention as an alias
                self.canonical_entities[canonical_id].add_alias(mention)
                self._update_alias_index(mention, canonical_id)
            
            resolved.append({
                **entity,
                "canonical_id": canonical_id,
                "canonical_name": self.canonical_entities[canonical_id].canonical_name
            })
        
        return resolved
    
    def _exact_match(self, mention: str) -> Optional[str]:
        """Check for exact alias match (case-insensitive)."""
        normalized = mention.lower().strip()
        return self.alias_index.get(normalized)
    
    def _fuzzy_match(
        self, 
        mention: str, 
        entity_type: str
    ) -> Tuple[Optional[str], float]:
        """Find best fuzzy match among same-type entities."""
        candidates = [
            (cid, ce.canonical_name)
            for cid, ce in self.canonical_entities.items()
            if ce.entity_type == entity_type or entity_type == "unknown"
        ]
        
        if not candidates:
            return None, 0.0
        
        # Also check all aliases
        all_candidates = []
        for cid, ce in self.canonical_entities.items():
            if ce.entity_type == entity_type or entity_type == "unknown":
                all_candidates.append((cid, ce.canonical_name))
                for alias in ce.aliases:
                    all_candidates.append((cid, alias))
        
        if not all_candidates:
            return None, 0.0
        
        # Find best match
        names = [c[1] for c in all_candidates]
        result = process.extractOne(
            mention, 
            names,
            scorer=fuzz.token_sort_ratio
        )
        
        if result:
            matched_name, score, idx = result
            return all_candidates[idx][0], score
        
        return None, 0.0
    
    def _create_canonical(self, name: str, entity_type: str) -> str:
        """Create a new canonical entity."""
        canonical_id = f"{entity_type}_{len(self.canonical_entities)}_{name.lower().replace(' ', '_')[:20]}"
        
        self.canonical_entities[canonical_id] = CanonicalEntity(
            canonical_id=canonical_id,
            canonical_name=name,
            entity_type=entity_type,
            aliases=[name],
            first_mention=datetime.now(),
            mention_count=1
        )
        
        self._update_alias_index(name, canonical_id)
        return canonical_id
    
    def _update_alias_index(self, alias: str, canonical_id: str):
        """Update the alias lookup index."""
        normalized = alias.lower().strip()
        self.alias_index[normalized] = canonical_id
    
    def merge_entities(self, source_id: str, target_id: str) -> bool:
        """
        Merge source entity into target entity.
        
        Use when discovering two entities are actually the same.
        """
        if source_id not in self.canonical_entities:
            return False
        if target_id not in self.canonical_entities:
            return False
        
        source = self.canonical_entities[source_id]
        target = self.canonical_entities[target_id]
        
        # Merge aliases
        for alias in source.aliases:
            target.add_alias(alias)
            self._update_alias_index(alias, target_id)
        
        # Merge mention counts
        target.mention_count += source.mention_count
        
        # Update metadata
        target.metadata.update(source.metadata)
        
        # Remove source entity
        del self.canonical_entities[source_id]
        
        return True
    
    def add_known_alias(self, canonical_id: str, alias: str) -> bool:
        """Manually add a known alias (e.g., user tells us 'Bob' = 'Robert')."""
        if canonical_id not in self.canonical_entities:
            return False
        
        self.canonical_entities[canonical_id].add_alias(alias)
        self._update_alias_index(alias, canonical_id)
        return True
    
    def save_state(self, path: str):
        """Save resolver state to disk."""
        state = {
            "canonical_entities": {
                cid: {
                    "canonical_id": ce.canonical_id,
                    "canonical_name": ce.canonical_name,
                    "entity_type": ce.entity_type,
                    "aliases": ce.aliases,
                    "mention_count": ce.mention_count,
                    "metadata": ce.metadata
                }
                for cid, ce in self.canonical_entities.items()
            },
            "alias_index": self.alias_index
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load resolver state from disk."""
        with open(path) as f:
            state = json.load(f)
        
        self.canonical_entities = {
            cid: CanonicalEntity(**data)
            for cid, data in state["canonical_entities"].items()
        }
        self.alias_index = state["alias_index"]
```

**Tasks:**
- [ ] Implement EntityResolver with fuzzy matching
- [ ] Add coreference hints from conversation context
- [ ] Create manual alias management API
- [ ] Build entity merge functionality for discovered duplicates
- [ ] Test with realistic name variations

---

### Phase 3: Storage Layer (Week 3-4)

#### Step 3.1: Vector Store Implementation

**Objective:** Store and retrieve memory embeddings efficiently.

**File: `src/storage/vector_store.py`**

```python
import faiss
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import pickle

class VectorStore:
    """FAISS-based vector storage for memory embeddings."""
    
    def __init__(
        self, 
        dimension: int = 384,
        index_path: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (384 for bge-small)
            index_path: Path to load existing index
        """
        self.dimension = dimension
        self.index_path = index_path
        
        if index_path and Path(index_path).exists():
            self._load_index(index_path)
        else:
            # Use IVF for scalability, Flat for small datasets
            self.index = faiss.IndexFlatIP(dimension)  # Inner product
            self.id_map = {}  # faiss_idx -> memory_id
            self.reverse_map = {}  # memory_id -> faiss_idx
    
    def add(
        self, 
        memory_id: str, 
        embedding: np.ndarray
    ) -> int:
        """Add a single embedding."""
        embedding = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)  # Normalize for cosine similarity
        
        idx = self.index.ntotal
        self.index.add(embedding)
        
        self.id_map[idx] = memory_id
        self.reverse_map[memory_id] = idx
        
        return idx
    
    def add_batch(
        self, 
        memory_ids: List[str], 
        embeddings: np.ndarray
    ) -> List[int]:
        """Add multiple embeddings."""
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        indices = []
        for i, memory_id in enumerate(memory_ids):
            idx = start_idx + i
            self.id_map[idx] = memory_id
            self.reverse_map[memory_id] = idx
            indices.append(idx)
        
        return indices
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar memories.
        
        Returns:
            List of (memory_id, score) tuples
        """
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        path = path or self.index_path
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.map", 'wb') as f:
            pickle.dump((self.id_map, self.reverse_map), f)
    
    def _load_index(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.map", 'rb') as f:
            self.id_map, self.reverse_map = pickle.load(f)
```

---

#### Step 3.2: Relational Database Layer

**Objective:** Store structured memory metadata with temporal indexing.

**File: `src/storage/relational_db.py`**

```python
import duckdb
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

class MemoryDatabase:
    """DuckDB-based relational storage for memories."""
    
    def __init__(self, db_path: str = "data/memories.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                event_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                content TEXT NOT NULL,
                type VARCHAR NOT NULL,
                topic VARCHAR,
                importance FLOAT,
                emotion VARCHAR,
                confidence FLOAT,
                supersedes VARCHAR,
                superseded_by VARCHAR,
                embedding_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entities (
                id INTEGER PRIMARY KEY,
                memory_id VARCHAR NOT NULL,
                entity_name VARCHAR NOT NULL,
                entity_type VARCHAR,
                FOREIGN KEY (memory_id) REFERENCES memories(event_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_links (
                id INTEGER PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                link_type VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES memories(event_id),
                FOREIGN KEY (target_id) REFERENCES memories(event_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS belief_deltas (
                id INTEGER PRIMARY KEY,
                topic VARCHAR NOT NULL,
                old_belief_id VARCHAR,
                new_belief_id VARCHAR NOT NULL,
                change_timestamp TIMESTAMP NOT NULL,
                change_type VARCHAR NOT NULL
            )
        """)
        
        # Create indices for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memories(topic)")
    
    def insert_memory(self, memory: Dict[str, Any]) -> str:
        """Insert a memory object."""
        self.conn.execute("""
            INSERT INTO memories 
            (event_id, timestamp, content, type, topic, importance, emotion, confidence, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            memory['event_id'],
            memory['timestamp'],
            memory['content'],
            memory['type'],
            memory.get('topic'),
            memory.get('importance', 0.5),
            memory.get('emotion'),
            memory.get('confidence', 0.7),
            memory.get('embedding_id')
        ])
        
        # Insert entities
        for entity in memory.get('entities', []):
            self.conn.execute("""
                INSERT INTO memory_entities (memory_id, entity_name, entity_type)
                VALUES (?, ?, ?)
            """, [memory['event_id'], entity, None])
        
        return memory['event_id']
    
    def query_by_timerange(
        self, 
        start: datetime, 
        end: datetime,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Query memories within a time range."""
        query = """
            SELECT * FROM memories 
            WHERE timestamp BETWEEN ? AND ?
        """
        params = [start, end]
        
        if memory_types:
            placeholders = ','.join(['?' for _ in memory_types])
            query += f" AND type IN ({placeholders})"
            params.extend(memory_types)
        
        query += " ORDER BY timestamp DESC"
        
        result = self.conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def query_by_topic(self, topic: str) -> List[Dict]:
        """Query memories by topic."""
        result = self.conn.execute("""
            SELECT * FROM memories 
            WHERE topic LIKE ?
            ORDER BY timestamp DESC
        """, [f"%{topic}%"]).fetchall()
        
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
    
    def get_memory(self, event_id: str) -> Optional[Dict]:
        """Get a single memory by ID."""
        result = self.conn.execute(
            "SELECT * FROM memories WHERE event_id = ?", 
            [event_id]
        ).fetchone()
        
        if result:
            columns = [desc[0] for desc in self.conn.description]
            return dict(zip(columns, result))
        return None
    
    def add_causal_link(
        self, 
        source_id: str, 
        target_id: str, 
        link_type: str = "caused",
        confidence: float = 0.5
    ):
        """Add a causal link between memories."""
        self.conn.execute("""
            INSERT INTO causal_links (source_id, target_id, link_type, confidence)
            VALUES (?, ?, ?, ?)
        """, [source_id, target_id, link_type, confidence])
    
    def get_causal_chain(self, event_id: str, direction: str = "effects") -> List[Dict]:
        """Get causal chain from a memory."""
        if direction == "effects":
            query = "SELECT * FROM causal_links WHERE source_id = ?"
        else:  # causes
            query = "SELECT * FROM causal_links WHERE target_id = ?"
        
        result = self.conn.execute(query, [event_id]).fetchall()
        columns = [desc[0] for desc in self.conn.description]
        return [dict(zip(columns, row)) for row in result]
```

---

#### Step 3.3: Knowledge Graph Storage

**Objective:** Store entity relationships as simple triples.

**File: `src/storage/knowledge_graph.py`**

```python
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
import pickle
from pathlib import Path

class KnowledgeGraph:
    """Lightweight knowledge graph using NetworkX."""
    
    def __init__(self, graph_path: Optional[str] = None):
        self.graph_path = graph_path
        
        if graph_path and Path(graph_path).exists():
            self.graph = self._load(graph_path)
        else:
            self.graph = nx.MultiDiGraph()
    
    def add_entity(self, name: str, entity_type: str, **attributes):
        """Add an entity node."""
        self.graph.add_node(name, type=entity_type, **attributes)
    
    def add_relation(
        self, 
        subject: str, 
        predicate: str, 
        obj: str,
        **attributes
    ):
        """Add a relation (edge) between entities."""
        # Ensure nodes exist
        if subject not in self.graph:
            self.graph.add_node(subject, type="unknown")
        if obj not in self.graph:
            self.graph.add_node(obj, type="unknown")
        
        self.graph.add_edge(subject, obj, relation=predicate, **attributes)
    
    def add_triple(self, triple: Tuple[str, str, str], **attributes):
        """Add a (subject, predicate, object) triple."""
        subject, predicate, obj = triple
        self.add_relation(subject, predicate, obj, **attributes)
    
    def add_triples(self, triples: List[Tuple[str, str, str]]):
        """Add multiple triples."""
        for triple in triples:
            self.add_triple(triple)
    
    def get_neighbors(
        self, 
        entity: str, 
        relation_type: Optional[str] = None,
        direction: str = "both"
    ) -> List[Dict]:
        """
        Get neighboring entities.
        
        Args:
            entity: The source entity
            relation_type: Filter by relation type
            direction: "in", "out", or "both"
        """
        neighbors = []
        
        if direction in ["out", "both"]:
            for _, target, data in self.graph.out_edges(entity, data=True):
                if relation_type is None or data.get('relation') == relation_type:
                    neighbors.append({
                        "entity": target,
                        "relation": data.get('relation'),
                        "direction": "out",
                        **data
                    })
        
        if direction in ["in", "both"]:
            for source, _, data in self.graph.in_edges(entity, data=True):
                if relation_type is None or data.get('relation') == relation_type:
                    neighbors.append({
                        "entity": source,
                        "relation": data.get('relation'),
                        "direction": "in",
                        **data
                    })
        
        return neighbors
    
    def find_path(
        self, 
        source: str, 
        target: str, 
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """Find shortest path between entities."""
        try:
            path = nx.shortest_path(
                self.graph, 
                source, 
                target,
                weight=None
            )
            if len(path) <= max_depth + 1:
                return path
        except nx.NetworkXNoPath:
            pass
        return None
    
    def get_subgraph(
        self, 
        entities: List[str], 
        depth: int = 1
    ) -> 'KnowledgeGraph':
        """Extract subgraph around given entities."""
        nodes = set(entities)
        
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.predecessors(node))
                    new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)
        
        subgraph = KnowledgeGraph()
        subgraph.graph = self.graph.subgraph(nodes).copy()
        return subgraph
    
    def query_pattern(
        self, 
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """Query triples matching a pattern (None = wildcard)."""
        results = []
        
        for s, o, data in self.graph.edges(data=True):
            p = data.get('relation')
            
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if obj is not None and o != obj:
                continue
            
            results.append((s, p, o))
        
        return results
    
    def save(self, path: Optional[str] = None):
        """Save graph to disk."""
        path = path or self.graph_path
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def _load(self, path: str) -> nx.MultiDiGraph:
        """Load graph from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
```

---

### Phase 4: Retrieval Layer (Week 4-5)

#### Step 4.1: Embedding Model Wrapper

**Objective:** Unified interface for generating embeddings.

**File: `src/models/embeddings.py`**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingModel:
    """Wrapper for embedding models."""
    
    SUPPORTED_MODELS = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "e5-small": "intfloat/e5-small-v2",
        "e5-base": "intfloat/e5-base-v2",
        "gte-small": "thenlper/gte-small"
    }
    
    def __init__(self, model_name: str = "bge-small", device: str = "cpu"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Short name or HuggingFace model path
            device: cpu or cuda
        """
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.model = SentenceTransformer(model_path, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]],
        is_query: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            is_query: Whether this is a query (adds prefix for some models)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Some models like e5 expect query prefix
        if is_query and "e5" in self.model.get_config_dict().get("_name_or_path", ""):
            texts = [f"query: {t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
```

---

#### Step 4.2: Multi-Channel Retriever

**Objective:** Implement hybrid retrieval with fusion.

**File: `src/retrieval/hybrid_retriever.py`**

```python
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.sql_retriever import SQLRetriever

@dataclass
class RetrievalResult:
    """Unified retrieval result."""
    memory_id: str
    score: float
    source: str  # dense, sparse, graph, sql
    content: Optional[str] = None
    metadata: Optional[Dict] = None

class HybridRetriever:
    """Multi-channel retrieval with fusion."""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        graph_retriever: GraphRetriever,
        sql_retriever: SQLRetriever,
        fusion_weights: Optional[Dict[str, float]] = None
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.graph = graph_retriever
        self.sql = sql_retriever
        
        self.weights = fusion_weights or {
            "dense": 0.4,
            "sparse": 0.2,
            "graph": 0.2,
            "sql": 0.2
        }
    
    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        entities: List[str] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        memory_types: List[str] = None,
        k: int = 20,
        channels: List[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval across all channels.
        
        Args:
            query: Raw query text
            query_embedding: Pre-computed query embedding
            entities: Extracted entities for graph retrieval
            time_start/end: Time filters for SQL retrieval
            memory_types: Filter by memory type
            k: Number of results per channel
            channels: Which channels to use (default: all)
        """
        channels = channels or ["dense", "sparse", "graph", "sql"]
        all_results = []
        
        # Dense retrieval
        if "dense" in channels:
            dense_results = self.dense.search(query_embedding, k=k)
            for memory_id, score in dense_results:
                all_results.append(RetrievalResult(
                    memory_id=memory_id,
                    score=score * self.weights["dense"],
                    source="dense"
                ))
        
        # Sparse retrieval (BM25)
        if "sparse" in channels:
            sparse_results = self.sparse.search(query, k=k)
            for memory_id, score in sparse_results:
                all_results.append(RetrievalResult(
                    memory_id=memory_id,
                    score=score * self.weights["sparse"],
                    source="sparse"
                ))
        
        # Graph retrieval
        if "graph" in channels and entities:
            graph_results = self.graph.search(entities, k=k)
            for memory_id, score in graph_results:
                all_results.append(RetrievalResult(
                    memory_id=memory_id,
                    score=score * self.weights["graph"],
                    source="graph"
                ))
        
        # SQL retrieval (time/type filters)
        if "sql" in channels and (time_start or time_end or memory_types):
            sql_results = self.sql.search(
                time_start=time_start,
                time_end=time_end,
                memory_types=memory_types,
                k=k
            )
            for memory_id, score in sql_results:
                all_results.append(RetrievalResult(
                    memory_id=memory_id,
                    score=score * self.weights["sql"],
                    source="sql"
                ))
        
        # Fusion: aggregate scores by memory_id
        fused = self._reciprocal_rank_fusion(all_results, k=k)
        
        return fused
    
    def _reciprocal_rank_fusion(
        self, 
        results: List[RetrievalResult], 
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining results.
        
        RRF score = Σ 1/(k + rank_i) across all lists
        """
        # Group by source
        by_source = {}
        for r in results:
            if r.source not in by_source:
                by_source[r.source] = []
            by_source[r.source].append(r)
        
        # Sort each source list by score
        for source in by_source:
            by_source[source].sort(key=lambda x: x.score, reverse=True)
        
        # Compute RRF scores
        rrf_scores = {}
        for source, source_results in by_source.items():
            for rank, result in enumerate(source_results):
                if result.memory_id not in rrf_scores:
                    rrf_scores[result.memory_id] = {
                        "score": 0,
                        "sources": [],
                        "result": result
                    }
                rrf_scores[result.memory_id]["score"] += 1 / (k + rank + 1)
                rrf_scores[result.memory_id]["sources"].append(source)
        
        # Convert to results
        fused_results = []
        for memory_id, data in rrf_scores.items():
            result = data["result"]
            result.score = data["score"]
            result.source = "+".join(sorted(set(data["sources"])))
            fused_results.append(result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
```

---

#### Step 4.3: Individual Retrievers

**Dense Retriever (`src/retrieval/dense_retriever.py`):**
```python
from src.storage.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel
import numpy as np
from typing import List, Tuple

class DenseRetriever:
    """Dense retrieval using embeddings."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for similar memories."""
        return self.vector_store.search(query_embedding, k=k)
    
    def search_text(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using text query."""
        embedding = self.embedding_model.encode(query, is_query=True)
        return self.search(embedding, k=k)
```

**Sparse Retriever (`src/retrieval/sparse_retriever.py`):**
```python
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import re

class SparseRetriever:
    """BM25-based sparse retrieval."""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def index(self, documents: List[Dict]):
        """
        Index documents for BM25.
        
        Args:
            documents: List of {id, content} dicts
        """
        self.documents = documents
        self.doc_ids = [d['id'] for d in documents]
        
        tokenized = [self._tokenize(d['content']) for d in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results
```

---

### Phase 4.5: Memory Consolidation (Week 5)

**Critical Addition:** The original plan had no strategy for handling decades of memories. Without consolidation, the system will:
- Run out of storage
- Have slow retrieval (searching millions of vectors)
- Lose important patterns buried in noise

#### Step 4.5.1: Hierarchical Summarization

**Objective:** Compress old memories while preserving important information and causal relationships.

**File: `src/consolidation/memory_consolidator.py`**

```python
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from src.models.memory_types import CausalMemoryObject
from src.llm.local_llm import LocalLLM
from src.storage.relational_db import MemoryDatabase
from src.storage.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel

class ConsolidationLevel(Enum):
    """Levels of memory consolidation."""
    RAW = "raw"           # Individual memories (< 1 week old)
    DAILY = "daily"       # Daily summaries (1 week - 1 month)
    WEEKLY = "weekly"     # Weekly summaries (1-6 months)
    MONTHLY = "monthly"   # Monthly summaries (6 months - 2 years)
    YEARLY = "yearly"     # Yearly summaries (> 2 years)

@dataclass
class ConsolidatedMemory:
    """A summarized memory representing multiple raw memories."""
    consolidated_id: str
    level: ConsolidationLevel
    start_date: datetime
    end_date: datetime
    summary: str
    key_events: List[str]
    entities_mentioned: List[str]
    emotional_tone: str
    source_memory_ids: List[str]
    importance_score: float
    embedding: Optional[List[float]] = None

class MemoryConsolidator:
    """
    Hierarchical memory consolidation system.
    
    Implements time-decay based consolidation:
    - Recent memories: preserved in full detail
    - Older memories: progressively summarized
    - Important memories: retained regardless of age
    
    Architecture:
    
    Day 1-7: Raw memories (full detail)
         ↓ (weekly consolidation job)
    Week 2-4: Daily summaries + high-importance raw
         ↓ (monthly consolidation job)
    Month 2-6: Weekly summaries + high-importance daily
         ↓ (yearly consolidation job)
    Month 7+: Monthly summaries + landmark memories
    """
    
    # Consolidation thresholds
    DAILY_AGE_DAYS = 7          # Start daily consolidation after 7 days
    WEEKLY_AGE_DAYS = 30        # Start weekly consolidation after 30 days
    MONTHLY_AGE_DAYS = 180      # Start monthly consolidation after 6 months
    YEARLY_AGE_DAYS = 730       # Start yearly consolidation after 2 years
    
    # Importance thresholds (memories above these are preserved)
    PRESERVE_IMPORTANCE_DAILY = 0.8
    PRESERVE_IMPORTANCE_WEEKLY = 0.9
    PRESERVE_IMPORTANCE_MONTHLY = 0.95
    
    # Summary limits
    MAX_DAILY_SUMMARY_LENGTH = 500
    MAX_WEEKLY_SUMMARY_LENGTH = 1000
    MAX_MONTHLY_SUMMARY_LENGTH = 2000
    
    def __init__(
        self, 
        memory_db: MemoryDatabase,
        vector_store: VectorStore,
        llm: LocalLLM,
        embedding_model: EmbeddingModel
    ):
        self.memory_db = memory_db
        self.vector_store = vector_store
        self.llm = llm
        self.embedding_model = embedding_model
        
        self.summarization_prompt = """
        Summarize the following memories from {date_range}.
        
        Memories:
        {memories}
        
        Create a summary that:
        1. Captures the key events and decisions
        2. Notes any emotional highs/lows
        3. Preserves causal relationships (X led to Y)
        4. Mentions important people/projects/places
        
        Keep the summary under {max_length} words.
        Focus on what would be important to remember years later.
        
        Return JSON:
        {{
            "summary": "...",
            "key_events": ["event1", "event2", ...],
            "emotional_tone": "positive|negative|neutral|mixed",
            "entities": ["person1", "project1", ...]
        }}
        """
    
    def run_consolidation(self, reference_date: Optional[datetime] = None):
        """
        Run full consolidation pipeline.
        
        Should be run as a daily background job.
        """
        reference_date = reference_date or datetime.now()
        
        # Consolidate in order: daily -> weekly -> monthly -> yearly
        self._consolidate_to_daily(reference_date)
        self._consolidate_to_weekly(reference_date)
        self._consolidate_to_monthly(reference_date)
        self._consolidate_to_yearly(reference_date)
    
    def _consolidate_to_daily(self, reference_date: datetime):
        """Create daily summaries from raw memories older than threshold."""
        cutoff = reference_date - timedelta(days=self.DAILY_AGE_DAYS)
        
        # Find days with unconsolidated raw memories
        days_to_consolidate = self._find_unconsolidated_days(
            cutoff, 
            ConsolidationLevel.RAW,
            ConsolidationLevel.DAILY
        )
        
        for day in days_to_consolidate:
            memories = self._get_memories_for_day(day)
            
            # Separate high-importance memories (preserve) from others (summarize)
            to_preserve, to_summarize = self._partition_by_importance(
                memories, 
                self.PRESERVE_IMPORTANCE_DAILY
            )
            
            if to_summarize:
                consolidated = self._create_summary(
                    memories=to_summarize,
                    level=ConsolidationLevel.DAILY,
                    date_range=f"{day.strftime('%Y-%m-%d')}",
                    max_length=self.MAX_DAILY_SUMMARY_LENGTH
                )
                self._store_consolidated(consolidated)
                self._archive_source_memories(to_summarize)
    
    def _consolidate_to_weekly(self, reference_date: datetime):
        """Create weekly summaries from daily summaries older than threshold."""
        cutoff = reference_date - timedelta(days=self.WEEKLY_AGE_DAYS)
        
        weeks_to_consolidate = self._find_unconsolidated_weeks(
            cutoff,
            ConsolidationLevel.DAILY,
            ConsolidationLevel.WEEKLY
        )
        
        for week_start in weeks_to_consolidate:
            week_end = week_start + timedelta(days=7)
            daily_summaries = self._get_daily_summaries_for_range(week_start, week_end)
            
            to_preserve, to_summarize = self._partition_by_importance(
                daily_summaries,
                self.PRESERVE_IMPORTANCE_WEEKLY
            )
            
            if to_summarize:
                consolidated = self._create_summary(
                    memories=to_summarize,
                    level=ConsolidationLevel.WEEKLY,
                    date_range=f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
                    max_length=self.MAX_WEEKLY_SUMMARY_LENGTH
                )
                self._store_consolidated(consolidated)
    
    def _consolidate_to_monthly(self, reference_date: datetime):
        """Create monthly summaries from weekly summaries."""
        cutoff = reference_date - timedelta(days=self.MONTHLY_AGE_DAYS)
        
        months_to_consolidate = self._find_unconsolidated_months(
            cutoff,
            ConsolidationLevel.WEEKLY,
            ConsolidationLevel.MONTHLY
        )
        
        for month_start in months_to_consolidate:
            # Get all weeks in this month
            month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
            weekly_summaries = self._get_weekly_summaries_for_range(month_start, month_end)
            
            to_preserve, to_summarize = self._partition_by_importance(
                weekly_summaries,
                self.PRESERVE_IMPORTANCE_MONTHLY
            )
            
            if to_summarize:
                consolidated = self._create_summary(
                    memories=to_summarize,
                    level=ConsolidationLevel.MONTHLY,
                    date_range=f"{month_start.strftime('%B %Y')}",
                    max_length=self.MAX_MONTHLY_SUMMARY_LENGTH
                )
                self._store_consolidated(consolidated)
    
    def _consolidate_to_yearly(self, reference_date: datetime):
        """Create yearly summaries from monthly summaries (for very old memories)."""
        cutoff = reference_date - timedelta(days=self.YEARLY_AGE_DAYS)
        
        years_to_consolidate = self._find_unconsolidated_years(cutoff)
        
        for year in years_to_consolidate:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31)
            
            monthly_summaries = self._get_monthly_summaries_for_range(year_start, year_end)
            
            if monthly_summaries:
                consolidated = self._create_summary(
                    memories=monthly_summaries,
                    level=ConsolidationLevel.YEARLY,
                    date_range=str(year),
                    max_length=self.MAX_MONTHLY_SUMMARY_LENGTH * 2
                )
                self._store_consolidated(consolidated)
    
    def _partition_by_importance(
        self, 
        memories: List, 
        threshold: float
    ) -> Tuple[List, List]:
        """Separate memories into preserve and summarize groups."""
        to_preserve = [m for m in memories if getattr(m, 'importance', 0) >= threshold]
        to_summarize = [m for m in memories if getattr(m, 'importance', 0) < threshold]
        return to_preserve, to_summarize
    
    def _create_summary(
        self, 
        memories: List,
        level: ConsolidationLevel,
        date_range: str,
        max_length: int
    ) -> ConsolidatedMemory:
        """Create a consolidated memory from multiple source memories."""
        # Format memories for prompt
        memory_texts = []
        source_ids = []
        all_entities = set()
        
        for m in memories:
            if hasattr(m, 'content'):
                memory_texts.append(f"- {m.content}")
                source_ids.append(m.event_id)
                all_entities.update(getattr(m, 'entities', []))
            elif hasattr(m, 'summary'):
                memory_texts.append(f"- {m.summary}")
                source_ids.extend(m.source_memory_ids)
                all_entities.update(m.entities_mentioned)
        
        # Generate summary
        prompt = self.summarization_prompt.format(
            date_range=date_range,
            memories="\n".join(memory_texts),
            max_length=max_length
        )
        
        response = self.llm.generate(prompt)
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "summary": response[:max_length * 5],  # Rough char estimate
                "key_events": [],
                "emotional_tone": "neutral",
                "entities": list(all_entities)
            }
        
        # Calculate importance as max of source memories
        importance = max(
            getattr(m, 'importance', 0.5) for m in memories
        ) if memories else 0.5
        
        # Generate embedding for the summary
        summary_text = result.get("summary", "")
        embedding = self.embedding_model.embed(summary_text).tolist()
        
        # Create consolidated memory
        consolidated_id = f"consolidated_{level.value}_{date_range.replace(' ', '_')}"
        
        return ConsolidatedMemory(
            consolidated_id=consolidated_id,
            level=level,
            start_date=memories[0].timestamp if memories else datetime.now(),
            end_date=memories[-1].timestamp if memories else datetime.now(),
            summary=summary_text,
            key_events=result.get("key_events", []),
            entities_mentioned=result.get("entities", list(all_entities)),
            emotional_tone=result.get("emotional_tone", "neutral"),
            source_memory_ids=source_ids,
            importance_score=importance,
            embedding=embedding
        )
    
    def _store_consolidated(self, consolidated: ConsolidatedMemory):
        """Store a consolidated memory in the database and vector store."""
        # Store in relational DB
        self.memory_db.conn.execute("""
            INSERT INTO consolidated_memories 
            (consolidated_id, level, start_date, end_date, summary, 
             key_events, entities, emotional_tone, source_ids, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            consolidated.consolidated_id,
            consolidated.level.value,
            consolidated.start_date,
            consolidated.end_date,
            consolidated.summary,
            json.dumps(consolidated.key_events),
            json.dumps(consolidated.entities_mentioned),
            consolidated.emotional_tone,
            json.dumps(consolidated.source_memory_ids),
            consolidated.importance_score
        ])
        
        # Store embedding in vector store
        if consolidated.embedding:
            self.vector_store.add(
                memory_id=consolidated.consolidated_id,
                embedding=consolidated.embedding
            )
    
    def _archive_source_memories(self, memories: List):
        """Move source memories to archive table (preserving data but freeing main index)."""
        for memory in memories:
            # Move to archive table
            self.memory_db.conn.execute("""
                INSERT INTO archived_memories SELECT * FROM memories WHERE event_id = ?
            """, [memory.event_id])
            
            # Remove from main table
            self.memory_db.conn.execute("""
                DELETE FROM memories WHERE event_id = ?
            """, [memory.event_id])
            
            # Remove from vector index (keep in archive index)
            # Note: FAISS doesn't support deletion, so we mark as deleted
            self.vector_store.mark_deleted(memory.event_id)
    
    def _find_unconsolidated_days(self, cutoff, source_level, target_level) -> List[datetime]:
        """Find days that need consolidation."""
        # Implementation: query DB for days with raw memories older than cutoff
        # that don't have corresponding daily summaries
        result = self.memory_db.conn.execute("""
            SELECT DISTINCT DATE(timestamp) as day
            FROM memories
            WHERE timestamp < ?
            AND event_id NOT IN (
                SELECT json_each.value 
                FROM consolidated_memories, json_each(source_ids)
                WHERE level = ?
            )
            ORDER BY day
        """, [cutoff, target_level.value]).fetchall()
        
        return [datetime.strptime(r[0], '%Y-%m-%d') for r in result]
    
    # Additional helper methods would be implemented similarly...
    def _get_memories_for_day(self, day: datetime) -> List[CausalMemoryObject]:
        """Get all memories for a specific day."""
        pass  # Implementation
    
    def _find_unconsolidated_weeks(self, cutoff, source_level, target_level):
        """Find weeks needing consolidation."""
        pass  # Implementation
    
    def _find_unconsolidated_months(self, cutoff, source_level, target_level):
        """Find months needing consolidation."""
        pass  # Implementation
    
    def _find_unconsolidated_years(self, cutoff):
        """Find years needing consolidation."""
        pass  # Implementation
    
    def _get_daily_summaries_for_range(self, start, end):
        """Get daily summaries in date range."""
        pass  # Implementation
    
    def _get_weekly_summaries_for_range(self, start, end):
        """Get weekly summaries in date range."""
        pass  # Implementation
    
    def _get_monthly_summaries_for_range(self, start, end):
        """Get monthly summaries in date range."""
        pass  # Implementation

    def get_compression_stats(self) -> Dict:
        """Get statistics about memory compression."""
        stats = self.memory_db.conn.execute("""
            SELECT 
                (SELECT COUNT(*) FROM memories) as raw_count,
                (SELECT COUNT(*) FROM archived_memories) as archived_count,
                (SELECT COUNT(*) FROM consolidated_memories WHERE level = 'daily') as daily_count,
                (SELECT COUNT(*) FROM consolidated_memories WHERE level = 'weekly') as weekly_count,
                (SELECT COUNT(*) FROM consolidated_memories WHERE level = 'monthly') as monthly_count,
                (SELECT COUNT(*) FROM consolidated_memories WHERE level = 'yearly') as yearly_count
        """).fetchone()
        
        return {
            "raw_memories": stats[0],
            "archived_memories": stats[1],
            "daily_summaries": stats[2],
            "weekly_summaries": stats[3],
            "monthly_summaries": stats[4],
            "yearly_summaries": stats[5],
            "compression_ratio": stats[1] / max(stats[0] + stats[1], 1)
        }
```

**Database Schema Addition:**

```sql
-- Add to schema initialization
CREATE TABLE IF NOT EXISTS consolidated_memories (
    consolidated_id VARCHAR PRIMARY KEY,
    level VARCHAR NOT NULL,  -- daily, weekly, monthly, yearly
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    summary TEXT NOT NULL,
    key_events JSON,
    entities JSON,
    emotional_tone VARCHAR,
    source_ids JSON,
    importance FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS archived_memories (
    -- Same schema as memories table
    event_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    content TEXT NOT NULL,
    type VARCHAR NOT NULL,
    topic VARCHAR,
    importance FLOAT,
    emotion VARCHAR,
    confidence FLOAT,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_consolidated_level ON consolidated_memories(level);
CREATE INDEX IF NOT EXISTS idx_consolidated_dates ON consolidated_memories(start_date, end_date);
```

**Tasks:**
- [ ] Implement hierarchical consolidation pipeline
- [ ] Add database tables for consolidated and archived memories
- [ ] Create background job scheduler for nightly consolidation
- [ ] Implement retrieval across consolidation levels
- [ ] Add compression statistics monitoring
- [ ] Test consolidation with simulated time progression

---

### Phase 5: Agent Layer (Week 5-6)

#### Step 5.1: Base Agent Framework

**File: `src/agents/base_agent.py`**

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.llm.local_llm import LocalLLM
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult

@dataclass
class AgentResponse:
    """Standardized agent response."""
    answer: str
    evidence: List[RetrievalResult]
    reasoning_trace: List[str]
    confidence: float
    follow_up_queries: List[str] = None

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self, 
        llm: LocalLLM, 
        retriever: HybridRetriever,
        name: str = "BaseAgent"
    ):
        self.llm = llm
        self.retriever = retriever
        self.name = name
        self.reasoning_trace = []
    
    @abstractmethod
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Process a query and return response."""
        pass
    
    def _log_reasoning(self, step: str):
        """Log a reasoning step."""
        self.reasoning_trace.append(f"[{self.name}] {step}")
    
    def _retrieve(
        self, 
        query: str, 
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve relevant memories."""
        from src.models.embeddings import EmbeddingModel
        
        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.encode(query, is_query=True)
        
        return self.retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            **kwargs
        )
```

---

#### Step 5.2: Specialized Agents

**Timeline Agent (`src/agents/timeline_agent.py`):**
```python
from src.agents.base_agent import BaseAgent, AgentResponse
from typing import Dict, Any, List
from datetime import datetime

class TimelineAgent(BaseAgent):
    """Agent for timeline-based queries."""
    
    def __init__(self, llm, retriever):
        super().__init__(llm, retriever, name="TimelineAgent")
        
        self.intent_prompt = """
        Analyze this query and extract temporal information:
        Query: {query}
        
        Return JSON:
        {{
            "time_references": ["explicit dates/periods mentioned"],
            "time_start": "YYYY-MM-DD or null",
            "time_end": "YYYY-MM-DD or null",
            "temporal_scope": "specific_date|date_range|period|relative|all_time",
            "focus": "what the user wants to know about this time"
        }}
        """
    
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        self._log_reasoning(f"Processing query: {query}")
        
        # Step 1: Extract temporal information
        temporal_info = self._extract_temporal_info(query)
        self._log_reasoning(f"Extracted temporal info: {temporal_info}")
        
        # Step 2: Retrieve memories from timeframe
        results = self._retrieve(
            query,
            time_start=temporal_info.get("time_start"),
            time_end=temporal_info.get("time_end"),
            k=20
        )
        self._log_reasoning(f"Retrieved {len(results)} memories")
        
        # Step 3: Build timeline narrative
        answer = self._build_timeline_narrative(query, results, temporal_info)
        
        return AgentResponse(
            answer=answer,
            evidence=results[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.8 if results else 0.3
        )
    
    def _extract_temporal_info(self, query: str) -> Dict:
        prompt = self.intent_prompt.format(query=query)
        response = self.llm.generate(prompt)
        # Parse JSON response
        import json
        try:
            return json.loads(response)
        except:
            return {"temporal_scope": "all_time"}
    
    def _build_timeline_narrative(
        self, 
        query: str, 
        results: List, 
        temporal_info: Dict
    ) -> str:
        # Sort by timestamp and build narrative
        evidence_text = "\n".join([
            f"- [{r.metadata.get('timestamp', 'unknown')}] {r.content}"
            for r in results[:10] if r.content
        ])
        
        narrative_prompt = f"""
        Based on these memories, answer the user's question about their timeline.
        
        Query: {query}
        Time focus: {temporal_info.get('focus', 'general')}
        
        Memories (chronological):
        {evidence_text}
        
        Provide a clear, chronological answer that traces the evolution of events/thoughts.
        """
        
        return self.llm.generate(narrative_prompt)
```

**Cause-Effect Agent (`src/agents/causal_agent.py`):**
```python
from src.agents.base_agent import BaseAgent, AgentResponse
from typing import Dict, Any

class CausalAgent(BaseAgent):
    """Agent for causal reasoning queries."""
    
    def __init__(self, llm, retriever, memory_db):
        super().__init__(llm, retriever, name="CausalAgent")
        self.memory_db = memory_db
    
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        self._log_reasoning(f"Processing causal query: {query}")
        
        # Step 1: Identify the event/decision in question
        target_event = self._identify_target_event(query)
        self._log_reasoning(f"Target event: {target_event}")
        
        # Step 2: Retrieve related memories
        results = self._retrieve(query, k=15)
        
        # Step 3: Trace causal chain
        if results:
            causal_chain = self._trace_causal_chain(results[0].memory_id)
            self._log_reasoning(f"Causal chain depth: {len(causal_chain)}")
        else:
            causal_chain = []
        
        # Step 4: Generate causal explanation
        answer = self._explain_causality(query, results, causal_chain)
        
        return AgentResponse(
            answer=answer,
            evidence=results[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.7 if causal_chain else 0.4
        )
    
    def _identify_target_event(self, query: str) -> Dict:
        prompt = f"""
        What event, decision, or outcome is the user asking about?
        Query: {query}
        
        Return: {{"event_type": "decision|outcome|belief|behavior", "description": "..."}}
        """
        return self.llm.generate(prompt)
    
    def _trace_causal_chain(self, event_id: str, depth: int = 3) -> list:
        """Trace causes and effects from an event."""
        chain = []
        
        # Get causes
        causes = self.memory_db.get_causal_chain(event_id, direction="causes")
        chain.extend([("cause", c) for c in causes])
        
        # Get effects
        effects = self.memory_db.get_causal_chain(event_id, direction="effects")
        chain.extend([("effect", e) for e in effects])
        
        return chain
    
    def _explain_causality(self, query, results, causal_chain) -> str:
        evidence = "\n".join([r.content for r in results[:5] if r.content])
        chain_text = "\n".join([f"{direction}: {link}" for direction, link in causal_chain[:10]])
        
        prompt = f"""
        Explain the causal relationships for this query.
        
        Query: {query}
        
        Related memories:
        {evidence}
        
        Causal links found:
        {chain_text}
        
        Provide a clear explanation of what caused what, and what effects followed.
        """
        return self.llm.generate(prompt)
```

---

#### Step 5.2.1: Enhanced Causal Link Detection (Critical Fix)

**Critical Fix:** The original plan assumed causal links could be automatically inferred. In practice, most causality is implicit and requires either:
1. Explicit user statements ("I did X because of Y")
2. Temporal proximity + semantic similarity
3. Manual user confirmation

**File: `src/ingestion/causal_detector.py`**

```python
"""
Enhanced causal link detection with explicit action handling.

Three-tier approach:
1. EXPLICIT: User states causality directly ("because", "led to", "resulted in")
2. INFERRED: Temporal + semantic proximity suggests causality (low confidence)
3. CONFIRMED: User confirms or rejects inferred links
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from src.models.memory_types import CausalMemoryObject
from src.llm.local_llm import LocalLLM
from src.models.embeddings import EmbeddingModel
import numpy as np

@dataclass
class CausalLink:
    """A causal relationship between two memories."""
    source_id: str
    target_id: str
    link_type: str  # caused, led_to, resulted_in, influenced, blocked
    confidence: float  # 0.0 - 1.0
    detection_method: str  # explicit, inferred, confirmed
    user_confirmed: Optional[bool] = None

class CausalLinkDetector:
    """
    Detect causal relationships in memory content.
    
    Explicit causality indicators (high confidence):
    - "because", "because of", "due to"
    - "led to", "resulted in", "caused"
    - "as a result", "consequently", "therefore"
    - "made me decide", "convinced me to"
    
    Inferred causality (low confidence):
    - High semantic similarity (>0.85)
    - Temporal proximity (within 7 days)
    - Same topic/entity mentions
    """
    
    # Explicit causality patterns
    EXPLICIT_PATTERNS = [
        # Forward causality: "X led to Y"
        (r'(?P<cause>.+?)\s+(led to|resulted in|caused|triggered)\s+(?P<effect>.+)', 'forward'),
        # Backward causality: "Y because of X"
        (r'(?P<effect>.+?)\s+(because|because of|due to)\s+(?P<cause>.+)', 'backward'),
        # Decision causality: "I decided X because Y"
        (r'(decided|chose|went with)\s+(?P<effect>.+?)\s+(because|since)\s+(?P<cause>.+)', 'decision'),
        # Consequence: "As a result, X"
        (r'(as a result|consequently|therefore|thus),?\s+(?P<effect>.+)', 'consequence'),
    ]
    
    # Confidence thresholds
    EXPLICIT_CONFIDENCE = 0.9
    INFERRED_HIGH_CONFIDENCE = 0.6
    INFERRED_LOW_CONFIDENCE = 0.4
    CONFIRMATION_THRESHOLD = 0.5  # Below this, ask user for confirmation
    
    def __init__(
        self, 
        llm: LocalLLM,
        embedding_model: EmbeddingModel,
        memory_db
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.memory_db = memory_db
    
    def detect_causal_links(
        self, 
        new_memory: CausalMemoryObject,
        recent_memories: List[CausalMemoryObject]
    ) -> List[CausalLink]:
        """
        Detect causal links between new memory and recent memories.
        
        Args:
            new_memory: The newly ingested memory
            recent_memories: Memories from the past 30 days
        
        Returns:
            List of detected causal links (may include low-confidence inferences)
        """
        links = []
        
        # Step 1: Check for explicit causality in new memory
        explicit_links = self._detect_explicit_causality(new_memory, recent_memories)
        links.extend(explicit_links)
        
        # Step 2: Infer potential causality from context
        if not explicit_links:
            inferred_links = self._infer_causality(new_memory, recent_memories)
            links.extend(inferred_links)
        
        return links
    
    def _detect_explicit_causality(
        self, 
        new_memory: CausalMemoryObject,
        recent_memories: List[CausalMemoryObject]
    ) -> List[CausalLink]:
        """Detect explicit causal statements in the memory content."""
        links = []
        content = new_memory.content
        
        for pattern, direction in self.EXPLICIT_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groupdict()
                cause_text = groups.get('cause', '')
                effect_text = groups.get('effect', '')
                
                if direction == 'backward':
                    # Effect is the new memory, find cause in recent
                    cause_memory = self._find_matching_memory(cause_text, recent_memories)
                    if cause_memory:
                        links.append(CausalLink(
                            source_id=cause_memory.event_id,
                            target_id=new_memory.event_id,
                            link_type="caused",
                            confidence=self.EXPLICIT_CONFIDENCE,
                            detection_method="explicit"
                        ))
                
                elif direction == 'forward':
                    # Cause is mentioned, effect is the result
                    cause_memory = self._find_matching_memory(cause_text, recent_memories)
                    if cause_memory:
                        links.append(CausalLink(
                            source_id=cause_memory.event_id,
                            target_id=new_memory.event_id,
                            link_type="led_to",
                            confidence=self.EXPLICIT_CONFIDENCE,
                            detection_method="explicit"
                        ))
                
                elif direction == 'decision':
                    # User explicitly states a decision and its reason
                    cause_memory = self._find_matching_memory(cause_text, recent_memories)
                    if cause_memory:
                        links.append(CausalLink(
                            source_id=cause_memory.event_id,
                            target_id=new_memory.event_id,
                            link_type="influenced",
                            confidence=self.EXPLICIT_CONFIDENCE,
                            detection_method="explicit"
                        ))
        
        return links
    
    def _find_matching_memory(
        self, 
        text_fragment: str,
        recent_memories: List[CausalMemoryObject]
    ) -> Optional[CausalMemoryObject]:
        """Find a memory that matches the text fragment."""
        if not text_fragment or len(text_fragment.strip()) < 5:
            return None
        
        # Embed the fragment
        fragment_embedding = self.embedding_model.embed(text_fragment)
        
        best_match = None
        best_score = 0.7  # Minimum similarity threshold
        
        for memory in recent_memories:
            memory_embedding = self.embedding_model.embed(memory.content)
            similarity = float(np.dot(fragment_embedding, memory_embedding))
            
            if similarity > best_score:
                best_score = similarity
                best_match = memory
        
        return best_match
    
    def _infer_causality(
        self, 
        new_memory: CausalMemoryObject,
        recent_memories: List[CausalMemoryObject]
    ) -> List[CausalLink]:
        """Infer potential causal links from context."""
        links = []
        new_embedding = self.embedding_model.embed(new_memory.content)
        
        for old_memory in recent_memories:
            # Skip if same memory
            if old_memory.event_id == new_memory.event_id:
                continue
            
            # Calculate temporal proximity (max 7 days)
            time_diff = new_memory.timestamp - old_memory.timestamp
            if time_diff.days < 0 or time_diff.days > 7:
                continue
            
            # Calculate semantic similarity
            old_embedding = self.embedding_model.embed(old_memory.content)
            similarity = float(np.dot(new_embedding, old_embedding))
            
            if similarity < 0.75:
                continue
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_inference_confidence(
                similarity=similarity,
                time_diff_hours=time_diff.total_seconds() / 3600,
                same_topic=(old_memory.topic == new_memory.topic),
                shared_entities=self._count_shared_entities(old_memory, new_memory)
            )
            
            if confidence >= self.INFERRED_LOW_CONFIDENCE:
                links.append(CausalLink(
                    source_id=old_memory.event_id,
                    target_id=new_memory.event_id,
                    link_type="potentially_related",
                    confidence=confidence,
                    detection_method="inferred"
                ))
        
        # Return top 3 inferred links by confidence
        links.sort(key=lambda x: x.confidence, reverse=True)
        return links[:3]
    
    def _calculate_inference_confidence(
        self,
        similarity: float,
        time_diff_hours: float,
        same_topic: bool,
        shared_entities: int
    ) -> float:
        """Calculate confidence score for inferred causality."""
        confidence = 0.0
        
        # Semantic similarity (0.75-1.0 maps to 0-0.4)
        confidence += (similarity - 0.75) / 0.25 * 0.4
        
        # Temporal proximity (0-168 hours maps to 0.3-0)
        time_factor = max(0, 1 - time_diff_hours / 168) * 0.2
        confidence += time_factor
        
        # Same topic bonus
        if same_topic:
            confidence += 0.15
        
        # Shared entities bonus
        confidence += min(shared_entities * 0.05, 0.15)
        
        return min(confidence, 1.0)
    
    def _count_shared_entities(
        self, 
        mem1: CausalMemoryObject, 
        mem2: CausalMemoryObject
    ) -> int:
        """Count entities shared between two memories."""
        entities1 = set(mem1.entities) if mem1.entities else set()
        entities2 = set(mem2.entities) if mem2.entities else set()
        return len(entities1 & entities2)
    
    def request_confirmation(self, link: CausalLink) -> Dict:
        """
        Generate a confirmation request for low-confidence links.
        
        Returns a dict that can be shown to the user.
        """
        source = self.memory_db.get_memory(link.source_id)
        target = self.memory_db.get_memory(link.target_id)
        
        return {
            "link_id": f"{link.source_id}_{link.target_id}",
            "question": f"Did this earlier memory lead to or influence the later one?",
            "earlier_memory": {
                "content": source["content"],
                "timestamp": source["timestamp"]
            },
            "later_memory": {
                "content": target["content"],
                "timestamp": target["timestamp"]
            },
            "inferred_confidence": link.confidence,
            "options": ["Yes, definitely", "Possibly", "No connection", "Opposite - later caused earlier"]
        }
    
    def process_confirmation(
        self, 
        link_id: str, 
        user_response: str
    ) -> Optional[CausalLink]:
        """Process user confirmation of a causal link."""
        source_id, target_id = link_id.split("_", 1)
        
        if user_response == "Yes, definitely":
            return CausalLink(
                source_id=source_id,
                target_id=target_id,
                link_type="caused",
                confidence=0.95,
                detection_method="confirmed",
                user_confirmed=True
            )
        elif user_response == "Possibly":
            return CausalLink(
                source_id=source_id,
                target_id=target_id,
                link_type="potentially_related",
                confidence=0.7,
                detection_method="confirmed",
                user_confirmed=True
            )
        elif user_response == "Opposite - later caused earlier":
            return CausalLink(
                source_id=target_id,
                target_id=source_id,
                link_type="caused",
                confidence=0.95,
                detection_method="confirmed",
                user_confirmed=True
            )
        else:
            return None  # No connection
```

**Tasks:**
- [ ] Implement explicit causality pattern detection
- [ ] Add inference with multi-factor confidence scoring
- [ ] Create user confirmation flow for low-confidence links
- [ ] Integrate with memory ingestion pipeline
- [ ] Store confirmation history for model improvement

**Arbitration Agent (`src/agents/arbitration_agent.py`):**
```python
from src.agents.base_agent import BaseAgent, AgentResponse
from typing import Dict, Any, List

class ArbitrationAgent(BaseAgent):
    """Agent for resolving conflicting memories."""
    
    def __init__(self, llm, retriever, memory_db):
        super().__init__(llm, retriever, name="ArbitrationAgent")
        self.memory_db = memory_db
    
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        
        # Step 1: Retrieve relevant memories
        results = self._retrieve(query, k=20)
        
        # Step 2: Detect conflicts
        conflicts = self._detect_conflicts(results)
        self._log_reasoning(f"Found {len(conflicts)} potential conflicts")
        
        # Step 3: Resolve conflicts
        resolution = self._resolve_conflicts(conflicts)
        
        # Step 4: Generate unified answer
        answer = self._generate_unified_answer(query, results, resolution)
        
        return AgentResponse(
            answer=answer,
            evidence=results[:10],
            reasoning_trace=self.reasoning_trace,
            confidence=0.6 if conflicts else 0.8
        )
    
    def _detect_conflicts(self, results: List) -> List[Dict]:
        """Detect contradictory memories."""
        conflicts = []
        
        # Compare pairs of memories for contradictions
        contents = [(r.memory_id, r.content) for r in results if r.content]
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                id1, content1 = contents[i]
                id2, content2 = contents[j]
                
                if self._are_contradictory(content1, content2):
                    conflicts.append({
                        "memory_1": id1,
                        "memory_2": id2,
                        "content_1": content1,
                        "content_2": content2
                    })
        
        return conflicts
    
    def _are_contradictory(self, content1: str, content2: str) -> bool:
        """Use LLM to detect contradiction."""
        prompt = f"""
        Are these two statements contradictory?
        
        Statement 1: {content1}
        Statement 2: {content2}
        
        Answer only: YES or NO
        """
        response = self.llm.generate(prompt).strip().upper()
        return response == "YES"
    
    def _resolve_conflicts(self, conflicts: List[Dict]) -> List[Dict]:
        """Resolve each conflict based on criteria."""
        resolutions = []
        
        for conflict in conflicts:
            # Get memory metadata
            mem1 = self.memory_db.get_memory(conflict["memory_1"])
            mem2 = self.memory_db.get_memory(conflict["memory_2"])
            
            # Resolution criteria: recency, confidence, repetition
            winner = self._pick_winner(mem1, mem2)
            
            resolutions.append({
                "conflict": conflict,
                "winner": winner,
                "reason": self._explain_resolution(mem1, mem2, winner)
            })
        
        return resolutions
    
    def _pick_winner(self, mem1: Dict, mem2: Dict) -> str:
        """Pick the more reliable memory."""
        # Simple heuristic: prefer more recent + higher confidence
        score1 = mem1.get("confidence", 0.5)
        score2 = mem2.get("confidence", 0.5)
        
        if mem1.get("timestamp") > mem2.get("timestamp"):
            score1 += 0.2
        else:
            score2 += 0.2
        
        return mem1["event_id"] if score1 >= score2 else mem2["event_id"]
    
    def _explain_resolution(self, mem1, mem2, winner) -> str:
        return f"Preferred {'first' if winner == mem1['event_id'] else 'second'} memory based on recency and confidence."
    
    def _generate_unified_answer(self, query, results, resolution) -> str:
        # Build answer incorporating resolution info
        prompt = f"""
        Answer this query, noting any belief changes over time.
        
        Query: {query}
        
        Evidence (may contain evolved/changed beliefs):
        {[r.content for r in results[:5]]}
        
        Provide a coherent answer that acknowledges how understanding evolved.
        """
        return self.llm.generate(prompt)
```

---

#### Step 5.3: Agent Orchestrator

**Critical Fix:** The original plan used simplistic keyword-based agent routing with no support for multi-step queries, agent chaining, or fallback mechanisms. This step adds a proper orchestrator.

**File: `src/agents/orchestrator.py`**

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from src.agents.base_agent import BaseAgent, AgentResponse
from src.agents.timeline_agent import TimelineAgent
from src.agents.causal_agent import CausalAgent
from src.agents.arbitration_agent import ArbitrationAgent
from src.llm.local_llm import LocalLLM
from src.retrieval.hybrid_retriever import HybridRetriever

class QueryIntent(Enum):
    """Types of query intents."""
    TIMELINE = "timeline"            # "What happened last week?"
    CAUSAL = "causal"               # "Why did I decide to...?"
    RECALL = "recall"               # "What do I know about...?"
    CONFLICT = "conflict"           # Detected contradictions
    PLANNING = "planning"           # "What should I do about...?"
    REFLECTION = "reflection"       # "How have my views on X changed?"
    MULTI_STEP = "multi_step"       # Complex queries needing multiple agents

@dataclass
class QueryPlan:
    """A plan for processing a complex query."""
    original_query: str
    intent: QueryIntent
    sub_queries: List[str] = field(default_factory=list)
    agent_sequence: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_complexity: float = 0.5

@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""
    answer: str
    evidence: List[Any]
    agent_trace: List[Dict[str, Any]]  # Which agents were used and why
    confidence: float
    processing_time_ms: float

class AgentOrchestrator:
    """
    Intelligent query routing and multi-agent coordination.
    
    Improvements over simple keyword routing:
    1. Intent classification with confidence
    2. Multi-step query decomposition
    3. Agent chaining for complex queries
    4. Fallback mechanisms when primary agent fails
    5. Result synthesis from multiple agents
    """
    
    # Intent detection patterns (as fallback to classifier)
    INTENT_PATTERNS = {
        QueryIntent.TIMELINE: [
            r'\b(when|what happened|timeline|last week|yesterday|this month)\b',
            r'\b(chronolog|sequence of events|history of)\b'
        ],
        QueryIntent.CAUSAL: [
            r'\b(why|because|caused|led to|result of|consequence)\b',
            r'\b(what made me|reason for|how did .* affect)\b'
        ],
        QueryIntent.REFLECTION: [
            r'\b(how have I changed|evolution of|used to think|changed my mind)\b',
            r'\b(over time|progression of|belief about)\b'
        ],
        QueryIntent.PLANNING: [
            r'\b(should I|what if|plan for|next steps|recommendation)\b',
            r'\b(based on .* what should|given .* what would)\b'
        ],
        QueryIntent.CONFLICT: [
            r'\b(contradict|conflict|inconsistent|but .* said)\b'
        ]
    }
    
    # Agent capability mapping
    AGENT_CAPABILITIES = {
        "timeline": ["date_queries", "sequence", "period_summary"],
        "causal": ["why_questions", "cause_effect", "influence_chains"],
        "arbitration": ["conflicts", "contradictions", "belief_changes"],
        "recall": ["factual", "entity_info", "general"]  # Hypothetical general agent
    }
    
    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever,
        timeline_agent: TimelineAgent,
        causal_agent: CausalAgent,
        arbitration_agent: ArbitrationAgent,
        memory_db = None
    ):
        self.llm = llm
        self.retriever = retriever
        self.agents = {
            "timeline": timeline_agent,
            "causal": causal_agent,
            "arbitration": arbitration_agent
        }
        self.memory_db = memory_db
        
        # Intent classification prompt
        self.intent_prompt = """
        Classify the intent of this memory query.
        
        Query: {query}
        
        Categories:
        - timeline: Questions about when things happened, sequences of events
        - causal: Questions about why things happened, causes and effects
        - recall: Simple factual recall about entities, projects, or events
        - reflection: Questions about how beliefs/understanding changed over time
        - planning: Questions seeking recommendations based on past experience
        - conflict: Queries that might involve contradictory information
        
        Also assess if this is a simple query (one agent) or complex (needs multiple agents).
        
        Return JSON:
        {{
            "primary_intent": "...",
            "secondary_intent": "..." or null,
            "is_complex": true/false,
            "sub_queries": ["...", "..."] if complex else [],
            "confidence": 0.0-1.0
        }}
        """
    
    def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestratorResponse:
        """
        Process a query using intelligent routing and coordination.
        """
        start_time = datetime.now()
        context = context or {}
        agent_trace = []
        
        # Step 1: Understand the query
        plan = self._create_query_plan(query)
        agent_trace.append({
            "step": "planning",
            "intent": plan.intent.value,
            "complexity": plan.estimated_complexity,
            "sub_queries": plan.sub_queries
        })
        
        # Step 2: Execute based on complexity
        if plan.intent == QueryIntent.MULTI_STEP or len(plan.sub_queries) > 1:
            response = self._execute_multi_agent(plan, context, agent_trace)
        else:
            response = self._execute_single_agent(plan, context, agent_trace)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return OrchestratorResponse(
            answer=response.answer,
            evidence=response.evidence,
            agent_trace=agent_trace,
            confidence=response.confidence,
            processing_time_ms=processing_time
        )
    
    def _create_query_plan(self, query: str) -> QueryPlan:
        """Analyze query and create an execution plan."""
        # Try classifier first
        try:
            classification = self._classify_intent(query)
            intent = QueryIntent(classification["primary_intent"])
            is_complex = classification.get("is_complex", False)
            sub_queries = classification.get("sub_queries", [])
            confidence = classification.get("confidence", 0.5)
        except:
            # Fall back to pattern matching
            intent, confidence = self._pattern_match_intent(query)
            is_complex = self._assess_complexity(query)
            sub_queries = self._decompose_query(query) if is_complex else []
        
        # Determine agent sequence
        agent_sequence = self._plan_agent_sequence(intent, is_complex, sub_queries)
        
        return QueryPlan(
            original_query=query,
            intent=QueryIntent.MULTI_STEP if is_complex else intent,
            sub_queries=sub_queries,
            agent_sequence=agent_sequence,
            estimated_complexity=0.8 if is_complex else 0.3
        )
    
    def _classify_intent(self, query: str) -> Dict:
        """Use LLM for intent classification."""
        prompt = self.intent_prompt.format(query=query)
        response = self.llm.generate(prompt)
        
        import json
        return json.loads(response)
    
    def _pattern_match_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Fallback pattern-based intent detection."""
        query_lower = query.lower()
        
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[intent] = score
        
        if max(scores.values()) == 0:
            return QueryIntent.RECALL, 0.5
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 2, 1.0)
        
        return best_intent, confidence
    
    def _assess_complexity(self, query: str) -> bool:
        """Determine if query needs multiple agents."""
        # Simple heuristics
        complexity_indicators = [
            len(query.split()) > 20,  # Long queries
            ' and ' in query.lower(),  # Multiple sub-questions
            '?' in query and query.count('?') > 1,  # Multiple questions
            re.search(r'(first .* then|after .* what)', query.lower()),  # Sequential
        ]
        return sum(complexity_indicators) >= 2
    
    def _decompose_query(self, query: str) -> List[str]:
        """Break complex query into sub-queries."""
        # Use LLM to decompose
        prompt = f"""
        Break this complex query into simpler sub-queries that can be answered independently.
        
        Query: {query}
        
        Return a JSON array of sub-queries (max 3).
        """
        response = self.llm.generate(prompt)
        
        try:
            import json
            return json.loads(response)
        except:
            return [query]
    
    def _plan_agent_sequence(
        self, 
        intent: QueryIntent,
        is_complex: bool,
        sub_queries: List[str]
    ) -> List[str]:
        """Determine which agents to use and in what order."""
        # Map intent to primary agent
        intent_to_agent = {
            QueryIntent.TIMELINE: ["timeline"],
            QueryIntent.CAUSAL: ["causal"],
            QueryIntent.CONFLICT: ["arbitration"],
            QueryIntent.REFLECTION: ["timeline", "arbitration"],  # Chain
            QueryIntent.PLANNING: ["causal", "timeline"],  # Chain
            QueryIntent.RECALL: ["timeline"],  # Default
            QueryIntent.MULTI_STEP: [],  # Determined by sub-queries
        }
        
        agents = intent_to_agent.get(intent, ["timeline"])
        
        if is_complex and not agents:
            # Analyze sub-queries to determine agents
            agents = []
            for sq in sub_queries:
                sq_intent, _ = self._pattern_match_intent(sq)
                sq_agents = intent_to_agent.get(sq_intent, ["timeline"])
                for a in sq_agents:
                    if a not in agents:
                        agents.append(a)
        
        return agents
    
    def _execute_single_agent(
        self, 
        plan: QueryPlan,
        context: Dict[str, Any],
        agent_trace: List[Dict]
    ) -> AgentResponse:
        """Execute query with a single agent."""
        agent_name = plan.agent_sequence[0] if plan.agent_sequence else "timeline"
        agent = self.agents.get(agent_name)
        
        if not agent:
            agent_name = "timeline"
            agent = self.agents["timeline"]
        
        try:
            response = agent.process(plan.original_query, context)
            agent_trace.append({
                "step": "execution",
                "agent": agent_name,
                "success": True,
                "confidence": response.confidence
            })
            return response
        except Exception as e:
            agent_trace.append({
                "step": "execution",
                "agent": agent_name,
                "success": False,
                "error": str(e)
            })
            # Fallback to another agent
            return self._try_fallback(plan.original_query, context, agent_name, agent_trace)
    
    def _execute_multi_agent(
        self, 
        plan: QueryPlan,
        context: Dict[str, Any],
        agent_trace: List[Dict]
    ) -> AgentResponse:
        """Execute complex query across multiple agents."""
        all_evidence = []
        sub_answers = []
        
        # If we have sub-queries, process each
        queries_to_process = plan.sub_queries if plan.sub_queries else [plan.original_query]
        
        for i, sub_query in enumerate(queries_to_process):
            # Get appropriate agent for this sub-query
            if i < len(plan.agent_sequence):
                agent_name = plan.agent_sequence[i]
            else:
                agent_name = plan.agent_sequence[-1] if plan.agent_sequence else "timeline"
            
            agent = self.agents.get(agent_name, self.agents["timeline"])
            
            try:
                response = agent.process(sub_query, context)
                sub_answers.append({
                    "query": sub_query,
                    "answer": response.answer,
                    "confidence": response.confidence
                })
                all_evidence.extend(response.evidence[:5])
                
                agent_trace.append({
                    "step": f"sub_query_{i+1}",
                    "agent": agent_name,
                    "query": sub_query,
                    "success": True,
                    "confidence": response.confidence
                })
                
                # Pass context forward for chaining
                context["previous_answer"] = response.answer
                context["previous_evidence"] = response.evidence
                
            except Exception as e:
                agent_trace.append({
                    "step": f"sub_query_{i+1}",
                    "agent": agent_name,
                    "success": False,
                    "error": str(e)
                })
        
        # Synthesize final answer
        final_answer = self._synthesize_answers(plan.original_query, sub_answers)
        avg_confidence = sum(sa["confidence"] for sa in sub_answers) / len(sub_answers) if sub_answers else 0.5
        
        agent_trace.append({
            "step": "synthesis",
            "sub_answer_count": len(sub_answers),
            "final_confidence": avg_confidence
        })
        
        return AgentResponse(
            answer=final_answer,
            evidence=all_evidence,
            reasoning_trace=[f"Processed {len(sub_answers)} sub-queries"],
            confidence=avg_confidence
        )
    
    def _synthesize_answers(self, original_query: str, sub_answers: List[Dict]) -> str:
        """Synthesize multiple sub-answers into a coherent response."""
        if not sub_answers:
            return "I couldn't find relevant information for this query."
        
        if len(sub_answers) == 1:
            return sub_answers[0]["answer"]
        
        # Use LLM to synthesize
        sub_answer_text = "\n\n".join([
            f"Q: {sa['query']}\nA: {sa['answer']}"
            for sa in sub_answers
        ])
        
        prompt = f"""
        Synthesize these answers into a coherent response to the original question.
        
        Original question: {original_query}
        
        Sub-answers:
        {sub_answer_text}
        
        Provide a unified, flowing answer that addresses the original question.
        """
        
        return self.llm.generate(prompt)
    
    def _try_fallback(
        self, 
        query: str, 
        context: Dict[str, Any],
        failed_agent: str,
        agent_trace: List[Dict]
    ) -> AgentResponse:
        """Try fallback agents when primary fails."""
        fallback_order = ["timeline", "causal", "arbitration"]
        
        for agent_name in fallback_order:
            if agent_name == failed_agent:
                continue
            
            agent = self.agents.get(agent_name)
            if not agent:
                continue
            
            try:
                response = agent.process(query, context)
                agent_trace.append({
                    "step": "fallback",
                    "agent": agent_name,
                    "success": True,
                    "confidence": response.confidence
                })
                return response
            except:
                continue
        
        # All agents failed - return a basic response
        return AgentResponse(
            answer="I was unable to process this query. Please try rephrasing.",
            evidence=[],
            reasoning_trace=["All agents failed"],
            confidence=0.1
        )
```

**Tasks:**
- [ ] Implement intent classification with confidence scoring
- [ ] Build multi-step query decomposition
- [ ] Create agent chaining mechanism
- [ ] Add fallback logic for agent failures
- [ ] Implement answer synthesis for multi-agent queries
- [ ] Test with complex, multi-part queries

---

### Phase 6: LLM Integration (Week 6-7)

#### Step 6.1: Local LLM Interface

**File: `src/llm/local_llm.py`**

```python
import ollama
from typing import Optional, List, Dict

class LocalLLM:
    """Interface to local LLMs via Ollama."""
    
    def __init__(
        self, 
        model: str = "mistral:7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize local LLM.
        
        Args:
            model: Ollama model name
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verify model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Pull model if not available."""
        try:
            ollama.show(self.model)
        except:
            print(f"Pulling model {self.model}...")
            ollama.pull(self.model)
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens
            }
        )
        
        return response["message"]["content"]
    
    def generate_structured(
        self, 
        prompt: str,
        schema: Dict
    ) -> Dict:
        """
        Generate structured JSON response.
        
        Args:
            prompt: User prompt
            schema: Expected JSON schema
        """
        import json
        
        structured_prompt = f"""
        {prompt}
        
        Respond with valid JSON matching this schema:
        {json.dumps(schema, indent=2)}
        
        Only output the JSON, no other text.
        """
        
        response = self.generate(structured_prompt, temperature=0.3)
        
        # Clean and parse
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """
        Multi-turn chat.
        
        Args:
            messages: List of {role, content} dicts
        """
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        return response["message"]["content"]
```

---

### Phase 7: Memory Evolution & Conflict Resolution (Week 7-8)

#### Step 7.1: Enhanced Belief Evolution Tracker

**Critical Fix:** The original belief evolution detection was too simplistic - using only topic matching and a single LLM call. Real belief changes are subtle and require multi-stage validation.

**Multi-Stage Belief Evolution Pipeline:**

```
New Memory → Semantic Similarity (>0.8) → Stance Detection → Temporal Context → Evolution Classification
                    ↓                           ↓                   ↓                    ↓
              Candidate Pool            agree/disagree/neutral   Same timeframe?    Final Decision
```

**File: `src/storage/belief_tracker.py`**

```python
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from src.models.memory_types import BeliefDelta, CausalMemoryObject
from src.llm.local_llm import LocalLLM
from src.models.embeddings import EmbeddingModel
import numpy as np

class Stance(Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    NEUTRAL = "neutral"
    REFINES = "refines"
    EXPANDS = "expands"

@dataclass
class EvolutionCandidate:
    """A candidate belief evolution pair."""
    old_memory: CausalMemoryObject
    new_memory: CausalMemoryObject
    semantic_similarity: float
    stance: Stance
    time_gap_days: int
    confidence: float

class BeliefEvolutionTracker:
    """
    Track how beliefs and understanding change over time.
    
    Uses multi-stage validation to avoid false positives:
    1. Semantic similarity filtering (must be discussing same concept)
    2. Stance detection (agree/disagree/neutral/refines/expands)
    3. Temporal context (same-day memories rarely contradict)
    4. LLM validation for borderline cases
    """
    
    # Thresholds for multi-stage filtering
    SEMANTIC_THRESHOLD = 0.75      # Must be discussing similar concepts
    SEMANTIC_STRICT = 0.90         # High similarity = likely related
    SAME_DAY_PENALTY = 0.3         # Reduce confidence for same-day "contradictions"
    MIN_CONFIDENCE = 0.6           # Minimum confidence to report evolution
    
    def __init__(
        self, 
        memory_db, 
        llm: LocalLLM,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.memory_db = memory_db
        self.llm = llm
        self.embedding_model = embedding_model or EmbeddingModel()
        
        # Stance detection prompt
        self.stance_prompt = """
        Compare these two statements from the same person at different times.
        
        Statement A: {content_a}
        Statement B: {content_b}
        
        Determine the stance of Statement B relative to Statement A:
        - "agree": B supports or confirms A
        - "disagree": B contradicts or opposes A
        - "refines": B adds nuance or precision to A
        - "expands": B builds upon A with new information
        - "neutral": B is unrelated or discusses something different
        
        Consider that people can:
        - Change their minds genuinely (months/years apart)
        - Express temporary frustration (same day)
        - Discuss different aspects of the same topic
        
        Respond with JSON: {{"stance": "...", "explanation": "brief reason"}}
        """
    
    def check_evolution(
        self, 
        new_memory: CausalMemoryObject,
        existing_memories: List[CausalMemoryObject]
    ) -> Optional[BeliefDelta]:
        """
        Check if new memory represents belief evolution using multi-stage validation.
        
        Returns:
            BeliefDelta if evolution detected with sufficient confidence, None otherwise
        """
        if new_memory.type not in ["semantic", "reflective", "belief"]:
            # Only track belief-like memories
            return None
        
        # Stage 1: Find semantically similar memories
        candidates = self._find_semantic_candidates(new_memory, existing_memories)
        
        if not candidates:
            return None
        
        # Stage 2 & 3: Apply stance detection and temporal context
        evolution_candidates = []
        for old_memory, similarity in candidates:
            candidate = self._analyze_candidate(old_memory, new_memory, similarity)
            if candidate and candidate.confidence >= self.MIN_CONFIDENCE:
                evolution_candidates.append(candidate)
        
        if not evolution_candidates:
            return None
        
        # Return highest confidence evolution
        best = max(evolution_candidates, key=lambda c: c.confidence)
        
        # Map stance to change_type
        change_type = self._stance_to_change_type(best.stance)
        
        if change_type:
            return BeliefDelta(
                topic=new_memory.topic or self._extract_topic(new_memory.content),
                old_belief_id=best.old_memory.event_id,
                new_belief_id=best.new_memory.event_id,
                change_timestamp=new_memory.timestamp,
                change_type=change_type,
                confidence=best.confidence
            )
        
        return None
    
    def _find_semantic_candidates(
        self, 
        new_memory: CausalMemoryObject,
        existing: List[CausalMemoryObject]
    ) -> List[Tuple[CausalMemoryObject, float]]:
        """
        Stage 1: Find semantically similar memories.
        
        Uses embedding similarity rather than just topic matching.
        """
        candidates = []
        
        # Get embedding for new memory
        new_embedding = self.embedding_model.embed(new_memory.content)
        
        for old_memory in existing:
            # Skip non-belief memories
            if old_memory.type not in ["semantic", "reflective", "belief"]:
                continue
            
            # Skip very recent memories (within 1 hour likely same conversation)
            time_diff = new_memory.timestamp - old_memory.timestamp
            if time_diff < timedelta(hours=1):
                continue
            
            # Calculate semantic similarity
            old_embedding = self.embedding_model.embed(old_memory.content)
            similarity = float(np.dot(new_embedding, old_embedding))
            
            if similarity >= self.SEMANTIC_THRESHOLD:
                candidates.append((old_memory, similarity))
        
        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return candidates[:10]
    
    def _analyze_candidate(
        self, 
        old_memory: CausalMemoryObject,
        new_memory: CausalMemoryObject,
        semantic_similarity: float
    ) -> Optional[EvolutionCandidate]:
        """
        Stage 2 & 3: Analyze a candidate pair for belief evolution.
        """
        # Calculate time gap
        time_gap = new_memory.timestamp - old_memory.timestamp
        time_gap_days = time_gap.days
        
        # Stage 2: Stance detection using lightweight prompt
        stance = self._detect_stance(old_memory.content, new_memory.content)
        
        if stance == Stance.NEUTRAL:
            return None
        
        # Stage 3: Calculate confidence with temporal context
        confidence = self._calculate_confidence(
            semantic_similarity=semantic_similarity,
            stance=stance,
            time_gap_days=time_gap_days
        )
        
        return EvolutionCandidate(
            old_memory=old_memory,
            new_memory=new_memory,
            semantic_similarity=semantic_similarity,
            stance=stance,
            time_gap_days=time_gap_days,
            confidence=confidence
        )
    
    def _detect_stance(self, content_a: str, content_b: str) -> Stance:
        """Detect the stance of B relative to A."""
        prompt = self.stance_prompt.format(content_a=content_a, content_b=content_b)
        response = self.llm.generate(prompt)
        
        try:
            import json
            result = json.loads(response)
            stance_str = result.get("stance", "neutral").lower()
            
            stance_map = {
                "agree": Stance.AGREE,
                "disagree": Stance.DISAGREE,
                "refines": Stance.REFINES,
                "expands": Stance.EXPANDS,
                "neutral": Stance.NEUTRAL
            }
            return stance_map.get(stance_str, Stance.NEUTRAL)
        except:
            return Stance.NEUTRAL
    
    def _calculate_confidence(
        self, 
        semantic_similarity: float,
        stance: Stance,
        time_gap_days: int
    ) -> float:
        """
        Calculate evolution confidence based on multiple factors.
        
        Higher confidence for:
        - High semantic similarity
        - Clear disagreement stance
        - Sufficient time gap (belief changes need time)
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Semantic similarity (0.75-1.0 maps to 0-0.3 bonus)
        similarity_factor = (semantic_similarity - self.SEMANTIC_THRESHOLD) / (1 - self.SEMANTIC_THRESHOLD)
        confidence += similarity_factor * 0.3
        
        # Factor 2: Stance clarity
        if stance == Stance.DISAGREE:
            confidence += 0.2
        elif stance in [Stance.REFINES, Stance.EXPANDS]:
            confidence += 0.1
        elif stance == Stance.AGREE:
            # Agreement isn't evolution
            confidence -= 0.3
        
        # Factor 3: Temporal context
        if time_gap_days == 0:
            # Same day - likely temporary or different context
            confidence *= (1 - self.SAME_DAY_PENALTY)
        elif time_gap_days < 7:
            # Within a week - mild skepticism
            confidence *= 0.9
        elif time_gap_days > 365:
            # Over a year - very likely genuine evolution
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _stance_to_change_type(self, stance: Stance) -> Optional[str]:
        """Convert stance to belief evolution change type."""
        mapping = {
            Stance.DISAGREE: "contradiction",
            Stance.REFINES: "refinement",
            Stance.EXPANDS: "expansion",
            Stance.AGREE: None,  # Agreement isn't evolution
            Stance.NEUTRAL: None
        }
        return mapping.get(stance)
    
    def _extract_topic(self, content: str) -> str:
        """Extract topic from content when not explicitly labeled."""
        # Simple extraction - take first noun phrase
        words = content.split()[:5]
        return " ".join(words) + "..."
    
    def get_belief_timeline(self, topic: str) -> List[Dict]:
        """Get evolution of beliefs on a topic over time."""
        deltas = self.memory_db.conn.execute("""
            SELECT * FROM belief_deltas 
            WHERE topic LIKE ?
            ORDER BY change_timestamp
        """, [f"%{topic}%"]).fetchall()
        
        timeline = []
        columns = [desc[0] for desc in self.memory_db.conn.description]
        
        for row in deltas:
            delta = dict(zip(columns, row))
            old_mem = self.memory_db.get_memory(delta["old_belief_id"])
            new_mem = self.memory_db.get_memory(delta["new_belief_id"])
            
            timeline.append({
                "timestamp": delta["change_timestamp"],
                "change_type": delta["change_type"],
                "confidence": delta.get("confidence", 0.5),
                "old_belief": old_mem["content"] if old_mem else None,
                "new_belief": new_mem["content"]
            })
        
        return timeline
    
    def validate_evolution(
        self, 
        evolution_id: str, 
        is_valid: bool
    ) -> None:
        """
        Manual validation of detected evolution (for learning).
        
        User can confirm or reject detected belief changes,
        improving future detection accuracy.
        """
        # Store validation for potential fine-tuning
        self.memory_db.conn.execute("""
            UPDATE belief_deltas 
            SET validated = ?, validation_timestamp = ?
            WHERE id = ?
        """, [is_valid, datetime.now(), evolution_id])
```

**Tasks:**
- [ ] Implement multi-stage belief evolution detection
- [ ] Add semantic similarity filtering
- [ ] Implement stance detection with confidence scoring
- [ ] Add temporal context weighting
- [ ] Build manual validation API for feedback loop
- [ ] Test with realistic belief change scenarios

---

### Phase 8: End-to-End Integration (Week 8-9)

#### Step 8.1: Main Pipeline Orchestrator

**File: `src/pipeline.py`**

```python
from datetime import datetime
from typing import Optional, Dict, Any

from src.ingestion.asr import LocalASR
from src.ingestion.memory_builder import MemoryEventBuilder
from src.ingestion.entity_extractor import EntityExtractor
from src.models.embeddings import EmbeddingModel
from src.models.memory_types import CausalMemoryObject, MemoryQuery
from src.storage.vector_store import VectorStore
from src.storage.relational_db import MemoryDatabase
from src.storage.knowledge_graph import KnowledgeGraph
from src.storage.belief_tracker import BeliefEvolutionTracker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.agents.timeline_agent import TimelineAgent
from src.agents.causal_agent import CausalAgent
from src.agents.arbitration_agent import ArbitrationAgent
from src.llm.local_llm import LocalLLM

class EdgeMemoryPipeline:
    """Main orchestrator for EdgeMemory system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize all components."""
        self.config = config
        
        # Initialize LLM
        self.llm = LocalLLM(
            model=config.get("llm_model", "mistral:7b-instruct")
        )
        
        # Initialize ASR
        self.asr = LocalASR(
            model_size=config.get("whisper_model", "base")
        )
        
        # Initialize embeddings
        self.embedding_model = EmbeddingModel(
            model_name=config.get("embedding_model", "bge-small")
        )
        
        # Initialize storage
        self.vector_store = VectorStore(
            dimension=self.embedding_model.dimension,
            index_path=config.get("vector_path", "data/vectors")
        )
        self.memory_db = MemoryDatabase(
            db_path=config.get("db_path", "data/memories.db")
        )
        self.knowledge_graph = KnowledgeGraph(
            graph_path=config.get("graph_path", "data/knowledge.pkl")
        )
        
        # Initialize ingestion
        self.memory_builder = MemoryEventBuilder(self.llm)
        self.entity_extractor = EntityExtractor(self.llm)
        self.belief_tracker = BeliefEvolutionTracker(self.memory_db, self.llm)
        
        # Initialize retrieval
        self.dense_retriever = DenseRetriever(self.vector_store, self.embedding_model)
        self.sparse_retriever = SparseRetriever()
        # ... initialize other retrievers
        
        # Initialize agents
        self.timeline_agent = TimelineAgent(self.llm, self.retriever)
        self.causal_agent = CausalAgent(self.llm, self.retriever, self.memory_db)
        self.arbitration_agent = ArbitrationAgent(self.llm, self.retriever, self.memory_db)
    
    # ==================== INGESTION ====================
    
    def ingest_audio(self, audio_path: str) -> CausalMemoryObject:
        """Ingest audio file into memory."""
        # Transcribe
        transcript = self.asr.transcribe(audio_path)
        
        # Build memory
        return self.ingest_text(transcript["text"])
    
    def ingest_text(
        self, 
        text: str, 
        timestamp: Optional[datetime] = None
    ) -> CausalMemoryObject:
        """Ingest text into memory."""
        timestamp = timestamp or datetime.now()
        
        # Build memory object
        memory = self.memory_builder.build(text, timestamp)
        
        # Generate embedding
        embedding = self.embedding_model.encode(text)
        
        # Store in vector DB
        self.vector_store.add(memory.event_id, embedding)
        memory.embedding_id = memory.event_id
        
        # Extract entities and build graph
        entities = self.entity_extractor.extract(text)
        triples = self.entity_extractor.build_relations(memory.event_id, entities)
        self.knowledge_graph.add_triples(triples)
        
        # Store in relational DB
        self.memory_db.insert_memory(memory.model_dump())
        
        # Check for belief evolution
        existing = self.memory_db.query_by_topic(memory.topic) if memory.topic else []
        existing_memories = [CausalMemoryObject(**m) for m in existing]
        
        evolution = self.belief_tracker.check_evolution(memory, existing_memories)
        if evolution:
            # Link evolution
            memory.supersedes = evolution.old_belief_id
            # Update old memory
            self.memory_db.conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE event_id = ?",
                [memory.event_id, evolution.old_belief_id]
            )
        
        return memory
    
    # ==================== QUERY ====================
    
    def query(self, query_text: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a query and return answer with evidence.
        
        Args:
            query_text: Natural language query
            context: Optional context (current goal, emotional state, etc.)
        """
        context = context or {}
        
        # Step 1: Understand query intent
        intent = self._classify_intent(query_text)
        
        # Step 2: Route to appropriate agent
        if intent["type"] == "timeline":
            response = self.timeline_agent.process(query_text, context)
        elif intent["type"] == "causal":
            response = self.causal_agent.process(query_text, context)
        elif intent["type"] == "conflict":
            response = self.arbitration_agent.process(query_text, context)
        else:
            # Default: timeline agent
            response = self.timeline_agent.process(query_text, context)
        
        return {
            "answer": response.answer,
            "evidence": [
                {"id": e.memory_id, "content": e.content, "score": e.score}
                for e in response.evidence
            ],
            "reasoning": response.reasoning_trace,
            "confidence": response.confidence
        }
    
    def _classify_intent(self, query: str) -> Dict:
        """Classify query intent for agent routing."""
        prompt = f"""
        Classify this query into one category:
        
        Query: {query}
        
        Categories:
        - timeline: questions about what happened when, evolution over time
        - causal: questions about why, causes, effects, decisions
        - conflict: questions involving contradictory information
        - pattern: questions about recurring patterns or behaviors
        - recall: simple fact recall
        
        Return JSON: {{"type": "category", "entities": [...], "time_scope": "..."}}
        """
        
        response = self.llm.generate_structured(prompt, {
            "type": "string",
            "entities": "array",
            "time_scope": "string"
        })
        
        return response or {"type": "timeline"}
    
    # ==================== NIGHTLY PROCESSING ====================
    
    def run_nightly_reflection(self):
        """Background processing for pattern extraction."""
        # Get recent memories
        recent = self.memory_db.query_by_timerange(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now()
        )
        
        # Extract patterns
        patterns = self._extract_patterns(recent)
        
        # Consolidate beliefs
        self._consolidate_beliefs()
        
        # Update sparse index
        self._rebuild_sparse_index()
        
        return patterns
    
    def _extract_patterns(self, memories: list) -> list:
        """Extract behavioral patterns from recent memories."""
        content = "\n".join([m["content"] for m in memories])
        
        prompt = f"""
        Analyze these recent memories and identify recurring patterns:
        
        {content}
        
        Look for:
        - Emotional patterns (what triggers stress, joy, etc.)
        - Decision patterns (how decisions are made)
        - Learning patterns (what helps understanding)
        - Behavioral cycles
        
        Return as JSON array of patterns.
        """
        
        return self.llm.generate_structured(prompt, {"patterns": "array"})
    
    # ==================== PERSISTENCE ====================
    
    def save(self):
        """Save all components to disk."""
        self.vector_store.save()
        self.knowledge_graph.save()
    
    def close(self):
        """Clean shutdown."""
        self.save()
        self.memory_db.conn.close()
```

---

### Phase 9: Testing & Evaluation (Week 9-10)

#### Step 9.1: Test Suite

**File: `tests/test_pipeline.py`**

```python
import pytest
from datetime import datetime, timedelta
from src.pipeline import EdgeMemoryPipeline

@pytest.fixture
def pipeline():
    config = {
        "llm_model": "phi3:mini",  # Fast for testing
        "whisper_model": "tiny",
        "embedding_model": "bge-small",
        "db_path": "data/test_memories.db",
        "vector_path": "data/test_vectors"
    }
    return EdgeMemoryPipeline(config)

class TestIngestion:
    def test_text_ingestion(self, pipeline):
        memory = pipeline.ingest_text(
            "I decided to focus on deep learning today",
            timestamp=datetime.now()
        )
        
        assert memory.event_id is not None
        assert memory.type in ["episodic", "semantic", "procedural", "reflective"]
        assert "deep learning" in memory.content.lower()
    
    def test_entity_extraction(self, pipeline):
        memory = pipeline.ingest_text(
            "Meeting with John about Project X at the office"
        )
        
        assert len(memory.entities) > 0
        assert "John" in memory.entities or "Project X" in memory.entities

class TestRetrieval:
    def test_dense_retrieval(self, pipeline):
        # Ingest some memories
        pipeline.ingest_text("I learned about transformers")
        pipeline.ingest_text("Attention mechanism is key")
        
        # Query
        result = pipeline.query("What did I learn about transformers?")
        
        assert result["answer"] is not None
        assert len(result["evidence"]) > 0
    
    def test_temporal_query(self, pipeline):
        # Ingest with timestamps
        pipeline.ingest_text(
            "Started my deep learning journey",
            timestamp=datetime(2025, 1, 1)
        )
        pipeline.ingest_text(
            "Finally understood backpropagation",
            timestamp=datetime(2025, 3, 1)
        )
        
        result = pipeline.query("What happened in January 2025?")
        assert "journey" in result["answer"].lower()

class TestBeliefEvolution:
    def test_contradiction_detection(self, pipeline):
        # Ingest contradictory beliefs
        pipeline.ingest_text(
            "I think neural networks are too complex to understand",
            timestamp=datetime(2025, 1, 1)
        )
        memory2 = pipeline.ingest_text(
            "Neural networks are actually quite intuitive once you get the basics",
            timestamp=datetime(2025, 6, 1)
        )
        
        # Check evolution was detected
        assert memory2.supersedes is not None
```

---

#### Step 9.2: Comprehensive Evaluation Framework

**Critical Addition:** Without a proper evaluation framework, you cannot measure progress or compare approaches. This section provides synthetic datasets and automated benchmarks.

**File: `evaluation/synthetic_dataset.py`**

```python
"""
Synthetic evaluation dataset for EdgeMemory.

Generates realistic memory scenarios covering:
- Various memory types (episodic, semantic, reflective)
- Entity relationships and coreferences
- Belief evolution patterns
- Causal chains
- Temporal queries

Dataset: 100 memories + 50 ground-truth queries
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random

@dataclass
class SyntheticMemory:
    """A synthetic memory with ground truth metadata."""
    content: str
    timestamp: datetime
    ground_truth_type: str
    ground_truth_entities: List[str]
    ground_truth_topic: str
    importance: float
    related_memory_ids: List[int]  # For causal chain ground truth
    coreference_cluster: str       # For entity resolution testing

@dataclass
class SyntheticQuery:
    """A query with expected ground truth answer."""
    query: str
    query_type: str  # timeline, causal, recall, reflection
    expected_memory_ids: List[int]  # Which memories should be retrieved
    expected_answer_contains: List[str]  # Keywords that should appear
    ground_truth_answer: str

class SyntheticDatasetGenerator:
    """Generate realistic synthetic evaluation data."""
    
    PERSONS = [
        ("Sarah", "my manager", "Sarah Chen", "the boss"),
        ("Mike", "Michael", "Mike from engineering"),
        ("Dr. Patel", "my thesis advisor", "Patel"),
        ("Alex", "Alex Kim", "my friend Alex"),
    ]
    
    PROJECTS = [
        ("Project Athena", "the ML project", "Athena"),
        ("EdgeMemory", "the memory system", "our side project"),
        ("Phoenix", "the legacy migration", "Phoenix project"),
    ]
    
    TOPICS = ["career", "health", "learning", "relationships", "finances", "projects"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.memories = []
        self.queries = []
        self.base_date = datetime(2024, 1, 1)
    
    def generate_dataset(self) -> Tuple[List[SyntheticMemory], List[SyntheticQuery]]:
        """Generate complete evaluation dataset."""
        self._generate_episodic_chain()       # 20 memories: Daily work events
        self._generate_belief_evolution()     # 15 memories: Opinion changes
        self._generate_causal_chain()         # 15 memories: X led to Y led to Z
        self._generate_entity_mentions()      # 20 memories: Same person, different names
        self._generate_semantic_facts()       # 15 memories: Learned facts/skills
        self._generate_reflections()          # 15 memories: Self-reflection
        
        self._generate_queries()              # 50 queries across all types
        
        return self.memories, self.queries
    
    def _generate_episodic_chain(self):
        """Generate a chain of episodic work memories."""
        events = [
            "Had kickoff meeting for {project}. {person} seemed optimistic about timeline.",
            "First sprint planning done. Taking on the data pipeline task.",
            "Debugging the embedding inference code. {person} helped identify the issue.",
            "Demo day - showed {project} progress to stakeholders. Positive feedback!",
            "Got blocked on infrastructure issues. Waiting on DevOps team.",
            "Breakthrough! Found that batching reduces latency by 60%.",
            "Mid-project review with {person}. On track for delivery.",
            "Integration testing revealed edge cases we hadn't considered.",
            "{project} MVP completed. {person} approved for production trial.",
            "Production deploy successful. Monitoring metrics look good.",
        ]
        
        project = random.choice(self.PROJECTS)
        person = random.choice(self.PERSONS)
        start_id = len(self.memories)
        
        for i, event in enumerate(events):
            # Replace placeholders
            content = event.format(
                project=project[random.randint(0, len(project)-1)],
                person=person[random.randint(0, len(person)-1)]
            )
            
            self.memories.append(SyntheticMemory(
                content=content,
                timestamp=self.base_date + timedelta(days=i*3),
                ground_truth_type="episodic",
                ground_truth_entities=[project[0], person[0]],
                ground_truth_topic="projects",
                importance=0.5 + random.random() * 0.3,
                related_memory_ids=list(range(start_id, start_id + i)),
                coreference_cluster=f"work_chain_{project[0]}"
            ))
        
        # Add more episodic memories (10 more)
        for i in range(10):
            self._add_random_episodic(i)
    
    def _generate_belief_evolution(self):
        """Generate beliefs that change over time."""
        evolutions = [
            # (old_belief, new_belief, topic, time_gap_months)
            (
                "I don't think AI will replace software engineers anytime soon.",
                "After using Claude, I'm starting to think AI might transform software engineering faster than I expected.",
                "career",
                6
            ),
            (
                "Remote work is less productive than in-office work.",
                "I've been much more productive working remotely - fewer interruptions and better focus.",
                "career",
                8
            ),
            (
                "I prefer React over Vue for frontend development.",
                "After using Vue 3 extensively, I actually prefer its composition API over React hooks.",
                "learning",
                4
            ),
            (
                "I should focus on breadth in my learning - know a bit of everything.",
                "Depth beats breadth. I've decided to focus deeply on ML systems this year.",
                "learning",
                12
            ),
            (
                "Exercise is hard to maintain regularly.",
                "Found a routine that works - morning workouts before checking email. Haven't missed a week in 3 months.",
                "health",
                5
            ),
        ]
        
        start_id = len(self.memories)
        
        for i, (old, new, topic, gap) in enumerate(evolutions):
            old_date = self.base_date + timedelta(days=i*40)
            new_date = old_date + timedelta(days=gap*30)
            
            # Old belief
            self.memories.append(SyntheticMemory(
                content=old,
                timestamp=old_date,
                ground_truth_type="semantic",
                ground_truth_entities=[],
                ground_truth_topic=topic,
                importance=0.7,
                related_memory_ids=[],
                coreference_cluster=f"belief_{topic}_{i}"
            ))
            
            # New belief (references old)
            self.memories.append(SyntheticMemory(
                content=new,
                timestamp=new_date,
                ground_truth_type="semantic",
                ground_truth_entities=[],
                ground_truth_topic=topic,
                importance=0.8,
                related_memory_ids=[len(self.memories) - 1],  # Points to old belief
                coreference_cluster=f"belief_{topic}_{i}"
            ))
        
        # Add 5 more belief evolution pairs
        more_evolutions = [
            ("I need to read more books to stay sharp.", 
             "Podcasts and long-form articles work better for me than books.", "learning", 3),
            ("Meditation seems like a waste of time.",
             "10 minutes of morning meditation has made a noticeable difference in my focus.", "health", 7),
            ("I should say yes to more opportunities.",
             "Saying no to good opportunities opens space for great ones. Being selective.", "career", 9),
            ("Networking events feel artificial.",
             "Found my style - small group discussions beat large networking events.", "relationships", 6),
            ("I prefer working alone on technical problems.",
             "Pair programming actually accelerates my learning on complex problems.", "learning", 4),
        ]
        
        for old, new, topic, gap in more_evolutions:
            old_date = self.base_date + timedelta(days=len(self.memories)*10)
            new_date = old_date + timedelta(days=gap*30)
            
            self.memories.append(SyntheticMemory(
                content=old, timestamp=old_date, ground_truth_type="semantic",
                ground_truth_entities=[], ground_truth_topic=topic, importance=0.7,
                related_memory_ids=[], coreference_cluster=f"belief_{topic}_{len(self.memories)}"
            ))
            self.memories.append(SyntheticMemory(
                content=new, timestamp=new_date, ground_truth_type="semantic",
                ground_truth_entities=[], ground_truth_topic=topic, importance=0.8,
                related_memory_ids=[len(self.memories) - 1],
                coreference_cluster=f"belief_{topic}_{len(self.memories)}"
            ))
    
    def _generate_causal_chain(self):
        """Generate explicit causal chains."""
        chains = [
            [
                "Noticed I was feeling burned out after 3 months of overtime.",
                "Because of the burnout, I decided to talk to Sarah about workload.",
                "Sarah agreed to redistribute tasks. That conversation led to a new team process.",
                "The new process reduced my weekly hours. I'm feeling more balanced now."
            ],
            [
                "Read an article about system design interviews being a weakness for many candidates.",
                "Started studying system design because I want to be better prepared for interviews.",
                "After studying system design for a month, I understood distributed systems much better.",
                "The system design knowledge helped me architect the caching layer for Project Athena."
            ],
            [
                "Mike mentioned that Python type hints improve code quality.",
                "Started adding type hints to all my code because of Mike's recommendation.",
                "Mypy caught several bugs that I would have missed otherwise.",
                "Convinced the team to adopt type hints project-wide based on my positive experience."
            ],
        ]
        
        for chain in chains:
            person = random.choice(self.PERSONS)
            chain_start = len(self.memories)
            
            for i, content in enumerate(chain):
                content = content.replace("Sarah", person[0])
                
                self.memories.append(SyntheticMemory(
                    content=content,
                    timestamp=self.base_date + timedelta(days=len(self.memories)*7),
                    ground_truth_type="episodic" if i == 0 else "reflective",
                    ground_truth_entities=[person[0]] if person[0] in content else [],
                    ground_truth_topic="career",
                    importance=0.6 + i * 0.1,
                    related_memory_ids=list(range(chain_start, chain_start + i)),
                    coreference_cluster=f"causal_chain_{chain_start}"
                ))
    
    def _generate_entity_mentions(self):
        """Generate memories referring to same entity with different names."""
        # Pick one person and create 10 memories with various names
        person = self.PERSONS[0]  # Sara
        mentions = [
            f"{person[0]} gave excellent feedback on my presentation.",
            f"Had 1:1 with {person[1]} - discussed career growth.",
            f"{person[2]} approved my PTO request.",
            f"Working late with {person[0]} to finish the deadline.",
            f"{person[1]} mentioned our team is up for an award.",
            f"Got praise from {person[2]} for the bug fix.",
            f"Lunch with {person[0]} - learned about her MBA journey.",
            f"{person[1]} is pushing for headcount increase.",
            f"Team offsite organized by {person[2]} was great.",
            f"Annual review with {person[0]} - exceeds expectations!",
        ]
        
        for i, content in enumerate(mentions):
            self.memories.append(SyntheticMemory(
                content=content,
                timestamp=self.base_date + timedelta(days=i*5),
                ground_truth_type="episodic",
                ground_truth_entities=[person[0]],  # Ground truth: canonical name
                ground_truth_topic="career",
                importance=0.5 + random.random() * 0.3,
                related_memory_ids=[],
                coreference_cluster=f"entity_{person[0]}"
            ))
        
        # Add 10 more entity mentions for projects
        project = self.PROJECTS[0]
        for i in range(10):
            variation = project[random.randint(0, len(project)-1)]
            content = random.choice([
                f"Sprint planning for {variation} went smoothly.",
                f"Blocked on {variation} - waiting for API approval.",
                f"Good progress on {variation} this week.",
                f"Deployed latest {variation} changes to staging.",
                f"{variation} metrics looking positive.",
            ])
            
            self.memories.append(SyntheticMemory(
                content=content,
                timestamp=self.base_date + timedelta(days=i*4),
                ground_truth_type="episodic",
                ground_truth_entities=[project[0]],
                ground_truth_topic="projects",
                importance=0.5,
                related_memory_ids=[],
                coreference_cluster=f"entity_{project[0]}"
            ))
    
    def _generate_semantic_facts(self):
        """Generate learned facts and skills."""
        facts = [
            ("Transformer attention is O(n²) in sequence length.", "learning"),
            ("FAISS IVF index is faster for large-scale search.", "learning"),
            ("DuckDB supports window functions like PostgreSQL.", "learning"),
            ("The Pomodoro technique: 25 min work, 5 min break.", "learning"),
            ("Best practice: log errors with stack traces, not just messages.", "learning"),
            ("Batch inference is 10x faster than sequential for embeddings.", "learning"),
            ("RRF fusion formula: 1/(k+rank) summed across retrievers.", "learning"),
            ("SetFit can fine-tune with just 8 examples per class.", "learning"),
            ("SQLite concurrent reads are fine; writes need mutex.", "learning"),
            ("BM25 is still competitive with dense retrieval on keyword-heavy queries.", "learning"),
            ("Regular sleep schedule matters more than total hours.", "health"),
            ("Strength training 3x/week maintains muscle during cutting.", "health"),
            ("Batch cooking on Sunday saves 5+ hours weekly.", "health"),
            ("Tax-advantaged accounts: 401k → IRA → HSA order.", "finances"),
            ("Emergency fund should be 6 months expenses.", "finances"),
        ]
        
        for content, topic in facts:
            self.memories.append(SyntheticMemory(
                content=content,
                timestamp=self.base_date + timedelta(days=len(self.memories)*3),
                ground_truth_type="semantic",
                ground_truth_entities=[],
                ground_truth_topic=topic,
                importance=0.6,
                related_memory_ids=[],
                coreference_cluster=f"fact_{topic}"
            ))
    
    def _generate_reflections(self):
        """Generate self-reflective memories."""
        reflections = [
            "Looking back at this quarter, I've grown most in system design.",
            "I notice I'm more patient in code reviews than I used to be.",
            "My communication skills have improved through the presentations.",
            "I tend to underestimate tasks by about 30% - need to account for this.",
            "I work best in the morning. Should protect that time for deep work.",
            "I've realized I value autonomy more than I value higher pay.",
            "My strength is connecting dots across domains. Should leverage this.",
            "I notice I procrastinate on ambiguous tasks. Need clearer definition.",
            "Looking at my notes, I'm happiest when learning something new.",
            "My anxiety about public speaking has decreased significantly.",
            "I've learned to say no to meetings that don't need me.",
            "I realize I need regular 1:1 social time to stay balanced.",
            "My writing has improved by publishing weekly for 6 months.",
            "I work better with background music - no lyrics preferred.",
            "I've noticed I'm more confident in technical discussions now.",
        ]
        
        for content in reflections:
            self.memories.append(SyntheticMemory(
                content=content,
                timestamp=self.base_date + timedelta(days=len(self.memories)*5),
                ground_truth_type="reflective",
                ground_truth_entities=[],
                ground_truth_topic="learning",
                importance=0.75,
                related_memory_ids=[],
                coreference_cluster="reflections"
            ))
    
    def _add_random_episodic(self, idx: int):
        """Add a random episodic memory."""
        templates = [
            "Had coffee with {person}. Discussed {topic}.",
            "Fixed a bug in the {project} pipeline.",
            "Onboarded a new team member today.",
            "Attended tech talk on {topic}.",
            "Code review took longer than expected.",
        ]
        content = random.choice(templates).format(
            person=random.choice(self.PERSONS)[0],
            project=random.choice(self.PROJECTS)[0],
            topic=random.choice(["ML pipelines", "system design", "team processes"])
        )
        
        self.memories.append(SyntheticMemory(
            content=content,
            timestamp=self.base_date + timedelta(days=len(self.memories)*2),
            ground_truth_type="episodic",
            ground_truth_entities=[],
            ground_truth_topic="career",
            importance=0.4,
            related_memory_ids=[],
            coreference_cluster="random_episodic"
        ))
    
    def _generate_queries(self):
        """Generate evaluation queries with ground truth."""
        
        # Timeline queries
        self.queries.extend([
            SyntheticQuery(
                query="What happened in my first week working on the project?",
                query_type="timeline",
                expected_memory_ids=[0, 1, 2],  # First few memories
                expected_answer_contains=["kickoff", "sprint", "meeting"],
                ground_truth_answer="In the first week, you had the kickoff meeting and started sprint planning."
            ),
            SyntheticQuery(
                query="What did I accomplish last month?",
                query_type="timeline",
                expected_memory_ids=list(range(5, 15)),
                expected_answer_contains=["demo", "progress", "deploy"],
                ground_truth_answer="You demonstrated progress, completed integrations, and had a successful deployment."
            ),
        ])
        
        # Causal queries
        self.queries.extend([
            SyntheticQuery(
                query="Why did I start studying system design?",
                query_type="causal",
                expected_memory_ids=[],  # Depends on actual IDs
                expected_answer_contains=["interview", "weakness", "preparation"],
                ground_truth_answer="You read about system design being a common weakness and wanted to be better prepared."
            ),
            SyntheticQuery(
                query="What led to the team adopting type hints?",
                query_type="causal",
                expected_memory_ids=[],
                expected_answer_contains=["Mike", "bugs", "mypy"],
                ground_truth_answer="Mike recommended type hints, you tried them, mypy caught bugs, and you convinced the team."
            ),
        ])
        
        # Belief evolution queries
        self.queries.extend([
            SyntheticQuery(
                query="How has my view on remote work changed?",
                query_type="reflection",
                expected_memory_ids=[],
                expected_answer_contains=["productive", "changed", "focus"],
                ground_truth_answer="Initially skeptical, you now find remote work more productive due to fewer interruptions."
            ),
            SyntheticQuery(
                query="What do I believe about depth vs breadth in learning?",
                query_type="reflection",
                expected_memory_ids=[],
                expected_answer_contains=["depth", "focus", "ML"],
                ground_truth_answer="You shifted from preferring breadth to focusing deeply on ML systems."
            ),
        ])
        
        # Entity recall queries
        self.queries.extend([
            SyntheticQuery(
                query="What do I know about Sarah?",
                query_type="recall",
                expected_memory_ids=[],
                expected_answer_contains=["manager", "feedback", "review"],
                ground_truth_answer="Sarah is your manager who gives excellent feedback and approved your exceeds expectations review."
            ),
            SyntheticQuery(
                query="Tell me about Project Athena progress.",
                query_type="recall",
                expected_memory_ids=[],
                expected_answer_contains=["ML", "caching", "progress"],
                ground_truth_answer="Project Athena is the ML project that showed positive progress and includes a caching layer."
            ),
        ])
        
        # Add more queries to reach 50
        additional_queries = [
            ("When did I feel burned out?", "timeline", ["overtime", "burnout", "months"]),
            ("Why did I talk to Sarah about workload?", "causal", ["burnout", "overtime"]),
            ("What meetings did I have?", "recall", ["kickoff", "planning", "1:1"]),
            ("How did my testing approach evolve?", "reflection", ["integration", "edge cases"]),
            ("What did Mike recommend?", "recall", ["type hints", "Python"]),
            ("What health habits have I developed?", "reflection", ["exercise", "morning", "meditation"]),
            ("When was the project deployed?", "timeline", ["deploy", "production"]),
            ("What did I learn about transformers?", "recall", ["attention", "O(n²)"]),
            ("Why do I prefer morning work?", "causal", ["focus", "deep work"]),
            ("What technical skills improved?", "reflection", ["system design", "communication"]),
        ]
        
        for q, qtype, expected in additional_queries:
            self.queries.append(SyntheticQuery(
                query=q,
                query_type=qtype,
                expected_memory_ids=[],
                expected_answer_contains=expected,
                ground_truth_answer=""
            ))
    
    def save_dataset(self, path: str):
        """Save dataset to JSON for reproducibility."""
        data = {
            "memories": [
                {
                    "id": i,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "ground_truth_type": m.ground_truth_type,
                    "ground_truth_entities": m.ground_truth_entities,
                    "ground_truth_topic": m.ground_truth_topic,
                    "importance": m.importance,
                    "related_memory_ids": m.related_memory_ids,
                    "coreference_cluster": m.coreference_cluster
                }
                for i, m in enumerate(self.memories)
            ],
            "queries": [
                {
                    "query": q.query,
                    "query_type": q.query_type,
                    "expected_memory_ids": q.expected_memory_ids,
                    "expected_answer_contains": q.expected_answer_contains,
                    "ground_truth_answer": q.ground_truth_answer
                }
                for q in self.queries
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
```

---

**File: `evaluation/evaluator.py`**

```python
"""
Automated evaluation framework for EdgeMemory.

Metrics:
1. Retrieval metrics (Precision@k, Recall@k, MRR)
2. Classification accuracy (memory type, emotion, importance)
3. Entity resolution accuracy (coreference F1)
4. Belief evolution detection accuracy
5. End-to-end answer quality (ROUGE, BERTScore)
6. Latency benchmarks
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time
import json
from collections import defaultdict

from evaluation.synthetic_dataset import SyntheticDatasetGenerator, SyntheticQuery

@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    retrieval_precision_at_5: float
    retrieval_precision_at_10: float
    retrieval_recall_at_10: float
    retrieval_mrr: float
    
    classification_accuracy: float
    entity_resolution_f1: float
    belief_evolution_accuracy: float
    
    answer_contains_rate: float  # % of answers containing expected keywords
    
    avg_latency_ms: float
    p95_latency_ms: float
    
    total_memories: int
    total_queries: int

class EdgeMemoryEvaluator:
    """Evaluate EdgeMemory system against ground truth."""
    
    def __init__(self, pipeline):
        """
        Args:
            pipeline: EdgeMemoryPipeline instance to evaluate
        """
        self.pipeline = pipeline
        self.generator = SyntheticDatasetGenerator()
    
    def run_full_evaluation(self) -> EvaluationResult:
        """Run complete evaluation suite."""
        # Generate dataset
        memories, queries = self.generator.generate_dataset()
        
        # Ingest all memories
        print(f"Ingesting {len(memories)} synthetic memories...")
        memory_id_map = self._ingest_memories(memories)
        
        # Run query evaluation
        print(f"Evaluating {len(queries)} queries...")
        query_results = self._evaluate_queries(queries, memory_id_map)
        
        # Calculate metrics
        retrieval_metrics = self._calculate_retrieval_metrics(query_results)
        classification_metrics = self._calculate_classification_metrics(memories, memory_id_map)
        entity_metrics = self._calculate_entity_metrics(memories, memory_id_map)
        belief_metrics = self._calculate_belief_metrics(memories, memory_id_map)
        latency_metrics = self._calculate_latency_metrics(query_results)
        
        return EvaluationResult(
            retrieval_precision_at_5=retrieval_metrics["precision@5"],
            retrieval_precision_at_10=retrieval_metrics["precision@10"],
            retrieval_recall_at_10=retrieval_metrics["recall@10"],
            retrieval_mrr=retrieval_metrics["mrr"],
            classification_accuracy=classification_metrics["accuracy"],
            entity_resolution_f1=entity_metrics["f1"],
            belief_evolution_accuracy=belief_metrics["accuracy"],
            answer_contains_rate=query_results["answer_contains_rate"],
            avg_latency_ms=latency_metrics["avg"],
            p95_latency_ms=latency_metrics["p95"],
            total_memories=len(memories),
            total_queries=len(queries)
        )
    
    def _ingest_memories(self, memories: List) -> Dict[int, str]:
        """Ingest memories and return mapping from synthetic ID to system ID."""
        id_map = {}
        for i, mem in enumerate(memories):
            result = self.pipeline.ingest_text(
                mem.content,
                timestamp=mem.timestamp
            )
            id_map[i] = result.event_id
        return id_map
    
    def _evaluate_queries(
        self, 
        queries: List[SyntheticQuery],
        memory_id_map: Dict[int, str]
    ) -> Dict[str, Any]:
        """Evaluate all queries and collect results."""
        results = {
            "query_results": [],
            "answer_contains_rate": 0.0
        }
        
        contains_count = 0
        
        for q in queries:
            start = time.time()
            response = self.pipeline.query(q.query)
            latency = (time.time() - start) * 1000
            
            # Check if answer contains expected keywords
            answer_lower = response["answer"].lower()
            contains_all = all(
                kw.lower() in answer_lower 
                for kw in q.expected_answer_contains
            )
            if contains_all:
                contains_count += 1
            
            # Map expected memory IDs to system IDs
            expected_system_ids = [
                memory_id_map[eid] 
                for eid in q.expected_memory_ids 
                if eid in memory_id_map
            ]
            
            # Get retrieved IDs
            retrieved_ids = [
                e.memory_id for e in response.get("evidence", [])
            ]
            
            results["query_results"].append({
                "query": q.query,
                "query_type": q.query_type,
                "expected_ids": expected_system_ids,
                "retrieved_ids": retrieved_ids,
                "answer": response["answer"],
                "latency_ms": latency,
                "contains_expected": contains_all
            })
        
        results["answer_contains_rate"] = contains_count / len(queries)
        return results
    
    def _calculate_retrieval_metrics(self, query_results: Dict) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        precision_at_5 = []
        precision_at_10 = []
        recall_at_10 = []
        mrr_values = []
        
        for qr in query_results["query_results"]:
            expected = set(qr["expected_ids"])
            if not expected:
                continue
            
            retrieved = qr["retrieved_ids"]
            
            # Precision@5
            hits_5 = len(set(retrieved[:5]) & expected)
            precision_at_5.append(hits_5 / 5)
            
            # Precision@10
            hits_10 = len(set(retrieved[:10]) & expected)
            precision_at_10.append(hits_10 / 10)
            
            # Recall@10
            recall_at_10.append(hits_10 / len(expected))
            
            # MRR
            for rank, rid in enumerate(retrieved, 1):
                if rid in expected:
                    mrr_values.append(1.0 / rank)
                    break
            else:
                mrr_values.append(0.0)
        
        return {
            "precision@5": sum(precision_at_5) / len(precision_at_5) if precision_at_5 else 0,
            "precision@10": sum(precision_at_10) / len(precision_at_10) if precision_at_10 else 0,
            "recall@10": sum(recall_at_10) / len(recall_at_10) if recall_at_10 else 0,
            "mrr": sum(mrr_values) / len(mrr_values) if mrr_values else 0
        }
    
    def _calculate_classification_metrics(
        self, 
        memories: List,
        memory_id_map: Dict
    ) -> Dict[str, float]:
        """Compare predicted memory types to ground truth."""
        correct = 0
        total = 0
        
        for i, mem in enumerate(memories):
            system_id = memory_id_map.get(i)
            if not system_id:
                continue
            
            stored = self.pipeline.get_memory(system_id)
            if stored and stored.get("type") == mem.ground_truth_type:
                correct += 1
            total += 1
        
        return {"accuracy": correct / total if total > 0 else 0}
    
    def _calculate_entity_metrics(
        self, 
        memories: List,
        memory_id_map: Dict
    ) -> Dict[str, float]:
        """Evaluate entity extraction and resolution."""
        # Group memories by coreference cluster
        clusters = defaultdict(list)
        for i, mem in enumerate(memories):
            if mem.coreference_cluster.startswith("entity_"):
                clusters[mem.coreference_cluster].append(i)
        
        # Check if system groups them correctly (same canonical entity)
        correct_pairs = 0
        total_pairs = 0
        
        for cluster_name, mem_indices in clusters.items():
            # Get canonical entities for each memory in cluster
            canonical_ids = []
            for idx in mem_indices:
                system_id = memory_id_map.get(idx)
                if system_id:
                    stored = self.pipeline.get_memory(system_id)
                    entities = stored.get("entities", [])
                    if entities:
                        canonical_ids.append(entities[0].get("canonical_id", "unknown"))
            
            # Count pairs that share same canonical ID
            for i in range(len(canonical_ids)):
                for j in range(i + 1, len(canonical_ids)):
                    total_pairs += 1
                    if canonical_ids[i] == canonical_ids[j]:
                        correct_pairs += 1
        
        precision = correct_pairs / total_pairs if total_pairs > 0 else 0
        # For simplicity, F1 ≈ precision (assuming balanced recall)
        return {"f1": precision}
    
    def _calculate_belief_metrics(
        self, 
        memories: List,
        memory_id_map: Dict
    ) -> Dict[str, float]:
        """Evaluate belief evolution detection."""
        # Find belief evolution pairs in ground truth
        evolution_pairs = []
        for i, mem in enumerate(memories):
            if mem.related_memory_ids:
                for related_id in mem.related_memory_ids:
                    if memories[related_id].coreference_cluster.startswith("belief_"):
                        evolution_pairs.append((related_id, i))
        
        if not evolution_pairs:
            return {"accuracy": 0.0}
        
        # Check if system detected these evolutions
        detected = 0
        for old_idx, new_idx in evolution_pairs:
            new_system_id = memory_id_map.get(new_idx)
            if new_system_id:
                stored = self.pipeline.get_memory(new_system_id)
                if stored and stored.get("supersedes"):
                    detected += 1
        
        return {"accuracy": detected / len(evolution_pairs)}
    
    def _calculate_latency_metrics(self, query_results: Dict) -> Dict[str, float]:
        """Calculate latency statistics."""
        latencies = [qr["latency_ms"] for qr in query_results["query_results"]]
        
        if not latencies:
            return {"avg": 0, "p95": 0}
        
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        
        return {
            "avg": sum(latencies) / len(latencies),
            "p95": latencies[p95_idx]
        }
    
    def generate_report(self, result: EvaluationResult) -> str:
        """Generate human-readable evaluation report."""
        return f"""
═══════════════════════════════════════════════════════════════
                    EdgeMemory Evaluation Report
═══════════════════════════════════════════════════════════════

Dataset: {result.total_memories} memories, {result.total_queries} queries

RETRIEVAL METRICS
─────────────────
  Precision@5:  {result.retrieval_precision_at_5:.2%}
  Precision@10: {result.retrieval_precision_at_10:.2%}
  Recall@10:    {result.retrieval_recall_at_10:.2%}
  MRR:          {result.retrieval_mrr:.3f}

CLASSIFICATION METRICS
──────────────────────
  Memory Type Accuracy:    {result.classification_accuracy:.2%}
  Entity Resolution F1:    {result.entity_resolution_f1:.2%}
  Belief Evolution Detect: {result.belief_evolution_accuracy:.2%}

ANSWER QUALITY
──────────────
  Contains Expected: {result.answer_contains_rate:.2%}

PERFORMANCE
───────────
  Avg Latency:  {result.avg_latency_ms:.0f}ms
  P95 Latency:  {result.p95_latency_ms:.0f}ms

═══════════════════════════════════════════════════════════════
"""
```

**Tasks:**
- [ ] Implement synthetic dataset generator with 100+ memories
- [ ] Create 50 ground-truth queries across all query types
- [ ] Build automated evaluation pipeline
- [ ] Add retrieval metrics (Precision, Recall, MRR)
- [ ] Add classification and entity resolution metrics
- [ ] Create latency benchmarks
- [ ] Generate human-readable evaluation reports
- [ ] Integrate with CI/CD for regression testing

---

### Phase 10: Optimization & Deployment (Week 10-12)

#### Step 10.1: Configuration

**File: `configs/config.yaml`**

```yaml
# EdgeMemory Configuration

# LLM Settings
llm:
  model: "mistral:7b-instruct"  # or "phi3:mini" for faster inference
  temperature: 0.7
  max_tokens: 1024

# ASR Settings
asr:
  model_size: "base"  # tiny, base, small, medium, large-v2
  device: "cpu"
  compute_type: "int8"  # int8, float16, float32

# Embedding Settings
embeddings:
  model: "bge-small"  # bge-small, bge-base, e5-small, gte-small
  device: "cpu"

# Storage Paths
storage:
  db_path: "data/memories.db"
  vector_path: "data/vectors"
  graph_path: "data/knowledge.pkl"
  audio_path: "data/audio"

# Retrieval Settings
retrieval:
  dense_k: 20
  sparse_k: 20
  graph_k: 10
  sql_k: 20
  fusion_weights:
    dense: 0.4
    sparse: 0.2
    graph: 0.2
    sql: 0.2

# Agent Settings
agents:
  enabled:
    - timeline
    - causal
    - reflection
    - arbitration

# Background Processing
background:
  reflection_interval_hours: 24
  pattern_window_days: 7
  consolidation_threshold: 0.8
```

---

## Part 4: Timeline & Milestones

**Updated Timeline (16-18 weeks):** The original 12-week estimate was optimistic. With proper entity resolution, memory consolidation, evaluation framework, and agent orchestration, a realistic timeline is 16-18 weeks for a production-quality system.

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1-2 | Foundation | Project structure, environment, data models, test harness |
| 2-3 | Ingestion (Core) | ASR integration, basic memory builder |
| 3-4 | Ingestion (Extended) | Lightweight classifiers, entity extraction + resolution |
| 4-5 | Storage | Vector store, relational DB, knowledge graph |
| 5-6 | Retrieval | Dense, sparse, graph, SQL retrievers + RRF fusion |
| 6-7 | Memory Consolidation | Hierarchical summarization, compression pipeline |
| 7-8 | Agents (Core) | Timeline, causal, arbitration agents |
| 8-9 | Agent Orchestrator | Multi-agent routing, chaining, fallback |
| 9-10 | LLM Integration | Local LLM setup, prompt engineering |
| 10-11 | Memory Evolution | Multi-stage belief tracking, conflict resolution |
| 11-12 | E2E Integration | Pipeline orchestrator, API, background jobs |
| 12-14 | Evaluation | Synthetic dataset, automated benchmarks, regression tests |
| 14-16 | Optimization | Performance tuning, memory profiling, latency reduction |
| 16-18 | Polish & Deploy | Documentation, error handling, edge deployment |

### Critical Path Dependencies

```
Foundation → Ingestion → Storage → Retrieval → Agents → Integration
                ↓                      ↓
         Entity Resolver         Memory Consolidation
                                       ↓
                               Evaluation Framework
```

### Milestone Checkpoints

| Milestone | Week | Success Criteria |
|-----------|------|------------------|
| **M1: First Memory** | 3 | Ingest text, store, retrieve by similarity |
| **M2: First Query** | 6 | Natural language query returns relevant memories |
| **M3: Belief Evolution** | 11 | System detects and tracks opinion changes |
| **M4: Causal Chain** | 9 | "Why" queries return causal explanations |
| **M5: 1000 Memories** | 14 | System handles 1000+ memories without degradation |
| **M6: Evaluation Pass** | 14 | Precision@10 > 0.7, Latency < 3s |
| **M7: Production Ready** | 18 | Full test coverage, documentation, deployment guide |

---

## Part 5: Success Metrics

### Retrieval Quality Metrics (measured via evaluation framework)
| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Precision@5 | > 0.65 | Synthetic dataset queries |
| Precision@10 | > 0.55 | Synthetic dataset queries |
| Recall@10 | > 0.70 | Synthetic dataset queries |
| MRR | > 0.60 | Synthetic dataset queries |

### Classification Accuracy
| Component | Target | Measurement |
|-----------|--------|-------------|
| Memory Type Classification | > 80% | Compare lightweight classifier vs ground truth |
| Emotion Detection | > 75% | Compare predicted emotion vs ground truth |
| Entity Resolution F1 | > 70% | Coreference cluster matching |
| Belief Evolution Detection | > 75% | Detected vs ground truth evolution pairs |

### Performance Metrics
| Metric | Target | Conditions |
|--------|--------|------------|
| Ingestion Latency | < 500ms | Per memory (text input) |
| Query Latency (avg) | < 2000ms | On consumer laptop |
| Query Latency (P95) | < 3500ms | On consumer laptop |
| Memory Footprint | < 4GB RAM | With 10,000 memories loaded |
| Storage Efficiency | < 1GB per year | With daily journaling |

### System Quality
- [ ] 100% offline operation (no network calls required)
- [ ] Runs on consumer laptop (8GB RAM, no GPU required)
- [ ] LLM fallback rate < 15% (most classification done by lightweight models)
- [ ] Zero data leakage (all processing local)

### Evaluation Automation
- [ ] Synthetic dataset: 100 memories, 50 queries generated
- [ ] Automated regression testing in CI/CD
- [ ] Benchmark comparison between model versions
- [ ] Human evaluation protocol for subjective quality

---

## Part 6: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM hallucination | LLM only reasons over retrieved evidence; never invents |
| Slow inference | Use quantized models (Q4/Q5), Phi-3 Mini as fallback |
| Storage growth | Compression, summarization, configurable retention |
| Embedding drift | Periodic re-indexing, model versioning |
| Complex queries | Multi-hop retrieval, agent decomposition |

---

## Part 7: Future Extensions

1. **Mobile deployment**: ONNX export, on-device inference
2. **Real-time streaming**: Continuous ASR with voice activity detection
3. **Multi-modal**: Image memories, document ingestion
4. **Federated learning**: Cross-device memory synchronization (encrypted)
5. **Pattern prediction**: Proactive insights based on detected patterns

---

## Part 8: Web UI — EdgeMemory Command Center (Critical Addition)

### 8.1 Why a Web UI Is Essential

EdgeMemory without a control interface is unusable. The Web UI serves as the **single pane of glass** to:
- Ingest memories (text/voice), browse/search all memories
- Query the memory system with a chat-like interface
- Visualize the knowledge graph and causal chains
- Monitor belief evolution over time
- Verify each integration (Ollama, FAISS, DuckDB, embeddings, ASR) is healthy
- Configure system settings (LLM model, retrieval weights, consolidation thresholds)

### 8.2 Web UI Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EDGEMEMORY WEB COMMAND CENTER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │    FRONTEND (SPA)    │    │        FASTAPI BACKEND                   │   │
│  │                      │    │                                          │   │
│  │  ┌────────────────┐  │    │  ┌──────────┐  ┌─────────────────────┐  │   │
│  │  │   Dashboard    │  │◄──►│  │ REST API │  │ WebSocket (live)    │  │   │
│  │  │   (Health +    │  │    │  │ /api/*   │  │ /ws (status,stream) │  │   │
│  │  │    Stats)      │  │    │  └──────────┘  └─────────────────────┘  │   │
│  │  ├────────────────┤  │    │        │                                 │   │
│  │  │ Memory Input   │  │    │        ▼                                 │   │
│  │  │ (Text/Voice)   │  │    │  ┌──────────────────────────────────┐   │   │
│  │  ├────────────────┤  │    │  │     EdgeMemoryPipeline           │   │   │
│  │  │ Memory Browser │  │    │  │  (Full backend orchestration)    │   │   │
│  │  │ + Timeline     │  │    │  └──────────────────────────────────┘   │   │
│  │  ├────────────────┤  │    │                                          │   │
│  │  │ Query Chat     │  │    └──────────────────────────────────────────┘   │
│  │  │ Interface      │  │                                                   │
│  │  ├────────────────┤  │                                                   │
│  │  │ Knowledge      │  │                                                   │
│  │  │ Graph Viewer   │  │                                                   │
│  │  ├────────────────┤  │                                                   │
│  │  │ Belief         │  │                                                   │
│  │  │ Evolution      │  │                                                   │
│  │  ├────────────────┤  │                                                   │
│  │  │ System         │  │                                                   │
│  │  │ Settings       │  │                                                   │
│  │  └────────────────┘  │                                                   │
│  └──────────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Web UI Pages

| Page | Purpose | Key Features |
|------|---------|-------------|
| **Dashboard** | System overview | Component health checks (Ollama, FAISS, DuckDB, Whisper), memory count, storage stats, recent activity |
| **Memory Ingestion** | Create new memories | Text input area, voice recording (via browser MediaRecorder API), real-time classification preview |
| **Memory Browser** | Browse all memories | Timeline view, filter by type/topic/emotion, search, pagination, memory detail view with causal links |
| **Query Interface** | Ask questions | Chat-like UI, shows reasoning trace, evidence cards, agent routing info, confidence scores |
| **Knowledge Graph** | Visualize entities | Interactive force-directed graph (vis.js/D3), entity detail panel, relations explorer |
| **Belief Tracker** | Track belief evolution | Timeline of contradictions/refinements, side-by-side belief comparisons, topic filter |
| **System Settings** | Configure system | LLM model selection, retrieval weights, consolidation settings, API endpoints |
| **Integration Verify** | Health checks | Step-by-step verification of each subsystem, latency benchmarks, error logs |

### 8.4 API Endpoints

```
POST   /api/memories/ingest          — Ingest text memory
POST   /api/memories/ingest-audio    — Ingest audio file
GET    /api/memories                  — List/search memories
GET    /api/memories/{id}             — Get memory detail
DELETE /api/memories/{id}             — Delete memory
POST   /api/query                    — Query the memory system
GET    /api/graph/entities            — Get knowledge graph data
GET    /api/graph/neighbors/{entity}  — Get entity neighbors
GET    /api/beliefs/timeline          — Get belief evolution timeline
GET    /api/system/health             — Full health check
GET    /api/system/stats              — System statistics
PUT    /api/system/settings           — Update settings
WS     /ws                           — WebSocket for live updates
```

### 8.5 Technology Stack

- **Backend**: FastAPI + Uvicorn (async, WebSocket support, auto-docs)
- **Frontend**: Single-page HTML/CSS/JS (no npm/build step needed)
- **Charts**: Chart.js for statistics
- **Graph**: vis.js for knowledge graph visualization
- **Real-time**: WebSocket for live ingestion status and health monitoring

---

## Quick Start Commands

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Install Ollama and pull models
ollama pull phi3
ollama pull nomic-embed-text

# 3. Start the system
python -m src.main

# 4. Open Web UI
# Navigate to http://localhost:8000
```

---

**This plan provides a complete roadmap from concept to implementation for EdgeMemory — your personal causal intelligence system with a full Web Command Center.**
