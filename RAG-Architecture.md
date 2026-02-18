# Cortex Lab: Advanced Agentic RAG Architecture
## Production-Ready Implementation Guide (2025)

> **Last Updated:** February 18, 2026  
> **Version:** 2.0  
> **Target Hardware:** NVIDIA GTX 1650 (4GB VRAM)  
> **Model:** DeepSeek-R1-1.5B (Fine-tuned)

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Agentic RAG: Paradigm Shift](#2-agentic-rag-paradigm-shift)
3. [Complete System Architecture](#3-complete-system-architecture)
4. [Phase 1: Multi-Level Indexing](#4-phase-1-multi-level-indexing)
5. [Phase 2: Query Intelligence Layer](#5-phase-2-query-intelligence-layer)
6. [Phase 3: Agentic Retrieval Engine](#6-phase-3-agentic-retrieval-engine)
7. [Phase 4: Multi-Channel Hybrid Retrieval](#7-phase-4-multi-channel-hybrid-retrieval)
8. [Phase 5: Self-Reflective Generation](#8-phase-5-self-reflective-generation)
9. [Phase 6: Belief Evolution & Memory Consolidation](#9-phase-6-belief-evolution--memory-consolidation)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Production Deployment](#12-production-deployment)

---

## 1. Executive Summary

### 1.1 What Makes This RAG System Different?

Cortex Lab implements **Agentic RAG** - a paradigm shift from traditional RAG systems that moves beyond simple retrieve-and-generate patterns to a sophisticated multi-agent architecture with:

| Traditional RAG | Cortex Lab Agentic RAG |
|----------------|------------------------|
| Single retrieval pass | Multi-hop iterative retrieval with agent coordination |
| Fixed retrieval strategy | Adaptive routing based on query complexity |
| Top-K similarity only | Multi-channel fusion (dense + sparse + graph + temporal) |
| No quality control | Self-reflective correction (CRAG + Self-RAG) |
| Flat document chunks | Hierarchical tree indexing (RAPTOR) + propositions |
| Static memory | Dynamic belief evolution tracking |
| Session-based | Persistent long-term memory with consolidation |
| No reasoning trace | Full agent reasoning chains with evidence |

### 1.2 Research Foundations

This architecture synthesizes 15+ cutting-edge techniques from top-tier venues:

**Indexing & Storage:**
- ✅ **RAPTOR** (ICLR 2024) - Hierarchical tree-structured indexing
- ✅ **Proposition Retrieval** (EMNLP 2024) - Atomic fact-level granularity
- ✅ **GraphRAG** (Microsoft 2024) - Entity-relationship knowledge graphs

**Retrieval Techniques:**
- ✅ **ColBERTv2** (NAACL 2022) - Multi-vector late interaction
- ✅ **HyDE** (ACL 2023) - Hypothetical document embeddings
- ✅ **RAG-Fusion** (2024) - Multi-query with RRF
- ✅ **BGE Embeddings** (MTEB 2024) - State-of-the-art dense retrieval

**Agentic Components:**
- ✅ **Self-RAG** (ICLR 2024) - Self-reflective generation with critique
- ✅ **CRAG** (2024) - Corrective retrieval quality evaluation
- ✅ **FLARE** (EMNLP 2023) - Forward-looking active retrieval
- ✅ **Adaptive-RAG** (NAACL 2024) - Query complexity routing

**Memory & Evolution:**
- ✅ **Belief Evolution Tracking** - Multi-stage contradiction detection
- ✅ **Memory Consolidation** - Hierarchical summarization with decay
- ✅ **Entity Resolution** - Coreference + fuzzy matching

### 1.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Retrieval Precision@10** | > 0.75 | Multi-channel fusion |
| **Answer Faithfulness** | > 0.85 | RAGAS evaluation |
| **Query Latency (Simple)** | < 2s | Single-agent queries |
| **Query Latency (Complex)** | < 5s | Multi-agent orchestration |
| **Memory Footprint** | < 4GB | GTX 1650 compatible |
| **Classification Speed** | < 100ms | Lightweight classifiers |
| **LLM Fallback Rate** | < 15% | Most tasks handled locally |

---

## 2. Agentic RAG: Paradigm Shift

### 2.1 Traditional RAG vs Agentic RAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL RAG FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Query → Embedding → Vector Search → Top-K → LLM → Answer             │
│                                                                         │
│   Problems:                                                             │
│   • No query understanding                                              │
│   • Single retrieval pass (may miss context)                            │
│   • No quality control                                                  │
│   • Cannot handle complex multi-step reasoning                          │
│   • No memory of past interactions                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC RAG FLOW (CORTEX LAB)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. INTENT DETECTION                                                   │
│      Query → Intent Classifier → Route to Specialized Agent            │
│                                                                         │
│   2. QUERY TRANSFORMATION                                               │
│      Agent → Multi-Query Generation + HyDE + Decomposition              │
│                                                                         │
│   3. ADAPTIVE ROUTING                                                   │
│      Complexity Score → [No-Retrieval | Single-Step | Multi-Step]       │
│                                                                         │
│   4. MULTI-CHANNEL RETRIEVAL                                            │
│      Dense + Sparse + Graph + Temporal → RRF Fusion → Rerank           │
│                                                                         │
│   5. QUALITY EVALUATION (CRAG)                                          │
│      Relevance Check → [Accept | Refine | Web-Search]                  │
│                                                                         │
│   6. ITERATIVE REFINEMENT (FLARE)                                       │
│      Confidence < Threshold → Retrieve More Context                     │
│                                                                         │
│   7. SELF-REFLECTIVE GENERATION (Self-RAG)                              │
│      Generate → Critique → [Accept | Regenerate | Retrieve More]       │
│                                                                         │
│   8. MEMORY UPDATE                                                      │
│      Store Interaction → Update Beliefs → Link Causally                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CORTEX LAB AGENT HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌────────────────────┐                              │
│                         │   ORCHESTRATOR     │                              │
│                         │      AGENT         │                              │
│                         │                    │                              │
│                         │ • Intent Detection │                              │
│                         │ • Complexity Score │                              │
│                         │ • Agent Routing    │                              │
│                         │ • Result Synthesis │                              │
│                         └─────────┬──────────┘                              │
│                                   │                                         │
│              ┌────────────────────┼────────────────────┐                    │
│              │                    │                    │                    │
│              ▼                    ▼                    ▼                    │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│   │  TIMELINE AGENT  │  │  CAUSAL AGENT    │  │ REFLECTION AGENT │        │
│   │                  │  │                  │  │                  │        │
│   │ • Temporal Query │  │ • Why Questions  │  │ • Belief Changes │        │
│   │ • Chronological  │  │ • Cause-Effect   │  │ • Pattern Find   │        │
│   │ • Time Filters   │  │ • Chain Tracing  │  │ • Meta-Cognition │        │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘        │
│                                                                             │
│              ┌─────────────────────────────────────────┐                    │
│              │                                         │                    │
│              ▼                                         ▼                    │
│   ┌──────────────────┐                      ┌──────────────────┐           │
│   │ PLANNING AGENT   │                      │ ARBITRATION     │           │
│   │                  │                      │    AGENT        │           │
│   │ • Multi-Step     │                      │                  │           │
│   │ • Goal Tracking  │                      │ • Conflict Res  │           │
│   │ • Sub-Query Gen  │                      │ • Evidence Eval │           │
│   └──────────────────┘                      └──────────────────┘           │
│                                                                             │
│                         ┌────────────────────┐                              │
│                         │   RETRIEVAL        │                              │
│                         │   COORDINATOR      │                              │
│                         │                    │                              │
│                         │ • Channel Select   │                              │
│                         │ • Fusion Strategy  │                              │
│                         │ • Quality Check    │                              │
│                         └────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Agent Capabilities Matrix

| Agent | Input | Output | Retrieval Strategy | Reasoning Type |
|-------|-------|--------|-------------------|----------------|
| **Orchestrator** | Raw query | Agent assignment + parameters | Intent-based routing | Meta-reasoning |
| **Timeline** | Temporal query | Chronological narrative | Time-filtered + dense | Sequential |
| **Causal** | Why/cause query | Causal chain with evidence | Graph traversal + temporal | Causal inference |
| **Reflection** | Meta-query | Belief evolution analysis | Semantic clustering + time | Comparative |
| **Planning** | Complex query | Sub-query plan + execution | Multi-hop iterative | Decomposition |
| **Arbitration** | Conflicting info | Resolution with confidence | Evidence-weighted | Dialectical |

---

## 3. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         CORTEX LAB: AGENTIC RAG SYSTEM                                          │
│                         (Production Architecture - 2025)                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 0: INPUT ACQUISITION                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │   Text     │      │   Voice    │      │  Document  │
    │   Input    │      │   Input    │      │   Import   │
    └──────┬─────┘      └──────┬─────┘      └──────┬─────┘
           │                   │                   │
           │                   ▼                   │
           │         ┌─────────────────┐           │
           │         │ Whisper ASR     │           │
           │         │ (faster-whisper)│           │
           │         │ Base Model      │           │
           │         └────────┬────────┘           │
           │                  │                    │
           └──────────────────┴────────────────────┘
                              │
                              ▼

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 1: MEMORY INGESTION & CLASSIFICATION                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    MEMORY EVENT BUILDER (Optimized)                                     │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │              PARALLEL LIGHTWEIGHT CLASSIFIERS (~50ms total)                       │ │
    │  │                                                                                   │ │
    │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐               │ │
    │  │  │  Memory Type     │  │  Emotion         │  │  Importance      │               │ │
    │  │  │  Classifier      │  │  Classifier      │  │  Scorer          │               │ │
    │  │  │                  │  │                  │  │                  │               │ │
    │  │  │  SetFit-based    │  │  DistilBERT      │  │  Hybrid          │               │ │
    │  │  │  4-shot tuned    │  │  6 emotions      │  │  Rule+ML         │               │ │
    │  │  │  ~20ms           │  │  ~30ms           │  │  ~10ms           │               │ │
    │  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘               │ │
    │  │           │                     │                     │                         │ │
    │  │           │    Confidence < 0.7? → LLM Fallback (3s) │                         │ │
    │  │           └─────────────────────┴─────────────────────┘                         │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │              ENTITY EXTRACTION & RESOLUTION                                       │ │
    │  │                                                                                   │ │
    │  │  Stage 1: NER Extraction (spaCy + LLM)                                            │ │
    │  │  Stage 2: Coreference Resolution (context-aware)                                  │ │
    │  │  Stage 3: Fuzzy Matching (RapidFuzz, threshold=0.75)                              │ │
    │  │  Stage 4: Canonical Entity Linking (alias index)                                  │ │
    │  │                                                                                   │ │
    │  │  Output: Resolved entities with canonical IDs                                     │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │              CAUSAL LINK DETECTION                                                │ │
    │  │                                                                                   │ │
    │  │  Explicit Pattern Matching:                                                       │ │
    │  │  • "because", "led to", "resulted in", "caused" → High confidence (0.9)           │ │
    │  │                                                                                   │ │
    │  │  Inferred Causality:                                                              │ │
    │  │  • Temporal proximity (<7 days) + Semantic similarity (>0.85) → Medium (0.6)      │ │
    │  │  • User confirmation → High confidence (0.95)                                     │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 2: MULTI-REPRESENTATION STORAGE                                                          ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌───────────────────────────────────────────────────────────────────────────────────────┐
    │                    HIERARCHICAL INDEXING (RAPTOR-inspired)                            │
    │                                                                                       │
    │   Level 0: RAW MEMORIES (Full Detail)                                                │
    │   ├─ Individual memories with complete context                                       │
    │   ├─ Temporal: Last 7 days                                                           │
    │   └─ Index: Vector + SQL + Graph                                                     │
    │                                                                                       │
    │   Level 1: DAILY SUMMARIES                                                           │
    │   ├─ Clustered by topic + emotion                                                    │
    │   ├─ Temporal: 1 week - 1 month                                                      │
    │   └─ Preserves: High-importance raw memories (>0.8)                                  │
    │                                                                                       │
    │   Level 2: WEEKLY SUMMARIES                                                          │
    │   ├─ Higher abstraction, key themes                                                  │
    │   ├─ Temporal: 1-6 months                                                            │
    │   └─ Preserves: Critical events (>0.9)                                               │
    │                                                                                       │
    │   Level 3: MONTHLY SUMMARIES                                                         │
    │   ├─ Major patterns and decisions                                                    │
    │   ├─ Temporal: 6 months - 2 years                                                    │
    │   └─ Preserves: Life milestones                                                      │
    │                                                                                       │
    │   Level 4: YEARLY SUMMARIES                                                          │
    │   ├─ Life narrative, transformations                                                 │
    │   ├─ Temporal: >2 years                                                              │
    │   └─ Preserves: Pivotal moments only                                                 │
    └───────────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  VECTOR STORE    │   │  RELATIONAL DB   │   │  KNOWLEDGE       │   │  PROPOSITION     │
    │  (FAISS)         │   │  (DuckDB)        │   │  GRAPH (NX)      │   │  INDEX           │
    │                  │   │                  │   │                  │   │                  │
    │ • BGE Embeddings │   │ • Metadata       │   │ • Entities       │   │ • Atomic Facts   │
    │ • 384 dimensions │   │ • Timestamps     │   │ • Relations      │   │ • Fine-grained   │
    │ • Dense search   │   │ • Filters        │   │ • Causal Links   │   │ • High precision │
    │ • ~100ms latency │   │ • SQL queries    │   │ • Graph Traverse │   │ • Complement     │
    └──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3: QUERY INTELLIGENCE & TRANSFORMATION                                                   ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    INTENT DETECTION & COMPLEXITY SCORING                                │
    │                                                                                         │
    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │
    │   │  Query → Lightweight Classifier (SetFit) → Intent + Confidence                   │ │
    │   │                                                                                  │ │
    │   │  Intent Types:                                                                   │ │
    │   │  • TEMPORAL: "When did...", "What was I doing..."                                │ │
    │   │  • CAUSAL: "Why did...", "What led to..."                                        │ │
    │   │  • REFLECTIVE: "How did my thinking...", "What changed..."                       │ │
    │   │  • FACTUAL: "What is...", "Define..."                                            │ │
    │   │  • MULTI_STEP: Complex queries requiring decomposition                           │ │
    │   └──────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │
    │   │  Complexity Scoring (0.0 - 1.0)                                                  │ │
    │   │                                                                                  │ │
    │   │  Factors:                                                                        │ │
    │   │  • Number of entities mentioned                                                  │ │
    │   │  • Temporal span required                                                        │ │
    │   │  • Logical operators (AND, OR, IF-THEN)                                          │ │
    │   │  • Multi-hop reasoning needed                                                    │ │
    │   │  • Abstraction level (specific → pattern)                                        │ │
    │   │                                                                                  │ │
    │   │  Routing Decision:                                                               │ │
    │   │  • Simple (0.0-0.3): Direct LLM, no retrieval                                    │ │
    │   │  • Moderate (0.3-0.7): Single-step RAG                                           │ │
    │   │  • Complex (0.7-1.0): Multi-agent agentic RAG                                    │ │
    │   └──────────────────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    QUERY TRANSFORMATION PIPELINE                                        │
    │                                                                                         │
    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │
    │   │  MULTI-QUERY GENERATION (RAG-Fusion)                                             │ │
    │   │                                                                                  │ │
    │   │  Original: "Why did I quit my job in March?"                                     │ │
    │   │  ↓                                                                               │ │
    │   │  Generated Variants:                                                             │ │
    │   │  1. "What were my reasons for leaving my job in March?"                          │ │
    │   │  2. "What factors contributed to my job resignation in March?"                   │ │
    │   │  3. "What events led to my decision to quit in March?"                           │ │
    │   │  4. "What was my emotional state about work before March?"                       │ │
    │   └──────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │
    │   │  HyDE (Hypothetical Document Embeddings)                                         │ │
    │   │                                                                                  │ │
    │   │  Query: "Why did I quit my job in March?"                                        │ │
    │   │  ↓                                                                               │ │
    │   │  Generated Hypothetical Answer:                                                  │ │
    │   │  "You quit your job in March due to burnout, lack of growth                      │ │
    │   │   opportunities, and misalignment with company values. A conversation            │ │
    │   │   with a mentor helped clarify your priorities..."                               │ │
    │   │  ↓                                                                               │ │
    │   │  Embed hypothetical answer → Retrieve similar memories                           │ │
    │   └──────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │
    │   │  QUERY DECOMPOSITION (for complex queries)                                       │ │
    │   │                                                                                  │ │
    │   │  Complex Query: "How did my thinking about AI safety evolve from 2024 to 2025,   │ │
    │   │                  and what experiences caused those changes?"                     │ │
    │   │  ↓                                                                               │ │
    │   │  Sub-Questions:                                                                  │ │
    │   │  1. "What were my beliefs about AI safety in 2024?"                              │ │
    │   │  2. "What were my beliefs about AI safety in 2025?"                              │ │
    │   │  3. "What experiences or conversations happened between 2024-2025?"              │ │
    │   │  4. "Which events caused changes in my AI safety opinions?"                      │ │
    │   └──────────────────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4: AGENT ORCHESTRATION & ROUTING                                                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                         ORCHESTRATOR AGENT                                              │
    │                                                                                         │
    │  Input: Query + Intent + Complexity Score                                               │
    │  Output: Agent Assignment + Retrieval Strategy + Parameters                             │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │                     ROUTING DECISION TREE                                         │ │
    │  │                                                                                   │ │
    │  │  if complexity < 0.3:                                                             │ │
    │  │      return direct_llm_response(query)  # No retrieval needed                    │ │
    │  │                                                                                   │ │
    │  │  elif intent == "TEMPORAL":                                                       │ │
    │  │      assign(TimelineAgent)                                                        │ │
    │  │      retrieval_strategy = "temporal_filter + dense"                               │ │
    │  │                                                                                   │ │
    │  │  elif intent == "CAUSAL":                                                         │ │
    │  │      assign(CausalAgent)                                                          │ │
    │  │      retrieval_strategy = "graph_traversal + temporal + dense"                    │ │
    │  │                                                                                   │ │
    │  │  elif intent == "REFLECTIVE":                                                     │ │
    │  │      assign(ReflectionAgent)                                                      │ │
    │  │      retrieval_strategy = "semantic_clustering + belief_tracker"                  │ │
    │  │                                                                                   │ │
    │  │  elif complexity > 0.7:                                                           │ │
    │  │      assign(PlanningAgent)                                                        │ │
    │  │      retrieval_strategy = "multi_hop_iterative + fusion"                          │ │
    │  │                                                                                   │ │
    │  │  # Multi-agent coordination for complex queries                                   │ │
    │  │  if requires_multi_agent(query):                                                  │ │
    │  │      agents = select_agent_team(query)                                            │ │
    │  │      results = parallel_execute(agents)                                           │ │
    │  │      return synthesize_multi_agent_results(results)                               │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────┬───────────────────────┬───────────────────────┐
    │   TIMELINE AGENT      │   CAUSAL AGENT        │   REFLECTION AGENT    │
    ├───────────────────────┼───────────────────────┼───────────────────────┤
    │ • Parse time range    │ • Identify target     │ • Detect topic        │
    │ • Extract entities    │   event               │ • Find belief pairs   │
    │ • Query memories by   │ • Trace causal graph  │ • Classify change     │
    │   timestamp           │   backward/forward    │   type                │
    │ • Build chronological │ • Rank by confidence  │ • Generate timeline   │
    │   narrative           │ • Explain causality   │ • Compare stances     │
    └───────────────────────┴───────────────────────┴───────────────────────┘

    ┌───────────────────────┬───────────────────────────────────────────────┐
    │   PLANNING AGENT      │   ARBITRATION AGENT                           │
    ├───────────────────────┼───────────────────────────────────────────────┤
    │ • Decompose query     │ • Identify conflicting memories               │
    │ • Generate sub-goals  │ • Evaluate evidence quality                   │
    │ • Coordinate agents   │ • Check temporal recency                      │
    │ • Track progress      │ • Assign confidence weights                   │
    │ • Synthesize results  │ • Recommend resolution or ask for clarification│
    └───────────────────────┴───────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 5: MULTI-CHANNEL HYBRID RETRIEVAL                                                        ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    PARALLEL RETRIEVAL CHANNELS                                          │
    │                                                                                         │
    │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐               │
    │  │  CHANNEL 1:        │  │  CHANNEL 2:        │  │  CHANNEL 3:        │               │
    │  │  DENSE RETRIEVAL   │  │  SPARSE RETRIEVAL  │  │  GRAPH TRAVERSAL   │               │
    │  │                    │  │                    │  │                    │               │
    │  │  Model: BGE-small  │  │  Method: BM25 +    │  │  Method: NetworkX  │               │
    │  │  Embedding: 384d   │  │          SPLADE    │  │  Hops: 2-3 levels  │               │
    │  │  Top-K: 50         │  │  Top-K: 50         │  │  Top-K: 30         │               │
    │  │  Latency: ~100ms   │  │  Latency: ~50ms    │  │  Latency: ~80ms    │               │
    │  │                    │  │                    │  │                    │               │
    │  │  Strengths:        │  │  Strengths:        │  │  Strengths:        │               │
    │  │  • Semantic sim    │  │  • Exact match     │  │  • Entity relations│               │
    │  │  • Paraphrase      │  │  • Keyword search  │  │  • Multi-hop       │               │
    │  │  • Context-aware   │  │  • Fast lookup     │  │  • Causal chains   │               │
    │  └────────────────────┘  └────────────────────┘  └────────────────────┘               │
    │                                                                                         │
    │  ┌────────────────────┐  ┌────────────────────────────────────────────┐               │
    │  │  CHANNEL 4:        │  │  CHANNEL 5:                                │               │
    │  │  TEMPORAL FILTER   │  │  PROPOSITION RETRIEVAL                     │               │
    │  │                    │  │                                            │               │
    │  │  Method: SQL       │  │  Method: Atomic fact matching              │               │
    │  │  Filters:          │  │  Granularity: Sentence-level               │               │
    │  │  • Date range      │  │  Top-K: 40                                 │               │
    │  │  • Memory type     │  │  Latency: ~90ms                            │               │
    │  │  • Importance      │  │                                            │               │
    │  │  • Emotion         │  │  Strengths:                                │               │
    │  │  Top-K: 40         │  │  • Fine-grained precision                  │               │
    │  │  Latency: ~30ms    │  │  • Fact-level retrieval                    │               │
    │  └────────────────────┘  └────────────────────────────────────────────┘               │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    RECIPROCAL RANK FUSION (RRF) + SEMANTIC FUSION                       │
    │                                                                                         │
    │  Step 1: Collect ranked lists from all channels                                         │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  Dense:  [mem_15(0.89), mem_42(0.85), mem_88(0.82), ...]                               │
    │  Sparse: [mem_42(12.3), mem_15(10.8), mem_103(9.5), ...]                               │
    │  Graph:  [mem_15, mem_88, mem_103, ...]                                                │
    │  SQL:    [mem_15, mem_42, mem_88, ...]                                                 │
    │  Prop:   [mem_42, mem_88, mem_150, ...]                                                │
    │                                                                                         │
    │  Step 2: Calculate RRF scores                                                           │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  RRF(mem_i) = Σ_channel (1 / (k + rank_channel(mem_i)))    [k=60 typical]             │
    │                                                                                         │
    │  mem_15: (1/61 + 1/62 + 1/61 + 1/61 + 1/63) = 0.081                                    │
    │  mem_42: (1/62 + 1/61 + 0 + 1/62 + 1/61) = 0.065                                       │
    │  mem_88: (1/63 + 0 + 1/62 + 1/63 + 1/62) = 0.047                                       │
    │                                                                                         │
    │  Step 3: Apply channel weights (configurable)                                           │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  Final_Score = α×Dense + β×Sparse + γ×Graph + δ×SQL + ε×Prop                           │
    │                                                                                         │
    │  Default weights: α=0.35, β=0.25, γ=0.20, δ=0.10, ε=0.10                               │
    │  (Tunable based on query intent)                                                        │
    │                                                                                         │
    │  Step 4: Metadata filtering (pre-fusion)                                                │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  • Document type filter (if specified)                                                  │
    │  • Temporal constraints (date range)                                                    │
    │  • Importance threshold (e.g., >0.5)                                                    │
    │  • Source authority (trusted sources prioritized)                                       │
    │                                                                                         │
    │  Output: Top-K fused results (default K=20) for reranking                               │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 6: POST-RETRIEVAL PROCESSING & CORRECTION                                                ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    CORRECTIVE RAG (CRAG) EVALUATION                                     │
    │                                                                                         │
    │  For each retrieved memory, evaluate 3 dimensions:                                      │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  1. RELEVANCE SCORE                                                               │ │
    │  │     • Query-memory semantic similarity                                            │ │
    │  │     • Keyword overlap                                                             │ │
    │  │     • Entity match                                                                │ │
    │  │     Score: 0.0 - 1.0                                                              │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  2. SUPPORT SCORE                                                                 │ │
    │  │     • Does this memory help answer the query?                                     │ │
    │  │     • Contains necessary facts/context?                                           │ │
    │  │     • LLM judges: "Is this useful for answering the query?"                       │ │
    │  │     Score: 0.0 - 1.0                                                              │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  3. CONFIDENCE SCORE                                                              │ │
    │  │     • Memory age (recent = higher confidence)                                     │ │
    │  │     • Importance score                                                            │ │
    │  │     • Source reliability                                                          │ │
    │  │     Score: 0.0 - 1.0                                                              │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  Aggregate Decision:                                                                    │
    │  ──────────────────────────────────────────────────────────────────────────────────     │
    │  Avg_Score = (Relevance + Support + Confidence) / 3                                     │
    │                                                                                         │
    │  if Avg_Score > 0.7:                                                                    │
    │      decision = "CORRECT" → Proceed to generation                                       │
    │                                                                                         │
    │  elif 0.3 < Avg_Score < 0.7:                                                            │
    │      decision = "AMBIGUOUS" → Refine query and re-retrieve                              │
    │      • Add more context to query                                                        │
    │      • Adjust retrieval parameters (increase K)                                         │
    │      • Try different channels                                                           │
    │                                                                                         │
    │  else:  # Avg_Score < 0.3                                                               │
    │      decision = "INCORRECT" → Fallback strategy                                         │
    │      • Expand search to web (if enabled)                                                │
    │      • Ask user for clarification                                                       │
    │      • Inform user: "No relevant memories found"                                        │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    CROSS-ENCODER RERANKING                                              │
    │                                                                                         │
    │  Purpose: Fine-grained relevance scoring beyond embedding similarity                    │
    │                                                                                         │
    │  Model: BGE-reranker-base (or similar cross-encoder)                                    │
    │  Input: (query, memory_text) pairs                                                      │
    │  Output: Relevance score 0.0-1.0                                                        │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  Reranking Process:                                                               │ │
    │  │                                                                                   │ │
    │  │  1. Take top-20 from RRF fusion                                                   │ │
    │  │  2. For each candidate:                                                           │ │
    │  │     cross_encoder_score = model(query, memory)                                    │ │
    │  │  3. Combine with original fusion score:                                           │ │
    │  │     final_score = 0.6 × cross_encoder + 0.4 × rrf_score                           │ │
    │  │  4. Re-sort by final_score                                                        │ │
    │  │  5. Return top-10 for generation                                                  │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  Latency: ~200ms for 20 candidates                                                      │
    │  Trade-off: Higher accuracy vs slight latency increase                                  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 7: SELF-REFLECTIVE GENERATION (Self-RAG + FLARE)                                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    ADAPTIVE GENERATION PIPELINE                                         │
    │                                                                                         │
    │  Step 1: INITIAL GENERATION                                                             │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  Input:                                                                                 │
    │  • Query                                                                                │
    │  • Top-K retrieved memories                                                             │
    │  • Agent context (reasoning instructions)                                               │
    │                                                                                         │
    │  Model: DeepSeek-R1-1.5B (4-bit quantized, LoRA fine-tuned)                            │
    │                                                                                         │
    │  Prompt Template:                                                                       │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │ You are Cortex Lab, a personal AI with access to the user's memories.             │ │
    │  │                                                                                   │ │
    │  │ Query: {query}                                                                    │ │
    │  │                                                                                   │ │
    │  │ Retrieved Memories:                                                               │ │
    │  │ 1. [{timestamp}] {memory_1}                                                       │ │
    │  │ 2. [{timestamp}] {memory_2}                                                       │ │
    │  │ ...                                                                               │ │
    │  │                                                                                   │ │
    │  │ Agent Context: {agent_instructions}                                               │ │
    │  │                                                                                   │ │
    │  │ Instructions:                                                                     │ │
    │  │ 1. Think step-by-step using <think> tags                                          │ │
    │  │ 2. Use only the provided memories as evidence                                     │ │
    │  │ 3. If information is missing, say so explicitly                                   │ │
    │  │ 4. Cite sources by memory timestamp                                               │ │
    │  │ 5. Express confidence level (high/medium/low)                                     │ │
    │  │                                                                                   │ │
    │  │ <think>                                                                           │ │
    │  │ {model generates reasoning here}                                                  │ │
    │  │ </think>                                                                          │ │
    │  │                                                                                   │ │
    │  │ {model generates answer here}                                                     │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  Step 2: SELF-CRITIQUE (Self-RAG)                                                       │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  Evaluate generated answer on 4 dimensions:                                             │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  1. RELEVANCE: Does the answer address the query?                                 │ │
    │  │     [Relevant | Partially Relevant | Irrelevant]                                  │ │
    │  │                                                                                   │ │
    │  │  2. SUPPORT: Is the answer grounded in retrieved memories?                        │ │
    │  │     [Fully Supported | Partially Supported | Unsupported]                         │ │
    │  │                                                                                   │ │
    │  │  3. USEFULNESS: Is this answer helpful to the user?                               │ │
    │  │     [5-point scale: Very Useful → Not Useful]                                     │ │
    │  │                                                                                   │ │
    │  │  4. CONFIDENCE: How confident is the model?                                       │ │
    │  │     [High | Medium | Low]                                                         │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  Critique Prompt:                                                                       │
    │  "Evaluate the generated answer. Is it relevant, supported, useful, and confident?"    │
    │                                                                                         │
    │  Step 3: DECISION LOGIC                                                                 │
    │  ────────────────────────────────────────────────────────────────────────────────────   │
    │  if Relevance == "Relevant" AND Support == "Fully Supported" AND Confidence == "High": │
    │      return answer  # Accept as is                                                      │
    │                                                                                         │
    │  elif Support == "Partially Supported" OR Confidence == "Low":                          │
    │      # FLARE: Active Retrieval Triggered                                                │
    │      missing_info = identify_gaps(answer, retrieved_memories)                           │
    │      additional_memories = retrieve_more(missing_info, k=5)                             │
    │      regenerate_answer(query, retrieved_memories + additional_memories)                 │
    │                                                                                         │
    │  elif Relevance == "Irrelevant":                                                        │
    │      # Restart with refined query                                                       │
    │      refined_query = refine_query_with_llm(query, critique)                             │
    │      restart_pipeline(refined_query)                                                    │
    │                                                                                         │
    │  else:                                                                                  │
    │      # Multiple regeneration attempts with different prompts                            │
    │      regenerate_with_alternative_prompt(query, retrieved_memories)                      │
    │                                                                                         │
    │  Max Iterations: 3 (prevent infinite loops)                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    FORWARD-LOOKING ACTIVE RETRIEVAL (FLARE)                             │
    │                                                                                         │
    │  Purpose: Detect when generation needs more context mid-stream                          │
    │                                                                                         │
    │  Mechanism:                                                                             │
    │  1. Model generates provisional next sentence                                           │
    │  2. Calculate confidence for each token (logprobs)                                      │
    │  3. If low-confidence tokens detected (prob < 0.5):                                     │
    │     a. Identify missing context from provisional sentence                               │
    │     b. Generate clarifying query                                                        │
    │     c. Retrieve additional memories                                                     │
    │     d. Re-generate with new context                                                     │
    │  4. Continue until high confidence or max length reached                                │
    │                                                                                         │
    │  Example:                                                                               │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  Generated (provisional):                                                         │ │
    │  │  "You decided to quit because you were [LOW-CONF: unhappy?] with..."              │ │
    │  │                       ↑                                                           │ │
    │  │  Low confidence detected → Retrieve more about "job dissatisfaction"              │ │
    │  │                                                                                   │ │
    │  │  After retrieval:                                                                 │ │
    │  │  "You decided to quit because you were experiencing burnout and..."               │ │
    │  │  (High confidence now)                                                            │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  Benefits:                                                                              │
    │  • Prevents hallucination (stops when uncertain)                                        │
    │  • Dynamically gathers context                                                          │
    │  • More accurate long-form generation                                                   │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 8: MEMORY UPDATE & BELIEF EVOLUTION                                                       ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    POST-INTERACTION MEMORY UPDATE                                       │
    │                                                                                         │
    │  After each query-answer cycle:                                                         │
    │                                                                                         │
    │  1. STORE INTERACTION                                                                   │
    │     ┌─────────────────────────────────────────────────────────────────────────────────┐│
    │     │ • Query text                                                                    ││
    │     │ • Retrieved memory IDs                                                          ││
    │     │ • Agent(s) used                                                                 ││
    │     │ • Generated answer                                                              ││
    │     │ • Confidence scores                                                             ││
    │     │ • Timestamp                                                                     ││
    │     │ • User feedback (if provided)                                                   ││
    │     └─────────────────────────────────────────────────────────────────────────────────┘│
    │                                                                                         │
    │  2. DETECT BELIEF EVOLUTION                                                             │
    │     ┌─────────────────────────────────────────────────────────────────────────────────┐│
    │     │ Multi-Stage Pipeline:                                                           ││
    │     │                                                                                 ││
    │     │ Stage 1: Semantic Similarity                                                    ││
    │     │ ─────────────────────────────────────────────────────────────────────           ││
    │     │ • Find memories about same topic (similarity > 0.85)                            ││
    │     │ • Create candidate pairs                                                        ││
    │     │                                                                                 ││
    │     │ Stage 2: Stance Detection                                                       ││
    │     │ ─────────────────────────────────────────────────────────────────────           ││
    │     │ • Classify stance: AGREE | DISAGREE | NEUTRAL | EXPANDS                         ││
    │     │ • Use lightweight classifier (50ms)                                             ││
    │     │ • Fallback to LLM if confidence < 0.7                                           ││
    │     │                                                                                 ││
    │     │ Stage 3: Temporal Context                                                       ││
    │     │ ─────────────────────────────────────────────────────────────────────           ││
    │     │ • Check time gap between memories                                               ││
    │     │ • Weight recent changes higher                                                  ││
    │     │ • Distinguish: momentary vs sustained change                                    ││
    │     │                                                                                 ││
    │     │ Stage 4: Classification                                                         ││
    │     │ ─────────────────────────────────────────────────────────────────────           ││
    │     │ • CONTRADICTION: Direct disagreement                                            ││
    │     │ • REFINEMENT: Nuanced evolution                                                 ││
    │     │ • EXPANSION: New information added                                              ││
    │     │ • ABANDONMENT: Old belief no longer held                                        ││
    │     │                                                                                 ││
    │     │ Stage 5: Storage                                                                ││
    │     │ ─────────────────────────────────────────────────────────────────────           ││
    │     │ Store as BeliefDelta:                                                           ││
    │     │ • topic: "work_satisfaction"                                                    ││
    │     │ • old_belief_id: mem_123                                                        ││
    │     │ • new_belief_id: mem_456                                                        ││
    │     │ • change_type: contradiction                                                    ││
    │     │ • confidence: 0.87                                                              ││
    │     │ • cause_event_ids: [mem_234, mem_345]                                           ││
    │     └─────────────────────────────────────────────────────────────────────────────────┘│
    │                                                                                         │
    │  3. UPDATE CAUSAL LINKS                                                                 │
    │     ┌─────────────────────────────────────────────────────────────────────────────────┐│
    │     │ • If explicit causality detected: add high-confidence link                      ││
    │     │ • If inferred causality: add low-confidence link, request confirmation          ││
    │     │ • Update knowledge graph with new relations                                     ││
    │     └─────────────────────────────────────────────────────────────────────────────────┘│
    │                                                                                         │
    │  4. TRIGGER CONSOLIDATION (if needed)                                                   │
    │     ┌─────────────────────────────────────────────────────────────────────────────────┐│
    │     │ • Check if memories > 7 days old need summarization                             ││
    │     │ • Run background consolidation job                                              ││
    │     └─────────────────────────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════════════════════════════════════╗
║  LAYER 9: RESPONSE DELIVERY & VISUALIZATION                                                      ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    RESPONSE PACKAGE                                                     │
    │                                                                                         │
    │  Final response includes:                                                               │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  1. ANSWER                                                                        │ │
    │  │     • Main response text                                                          │ │
    │  │     • Formatted with markdown                                                     │ │
    │  │     • Citations to source memories                                                │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  2. THINKING TRACE                                                                │ │
    │  │     • Step-by-step reasoning from <think> tags                                    │ │
    │  │     • Displayed in expandable UI panel                                            │ │
    │  │     • Shows how answer was derived                                                │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  3. EVIDENCE CARDS                                                                │ │
    │  │     For each source memory:                                                       │ │
    │  │     • Timestamp                                                                   │ │
    │  │     • Excerpt (highlighted relevant part)                                         │ │
    │  │     • Relevance score                                                             │ │
    │  │     • Memory type (episodic/semantic/reflective)                                  │ │
    │  │     • Clickable to view full memory                                               │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  4. METADATA                                                                      │ │
    │  │     • Agent(s) used                                                               │ │
    │  │     • Retrieval strategy                                                          │ │
    │  │     • Number of memories searched                                                 │ │
    │  │     • Confidence score                                                            │ │
    │  │     • Processing time                                                             │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                                                         │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  5. SUGGESTED FOLLOW-UPS                                                          │ │
    │  │     • Related questions to explore                                                │ │
    │  │     • Detected knowledge gaps                                                     │ │
    │  │     • Belief contradictions to resolve                                            │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Multi-Level Indexing

### 4.1 Implementation: RAPTOR Tree Indexing

**File: `src/storage/raptor_index.py`**

```python
"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
Based on: https://arxiv.org/abs/2401.18059 (ICLR 2024)

Key Concepts:
- Hierarchical clustering of memories
- Bottom-up summarization
- Multi-level retrieval (leaf → root)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.models.embeddings import EmbeddingModel
from src.llm.local_llm import LocalLLM
import networkx as nx

@dataclass
class RaptorNode:
    """A node in the RAPTOR tree."""
    node_id: str
    level: int  # 0 = leaf (raw memory), 1+ = summaries
    content: str
    embedding: np.ndarray
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    memory_ids: List[str] = field(default_factory=list)  # Source memories
    metadata: Dict = field(default_factory=dict)

class RaptorIndex:
    """
    Hierarchical tree index for multi-level retrieval.
    
    Architecture:
    Level 0 (Leaves): Individual memories
    Level 1: Clustered summaries (5-10 memories each)
    Level 2: Meta-summaries (summaries of summaries)
    Level N (Root): Top-level abstract
    
    Retrieval Strategy:
    - Specific queries: Search leaves (Level 0)
    - General queries: Search intermediate levels
    - Abstract queries: Search near root
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        max_cluster_size: int = 10,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        
        self.nodes: Dict[str, RaptorNode] = {}
        self.tree = nx.DiGraph()
        self.level_indices: Dict[int, List[str]] = {}  # level → node_ids
    
    def build_tree(self, memories: List[Dict]) -> str:
        """
        Build RAPTOR tree from memories.
        
        Process:
        1. Create leaf nodes from memories
        2. Cluster at each level
        3. Generate summaries
        4. Repeat until single root
        
        Returns:
            Root node ID
        """
        # Level 0: Create leaf nodes
        leaf_nodes = self._create_leaf_nodes(memories)
        self.level_indices[0] = [n.node_id for n in leaf_nodes]
        
        current_level = 0
        current_nodes = leaf_nodes
        
        # Build tree bottom-up
        while len(current_nodes) > 1:
            current_level += 1
            parent_nodes = self._cluster_and_summarize(current_nodes, current_level)
            self.level_indices[current_level] = [n.node_id for n in parent_nodes]
            current_nodes = parent_nodes
        
        # Return root node ID
        return current_nodes[0].node_id if current_nodes else None
    
    def _create_leaf_nodes(self, memories: List[Dict]) -> List[RaptorNode]:
        """Create leaf nodes from raw memories."""
        leaf_nodes = []
        
        for mem in memories:
            embedding = self.embedding_model.encode(mem['content'], is_query=False)
            
            node = RaptorNode(
                node_id=f"leaf_{mem['event_id']}",
                level=0,
                content=mem['content'],
                embedding=embedding,
                memory_ids=[mem['event_id']],
                metadata={
                    'timestamp': mem['timestamp'],
                    'type': mem['type'],
                    'importance': mem.get('importance', 0.5)
                }
            )
            
            self.nodes[node.node_id] = node
            self.tree.add_node(node.node_id, **node.__dict__)
            leaf_nodes.append(node)
        
        return leaf_nodes
    
    def _cluster_and_summarize(
        self, 
        nodes: List[RaptorNode], 
        level: int
    ) -> List[RaptorNode]:
        """
        Cluster nodes and generate summary nodes.
        
        Uses agglomerative clustering with cosine distance.
        """
        if len(nodes) <= self.min_cluster_size:
            # Create single summary for remaining nodes
            return [self._create_summary_node(nodes, level)]
        
        # Extract embeddings
        embeddings = np.array([n.embedding for n in nodes])
        
        # Agglomerative clustering
        n_clusters = max(2, len(nodes) // self.max_cluster_size)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = {}
        for node, label in zip(nodes, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        
        # Create summary node for each cluster
        summary_nodes = []
        for cluster_id, cluster_nodes in clusters.items():
            summary_node = self._create_summary_node(cluster_nodes, level)
            summary_nodes.append(summary_node)
        
        return summary_nodes
    
    def _create_summary_node(
        self, 
        child_nodes: List[RaptorNode], 
        level: int
    ) -> RaptorNode:
        """Generate summary from child nodes using LLM."""
        # Prepare content for summarization
        contents = [n.content for n in child_nodes]
        combined_text = "\n\n".join(contents)
        
        # Generate summary with token limit based on level
        max_tokens = 500 - (level * 50)  # Shorter summaries at higher levels
        
        prompt = f"""Summarize the following related memories into a concise overview. 
Focus on common themes, key insights, and important facts.
Maximum length: {max_tokens} tokens.

Memories:
{combined_text}

Summary:"""
        
        summary = self.llm.generate(prompt, max_tokens=max_tokens)
        
        # Embed summary
        embedding = self.embedding_model.encode(summary, is_query=False)
        
        # Collect all source memory IDs
        all_memory_ids = []
        for child in child_nodes:
            all_memory_ids.extend(child.memory_ids)
        
        # Create summary node
        node_id = f"L{level}_sum_{len(self.nodes)}"
        node = RaptorNode(
            node_id=node_id,
            level=level,
            content=summary,
            embedding=embedding,
            children=[c.node_id for c in child_nodes],
            memory_ids=all_memory_ids,
            metadata={
                'num_children': len(child_nodes),
                'created_from_level': level - 1
            }
        )
        
        # Add to tree
        self.nodes[node_id] = node
        self.tree.add_node(node_id, **node.__dict__)
        
        # Link children
        for child in child_nodes:
            child.parent = node_id
            self.tree.add_edge(node_id, child.node_id)
        
        return node
    
    def search(
        self, 
        query: str, 
        abstraction_level: str = "specific",
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search RAPTOR tree at appropriate level.
        
        Args:
            query: Search query
            abstraction_level: "specific" | "summary" | "pattern"
            k: Number of results
        
        Returns:
            List of (node_id, similarity_score)
        """
        # Map abstraction level to tree level
        level_map = {
            "specific": 0,       # Leaf nodes (raw memories)
            "summary": 1,        # First-level summaries
            "pattern": max(2, max(self.level_indices.keys()) - 1)  # Near-root
        }
        
        target_level = level_map.get(abstraction_level, 0)
        
        # Get nodes at target level
        if target_level not in self.level_indices:
            target_level = 0  # Fallback to leaves
        
        candidate_node_ids = self.level_indices[target_level]
        candidate_nodes = [self.nodes[nid] for nid in candidate_node_ids]
        
        # Compute similarities
        query_embedding = self.embedding_model.encode(query, is_query=True)
        
        results = []
        for node in candidate_nodes:
            similarity = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            results.append((node.node_id, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def get_leaf_memories(self, node_id: str) -> List[str]:
        """Get all leaf memory IDs under a given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return node.memory_ids
    
    def traverse_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root."""
        path = [node_id]
        current = self.nodes.get(node_id)
        
        while current and current.parent:
            path.append(current.parent)
            current = self.nodes.get(current.parent)
        
        return path
```

### 4.2 Implementation: Proposition-Based Indexing

**File: `src/storage/proposition_index.py`**

```python
"""
Proposition-Based Retrieval
Based on: https://arxiv.org/pdf/2312.06648 (EMNLP 2024)

Retrieves at the granularity of atomic facts rather than passages.
Higher precision for factual queries.
"""

from typing import List, Dict, Tuple
import re
from src.llm.local_llm import LocalLLM
from src.models.embeddings import EmbeddingModel
from src.storage.vector_store import VectorStore

class PropositionExtractor:
    """Extract atomic propositions from memory text."""
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
    
    def extract_propositions(self, text: str, memory_id: str) -> List[Dict]:
        """
        Break text into atomic, self-contained propositions.
        
        Example:
        Text: "I met with John yesterday to discuss Project X. 
               He suggested using React for the frontend."
        
        Propositions:
        1. "I met with John yesterday"
        2. "The meeting was about Project X"
        3. "John suggested using React"
        4. "React would be used for the frontend"
        """
        prompt = f"""Break the following text into atomic propositions. 
Each proposition should be:
- A single, self-contained fact
- Fully contextual (can be understood alone)
- Simple (subject-predicate-object)

Text: {text}

List of propositions (one per line):"""
        
        response = self.llm.generate(prompt, max_tokens=300)
        
        # Parse propositions
        props = []
        for line in response.strip().split('\n'):
            prop = line.strip().lstrip('- ').lstrip('• ').lstrip('1234567890. ')
            if prop:
                props.append({
                    'text': prop,
                    'source_memory_id': memory_id,
                    'source_text': text
                })
        
        return props

class PropositionIndex:
    """Index and retrieve at proposition granularity."""
    
    def __init__(
        self,
        extractor: PropositionExtractor,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore
    ):
        self.extractor = extractor
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.propositions: Dict[str, Dict] = {}  # prop_id → proposition
    
    def index_memory(self, memory: Dict) -> List[str]:
        """Extract and index propositions from a memory."""
        props = self.extractor.extract_propositions(
            memory['content'], 
            memory['event_id']
        )
        
        prop_ids = []
        for i, prop in enumerate(props):
            prop_id = f"{memory['event_id']}_prop_{i}"
            
            # Embed proposition
            embedding = self.embedding_model.encode(prop['text'], is_query=False)
            
            # Store in vector database
            self.vector_store.add(prop_id, embedding)
            
            # Store full proposition data
            self.propositions[prop_id] = {
                **prop,
                'prop_id': prop_id,
                'timestamp': memory['timestamp'],
                'memory_type': memory['type']
            }
            
            prop_ids.append(prop_id)
        
        return prop_ids
    
    def search(self, query: str, k: int = 20) -> List[Dict]:
        """Search propositions by query."""
        query_embedding = self.embedding_model.encode(query, is_query=True)
        
        # Vector search
        results = self.vector_store.search(query_embedding, k=k)
        
        # Retrieve full proposition data
        propositions = []
        for prop_id, score in results:
            if prop_id in self.propositions:
                prop = self.propositions[prop_id].copy()
                prop['score'] = score
                propositions.append(prop)
        
        return propositions
```

---

## 5. Phase 2: Query Intelligence Layer

### 5.1 Implementation: Intent Detection & Complexity Scoring

**File: `src/agents/query_analyzer.py`**

```python
"""
Query Intelligence: Intent Detection + Complexity Scoring
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from setfit import SetFitModel
import re

@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    intent: str  # TEMPORAL, CAUSAL, REFLECTIVE, FACTUAL, MULTI_STEP
    complexity: float  # 0.0 - 1.0
    entities: list
    temporal_constraints: dict
    confidence: float

class QueryAnalyzer:
    """Analyze queries to determine intent and complexity."""
    
    # Intent patterns (lightweight heuristics)
    INTENT_PATTERNS = {
        'TEMPORAL': [
            r'\b(when|what time|what date|how long|since when)\b',
            r'\b(yesterday|today|tomorrow|last week|next month)\b',
            r'\b(in \d{4}|on [A-Z][a-z]+ \d+)\b'
        ],
        'CAUSAL': [
            r'\b(why|how come|what caused|what led to|reason for)\b',
            r'\b(because|due to|resulted in|led to)\b'
        ],
        'REFLECTIVE': [
            r'\b(how did my (thinking|opinion|view) (change|evolve))\b',
            r'\b(what patterns|what trends)\b',
            r'\b(belief|opinion|perspective) .* (change|shift|evolution)\b'
        ],
        'FACTUAL': [
            r'\b(what is|define|explain|tell me about)\b',
            r'\b(how does .* work|how to)\b'
        ]
    }
    
    def __init__(self, intent_model_path: str = None):
        self.intent_model = None
        if intent_model_path:
            self.intent_model = SetFitModel.from_pretrained(intent_model_path)
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Perform full query analysis."""
        # Detect intent
        intent, intent_confidence = self._detect_intent(query)
        
        # Score complexity
        complexity = self._score_complexity(query)
        
        # Extract entities (simple regex-based)
        entities = self._extract_entities(query)
        
        # Extract temporal constraints
        temporal = self._extract_temporal(query)
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            entities=entities,
            temporal_constraints=temporal,
            confidence=intent_confidence
        )
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect query intent using patterns or model."""
        query_lower = query.lower()
        
        # Try pattern matching first (fast)
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent, 0.8  # High confidence for pattern match
        
        # Fallback to ML model if available
        if self.intent_model:
            prediction = self.intent_model.predict([query])[0]
            confidence = self.intent_model.predict_proba([query])[0].max()
            return prediction, confidence
        
        # Default to FACTUAL
        return "FACTUAL", 0.5
    
    def _score_complexity(self, query: str) -> float:
        """
        Score query complexity (0.0-1.0).
        
        Factors:
        - Length (longer = more complex)
        - Number of entities
        - Logical operators (AND, OR, IF)
        - Multi-part questions (semicolons, "and")
        - Temporal span
        """
        score = 0.0
        
        # Length factor (normalize to 0-0.3)
        word_count = len(query.split())
        score += min(0.3, word_count / 100)
        
        # Entity count (approximate)
        entities = self._extract_entities(query)
        score += min(0.2, len(entities) * 0.05)
        
        # Logical operators
        logical_ops = len(re.findall(r'\b(and|or|if|then|but|however)\b', query.lower()))
        score += min(0.2, logical_ops * 0.1)
        
        # Multi-part questions
        if ';' in query or query.count('?') > 1:
            score += 0.15
        
        # Temporal span
        if re.search(r'(from .* to|between .* and)', query.lower()):
            score += 0.15
        
        return min(1.0, score)
    
    def _extract_entities(self, query: str) -> list:
        """Simple entity extraction (capitalized words, proper nouns)."""
        # This is a placeholder - in production, use spaCy or NER model
        entities = []
        words = query.split()
        
        for word in words:
            # Check if capitalized (not at start of sentence)
            if word[0].isupper() and word not in ['I', 'A', 'The', 'When', 'What', 'How', 'Why']:
                entities.append(word)
        
        return list(set(entities))
    
    def _extract_temporal(self, query: str) -> dict:
        """Extract temporal constraints from query."""
        temporal = {}
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if years:
            temporal['years'] = years
        
        # Extract relative time
        relative_patterns = {
            'yesterday': -1,
            'last week': -7,
            'last month': -30,
            'last year': -365
        }
        
        for pattern, days_ago in relative_patterns.items():
            if pattern in query.lower():
                temporal['relative_days'] = days_ago
                break
        
        # Extract month names
        months = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', query, re.IGNORECASE)
        if months:
            temporal['months'] = months
        
        return temporal
```

### 5.2 Implementation: Multi-Query Generation (RAG-Fusion)

**File: `src/retrieval/query_transformer.py`**

```python
"""
Query Transformation: Multi-Query + HyDE + Decomposition
Based on: RAG-Fusion (arXiv:2402.03367)
"""

from typing import List, Dict
from src.llm.local_llm import LocalLLM

class QueryTransformer:
    """Transform queries for improved retrieval coverage."""
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
    
    def generate_multi_query(self, query: str, n: int = 4) -> List[str]:
        """
        Generate multiple variations of the query.
        
        Increases retrieval coverage by rephrasing.
        """
        prompt = f"""Generate {n} different versions of the following query. 
Each version should:
- Preserve the original meaning
- Use different wording and phrasing
- Focus on different aspects if applicable

Original query: {query}

Variant queries:"""
        
        response = self.llm.generate(prompt, max_tokens=200)
        
        # Parse variants
        variants = [query]  # Include original
        for line in response.strip().split('\n'):
            variant = line.strip().lstrip('- ').lstrip('• ').lstrip('1234567890. ')
            if variant and variant != query:
                variants.append(variant)
        
        return variants[:n+1]
    
    def generate_hyde(self, query: str) -> str:
        """
        Generate Hypothetical Document Embedding (HyDE).
        
        Creates a hypothetical answer, then retrieves documents
        similar to this hypothetical answer.
        
        Based on: https://arxiv.org/abs/2212.10496 (ACL 2023)
        """
        prompt = f"""Imagine you have perfect knowledge about the user's memories.
Write a detailed, specific answer to the following question based on what those memories might contain.

Question: {query}

Hypothetical answer (2-3 sentences):"""
        
        hypothetical_answer = self.llm.generate(prompt, max_tokens=150)
        
        return hypothetical_answer.strip()
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-questions.
        
        Useful for multi-hop reasoning queries.
        """
        prompt = f"""Break down the following complex question into simpler sub-questions.
Each sub-question should:
- Be answerable independently
- Together, lead to answering the original question
- Be ordered logically

Complex question: {query}

Sub-questions:"""
        
        response = self.llm.generate(prompt, max_tokens=250)
        
        # Parse sub-questions
        sub_questions = []
        for line in response.strip().split('\n'):
            sub_q = line.strip().lstrip('- ').lstrip('• ').lstrip('1234567890. ')
            if sub_q and '?' in sub_q:
                sub_questions.append(sub_q)
        
        return sub_questions if sub_questions else [query]
```

---

## 6. Phase 3: Agentic Retrieval Engine

### 6.1 Implementation: Agent Orchestrator

**File: `src/agents/orchestrator.py`**

```python
"""
Agent Orchestrator: Routes queries to specialized agents
Implements Adaptive-RAG routing logic
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.agents.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.agents.timeline_agent import TimelineAgent
from src.agents.causal_agent import CausalAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.arbitration_agent import ArbitrationAgent
from src.llm.local_llm import LocalLLM
from src.retrieval.hybrid_retriever import HybridRetriever

class RoutingStrategy(Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"

@dataclass
class OrchestratorResponse:
    """Response from orchestrator."""
    answer: str
    thinking_trace: List[str]
    evidence: List[Dict]
    confidence: float
    agents_used: List[str]
    retrieval_strategy: str
    processing_time_ms: float

class AgentOrchestrator:
    """
    Routes queries to appropriate agents based on intent and complexity.
    
    Routing Logic (Adaptive-RAG inspired):
    - Simple (complexity < 0.3): Direct LLM, no retrieval
    - Moderate (0.3-0.7): Single-step RAG with specialized agent
    - Complex (>0.7): Multi-step RAG with agent coordination
    """
    
    def __init__(
        self,
        llm: LocalLLM,
        retriever: HybridRetriever,
        query_analyzer: QueryAnalyzer
    ):
        self.llm = llm
        self.retriever = retriever
        self.query_analyzer = query_analyzer
        
        # Initialize specialized agents
        self.timeline_agent = TimelineAgent(llm, retriever)
        self.causal_agent = CausalAgent(llm, retriever)
        self.reflection_agent = ReflectionAgent(llm, retriever)
        self.planning_agent = PlanningAgent(llm, retriever, self)
        self.arbitration_agent = ArbitrationAgent(llm, retriever)
    
    def process(self, query: str, context: Dict = None) -> OrchestratorResponse:
        """Main entry point for query processing."""
        import time
        start_time = time.time()
        
        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(query)
        
        # Step 2: Determine routing strategy
        strategy = self._determine_strategy(analysis)
        
        # Step 3: Route to appropriate handler
        if strategy == RoutingStrategy.NO_RETRIEVAL:
            result = self._handle_no_retrieval(query, analysis)
        elif strategy == RoutingStrategy.SINGLE_STEP:
            result = self._handle_single_step(query, analysis, context)
        else:  # MULTI_STEP
            result = self._handle_multi_step(query, analysis, context)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return OrchestratorResponse(
            answer=result['answer'],
            thinking_trace=result['thinking_trace'],
            evidence=result['evidence'],
            confidence=result['confidence'],
            agents_used=result['agents_used'],
            retrieval_strategy=strategy.value,
            processing_time_ms=processing_time
        )
    
    def _determine_strategy(self, analysis: QueryAnalysis) -> RoutingStrategy:
        """Determine routing strategy based on query analysis."""
        if analysis.complexity < 0.3:
            return RoutingStrategy.NO_RETRIEVAL
        elif analysis.complexity < 0.7:
            return RoutingStrategy.SINGLE_STEP
        else:
            return RoutingStrategy.MULTI_STEP
    
    def _handle_no_retrieval(self, query: str, analysis: QueryAnalysis) -> Dict:
        """Handle simple queries without retrieval."""
        prompt = f"""You are Cortex Lab. Answer the following simple question directly.

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=200)
        
        return {
            'answer': answer,
            'thinking_trace': ["Simple query, answered without retrieval"],
            'evidence': [],
            'confidence': 0.7,
            'agents_used': ["Orchestrator (Direct)"]
        }
    
    def _handle_single_step(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        context: Optional[Dict]
    ) -> Dict:
        """Handle moderate queries with single-step RAG."""
        # Select agent based on intent
        agent = self._select_agent(analysis.intent)
        
        # Execute agent
        response = agent.process(query, context or {})
        
        return {
            'answer': response.answer,
            'thinking_trace': response.reasoning_trace,
            'evidence': response.evidence,
            'confidence': response.confidence,
            'agents_used': [agent.name]
        }
    
    def _handle_multi_step(
        self, 
        query: str, 
        analysis: QueryAnalysis,
        context: Optional[Dict]
    ) -> Dict:
        """Handle complex queries with multi-step RAG."""
        # Use planning agent for orchestration
        response = self.planning_agent.process(query, context or {})
        
        return {
            'answer': response.answer,
            'thinking_trace': response.reasoning_trace,
            'evidence': response.evidence,
            'confidence': response.confidence,
            'agents_used': response.follow_up_queries  # Contains agent trace
        }
    
    def _select_agent(self, intent: str):
        """Select appropriate agent based on intent."""
        agent_map = {
            'TEMPORAL': self.timeline_agent,
            'CAUSAL': self.causal_agent,
            'REFLECTIVE': self.reflection_agent,
            'FACTUAL': self.timeline_agent,  # Default to timeline
            'MULTI_STEP': self.planning_agent
        }
        
        return agent_map.get(intent, self.timeline_agent)
```

### 6.2 Implementation: Specialized Agents

**Timeline Agent Example:**

**File: `src/agents/timeline_agent.py`**

```python
"""
Timeline Agent: Handles temporal queries
"""

from src.agents.base_agent import BaseAgent, AgentResponse
from typing import Dict, Any, List
from datetime import datetime, timedelta
import re

class TimelineAgent(BaseAgent):
    """Agent for timeline-based queries."""
    
    def __init__(self, llm, retriever):
        super().__init__(llm, retriever, name="TimelineAgent")
        
        self.intent_prompt = """Analyze this temporal query and extract:
1. Time range (start date, end date)
2. Key entities mentioned
3. Event types to focus on

Query: {query}

Analysis (JSON format):"""
    
    def process(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        self.reasoning_trace = []
        
        # Step 1: Extract temporal information
        self._log_reasoning("Extracting temporal constraints...")
        temporal_info = self._extract_temporal_info(query)
        
        # Step 2: Retrieve memories in time range
        self._log_reasoning(f"Retrieving memories from {temporal_info.get('start_date')} to {temporal_info.get('end_date')}")
        
        results = self._retrieve(
            query,
            channels=['dense', 'sql'],
            filters={
                'start_date': temporal_info.get('start_date'),
                'end_date': temporal_info.get('end_date'),
                'memory_types': temporal_info.get('memory_types', [])
            },
            k=15
        )
        
        # Step 3: Build chronological narrative
        self._log_reasoning("Building chronological narrative...")
        narrative = self._build_timeline_narrative(query, results, temporal_info)
        
        return AgentResponse(
            answer=narrative,
            evidence=results,
            reasoning_trace=self.reasoning_trace,
            confidence=0.85,
            follow_up_queries=self._suggest_follow_ups(temporal_info)
        )
    
    def _extract_temporal_info(self, query: str) -> Dict:
        """Extract temporal information from query."""
        prompt = self.intent_prompt.format(query=query)
        response = self.llm.generate(prompt, max_tokens=150)
        
        # Parse temporal info (simplified - use actual JSON parsing)
        temporal_info = {
            'start_date': None,
            'end_date': None,
            'entities': [],
            'memory_types': []
        }
        
        # Extract year mentions
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if years:
            temporal_info['start_date'] = f"{years[0]}-01-01"
            temporal_info['end_date'] = f"{years[-1]}-12-31"
        
        # Extract month names
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        for month_name, month_num in month_map.items():
            if month_name.lower() in query.lower():
                # Assume current year if not specified
                year = years[0] if years else datetime.now().year
                temporal_info['start_date'] = f"{year}-{month_num:02d}-01"
                # Last day of month
                if month_num == 12:
                    temporal_info['end_date'] = f"{year}-12-31"
                else:
                    temporal_info['end_date'] = f"{year}-{month_num+1:02d}-01"
        
        return temporal_info
    
    def _build_timeline_narrative(
        self, 
        query: str, 
        results: List, 
        temporal_info: Dict
    ) -> str:
        """Build chronological narrative from results."""
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda x: x.metadata.get('timestamp', ''))
        
        # Format evidence
        evidence_text = "\n\n".join([
            f"[{r.metadata['timestamp']}] {r.content}" 
            for r in sorted_results[:10]
        ])
        
        narrative_prompt = f"""Based on the following chronological memories, answer the query.
Focus on the timeline, sequence of events, and temporal patterns.

Query: {query}

Memories (chronological):
{evidence_text}

Answer (narrative format):"""
        
        narrative = self.llm.generate(narrative_prompt, max_tokens=400)
        
        return narrative
    
    def _suggest_follow_ups(self, temporal_info: Dict) -> List[str]:
        """Suggest follow-up questions."""
        return [
            "What happened before this period?",
            "What happened after this period?",
            "What patterns do you see in this timeline?"
        ]
```

---

*[Due to length constraints, the remaining sections (7-12) follow similar detailed implementation patterns with code examples, architecture diagrams, and production deployment guides. The complete document would be approximately 15,000 lines covering all aspects of the Agentic RAG system.]*

---

## 10. Implementation Roadmap

### Phase-by-Phase Implementation (18 Weeks)

**Week 1-2: Foundation**
- [ ] Project structure setup
- [ ] Core data models (CausalMemoryObject, MemoryQuery)
- [ ] Lightweight classifiers (SetFit training)
- [ ] Entity resolver implementation

**Week 3-4: Storage Layer**
- [ ] RAPTOR tree indexing
- [ ] Proposition-based indexing
- [ ] Vector store (FAISS)
- [ ] DuckDB integration
- [ ] Knowledge graph (NetworkX)

**Week 5-6: Query Intelligence**
- [ ] Query analyzer (intent + complexity)
- [ ] Query transformer (multi-query + HyDE)
- [ ] Test intent classification accuracy

**Week 7-9: Agentic Layer**
- [ ] Base agent framework
- [ ] Specialized agents (Timeline, Causal, Reflection)
- [ ] Agent orchestrator
- [ ] Multi-agent coordination

**Week 10-11: Hybrid Retrieval**
- [ ] Dense retriever (BGE)
- [ ] Sparse retriever (BM25 + SPLADE)
- [ ] Graph traversal
- [ ] RRF fusion + reranking

**Week 12-13: Self-Reflective Generation**
- [ ] CRAG quality evaluation
- [ ] Self-RAG critique loop
- [ ] FLARE active retrieval
- [ ] DeepSeek-R1 fine-tuning

**Week 14-15: Memory Evolution**
- [ ] Belief evolution tracker
- [ ] Causal link detector
- [ ] Memory consolidation pipeline

**Week 16-17: Evaluation & Testing**
- [ ] Synthetic dataset generation
- [ ] RAGAS evaluation
- [ ] Performance benchmarking
- [ ] Integration testing

**Week 18: Production Polish**
- [ ] Web UI refinement
- [ ] Documentation
- [ ] Deployment guide
- [ ] Final optimizations

---

## 11. Evaluation & Benchmarking

### Evaluation Framework

**File: `src/evaluation/rag_evaluator.py`**

```python
"""
Comprehensive RAG evaluation using RAGAS framework
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def evaluate_system(self, test_dataset: List[Dict]) -> Dict:
        """
        Evaluate on test dataset.
        
        test_dataset format:
        [
            {
                'query': "Why did I quit my job?",
                'ground_truth_answer': "Due to burnout...",
                'ground_truth_contexts': ["memory_1", "memory_2"]
            },
            ...
        ]
        """
        results = evaluate(
            dataset=test_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ]
        )
        
        return {
            'faithfulness': results['faithfulness'],
            'answer_relevancy': results['answer_relevancy'],
            'context_recall': results['context_recall'],
            'context_precision': results['context_precision'],
            'overall_score': (
                results['faithfulness'] + 
                results['answer_relevancy'] + 
                results['context_recall'] + 
                results['context_precision']
            ) / 4
        }
```

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Faithfulness** | > 0.85 | Answer grounded in retrieved context |
| **Answer Relevancy** | > 0.80 | Answer addresses the query |
| **Context Recall** | > 0.75 | Retrieved all relevant memories |
| **Context Precision** | > 0.70 | Retrieved memories are relevant |
| **E2E Latency (Simple)** | < 2s | Single-agent queries |
| **E2E Latency (Complex)** | < 5s | Multi-agent queries |

---

## 12. Production Deployment

### System Requirements

**Minimum Hardware:**
- GPU: NVIDIA GTX 1650 (4GB VRAM)
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8GB
- Storage: 20GB SSD

**Software Stack:**
- Python 3.10+
- CUDA 11.8+
- Node.js 18+ (for frontend)

### Deployment Checklist

- [ ] Models downloaded and quantized
- [ ] Virtual environment configured
- [ ] Database initialized (DuckDB)
- [ ] Vector store indexed (FAISS)
- [ ] Ollama/llama.cpp installed
- [ ] Environment variables set
- [ ] Health checks passing
- [ ] Backup strategy configured
- [ ] Monitoring enabled
- [ ] Documentation updated

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Suraj-creation/Cortex-Lab
cd Cortex-Lab

# 2. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Download models
python scripts/download_models.py

# 4. Initialize database
python scripts/init_db.py

# 5. Start backend
cd backend
python server.py

# 6. Start frontend (new terminal)
cd frontend
npm install
npm run dev

# 7. Access at http://localhost:3000
```

---

## Summary

This RAG architecture represents the state-of-the-art in personal AI memory systems, combining:

✅ **Hierarchical Indexing** (RAPTOR + Propositions)  
✅ **Agentic Reasoning** (Specialized agents with orchestration)  
✅ **Hybrid Retrieval** (Dense + Sparse + Graph + Temporal)  
✅ **Self-Reflection** (CRAG + Self-RAG + FLARE)  
✅ **Memory Evolution** (Belief tracking + Consolidation)  
✅ **Production-Ready** (Optimized for 4GB VRAM)

**Cortex Lab is not just a chatbot—it's your second brain, powered by cutting-edge research and optimized for consumer hardware.** 🧠🚀
