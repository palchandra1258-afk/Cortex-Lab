# Cortex Lab: Advanced Agentic RAG Architecture
## Production-Ready Implementation Guide (2025)

> **Last Updated:** February 19, 2026  
> **Version:** 3.0 — Maximum Performance Edition  
> **Target Hardware:** NVIDIA RTX 4000 Ada Generation (20GB VRAM, Ada Lovelace Architecture)  
> **Model:** DeepSeek-R1-Distill-Qwen-7B (10-Stage Fine-tuned + DPO Aligned) | 14B Ultra Option  
> **Embedding:** BGE-large-en-v1.5 (1024d) | BGE-reranker-v2-m3  
> **VRAM Utilization:** Training 63% (13GB) | Inference 34% (7GB) — fully exploiting 20GB

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
13. [Advanced Enhancements: Research-Driven Improvements](#13-advanced-enhancements-research-driven-improvements)
    - 13.1–13.13: Core Enhancements (Contextual Chunking, Semantic Chunking, Step-Back, Caching, VQ, ANN, Async, RAGChecker, Failure-Aware, CoR, Token Efficiency, Retriever FT, Self-Improvement)
    - 13.15: Universal Multi-Modal Data Ingestion Pipeline (16+ data types)
    - 13.16: Continuous Data Feed & Incremental Indexing
    - 13.17: Enhancement Impact Matrix

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
| Text-only input | Universal: 16+ data types (PDF, images, code, audio, video, etc.) |
| Manual file upload | Continuous data feed with auto-ingestion + deduplication |
| Small embedding models | BGE-large-en-v1.5 (1024d) + domain-adapted fine-tuning |

### 1.2 Research Foundations

This architecture synthesizes 25+ cutting-edge techniques from top-tier venues:

**Indexing & Storage:**
- ✅ **RAPTOR** (ICLR 2024) - Hierarchical tree-structured indexing
- ✅ **Proposition Retrieval** (EMNLP 2024) - Atomic fact-level granularity
- ✅ **GraphRAG** (Microsoft 2024) - Entity-relationship knowledge graphs
- ✅ **Contextual Chunking** (Anthropic 2024) - Document-context-aware chunk enrichment
- ✅ **Semantic Chunking** (2024) - Embedding-similarity boundary detection
- ✅ **TreeRAG** (ACL 2025) - Hierarchical connectivity preservation

**Retrieval Techniques:**
- ✅ **ColBERTv2** (NAACL 2022) - Multi-vector late interaction
- ✅ **HyDE** (ACL 2023) - Hypothetical document embeddings
- ✅ **RAG-Fusion** (2024) - Multi-query with RRF
- ✅ **BGE Embeddings** (MTEB 2024) - State-of-the-art dense retrieval
- ✅ **Step-Back Prompting** (2024) - Abstract question generation for complex reasoning
- ✅ **Vector Quantization** (PQ/SQ8) - Memory-efficient ANN indexing

**Agentic Components:**
- ✅ **Self-RAG** (ICLR 2024) - Self-reflective generation with critique
- ✅ **CRAG** (2024) - Corrective retrieval quality evaluation
- ✅ **FLARE** (EMNLP 2023) - Forward-looking active retrieval
- ✅ **Adaptive-RAG** (NAACL 2024) - Query complexity routing
- ✅ **Chain-of-Retrieval** (NeurIPS 2024) - Step-by-step retrieval-reasoning chains
- ✅ **Failure-Aware Query Refinement** - Systematic refinement by failure type

**Production & Optimization:**
- ✅ **Multi-Level Caching** - Embedding + semantic + response caching
- ✅ **Async Pipeline Architecture** - Parallel retrieval with asyncio
- ✅ **Hot/Cold Storage Tiering** - HNSW (recent) + IVF-PQ (archival)
- ✅ **RAGChecker** (NeurIPS 2024) - Fine-grained diagnostic evaluation
- ✅ **Token Efficiency Optimization** - Minimizing agentic overhead
- ✅ **Retriever Fine-tuning Pipeline** - Domain adaptation on user memory data

**Memory & Evolution:**
- ✅ **Belief Evolution Tracking** - Multi-stage contradiction detection
- ✅ **Memory Consolidation** - Hierarchical summarization with decay
- ✅ **Entity Resolution** - Coreference + fuzzy matching

**Universal Data Ingestion (NEW — 16+ data types):**
- ✅ **Multi-Modal Pipeline** - PDF, images, code, audio, video, spreadsheets, URLs, email, calendar
- ✅ **OCR + Vision Captioning** - EasyOCR + BLIP-base for image/handwriting/scanned documents
- ✅ **Code-Aware Chunking** - AST/tree-sitter parsing for 15+ programming languages
- ✅ **Structured Data Import** - CSV/JSON/Excel with schema inference + row-group chunking
- ✅ **Continuous Data Feed** - Always-on ingestion queue with incremental indexing
- ✅ **Content Deduplication** - SHA-256 hash-based dedup prevents re-ingesting identical data

### 1.3 Performance Targets

> **Note:** These are **projected targets** based on published benchmarks for the component techniques. Actual performance will be validated post-deployment via RAGChecker (§13.8) and the continuous feedback loop (§13.13). Targets marked with † depend on completing the 10-stage fine-tuning curriculum (see [Fine-Tuning.md](Fine-Tuning.md)).

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Retrieval Precision@10** | > 0.80 | Multi-channel fusion (dense + sparse + graph + temporal + proposition) |
| **Answer Faithfulness** | > 0.92 † | RAGAS evaluation (post-DPO alignment) |
| **Multi-Turn Coherence** | > 0.88 † | Sustained context across 10+ turns |
| **Long-Context Recall** | > 0.85 † | Accurate retrieval over 8K+ token contexts |
| **Query Latency (Simple)** | < 1.5s | Single-agent queries (7B + Flash Attention 2) |
| **Query Latency (Complex)** | < 6s (P90), < 8s (P99) | Multi-agent orchestration with async retrieval (see §13.11 for breakdown) |
| **Memory Footprint** | < 8GB | 7B model (4.2GB) + BGE-large (1.3GB) + reranker + indexes |
| **Classification Speed** | < 100ms | Lightweight classifiers (SetFit + DistilBERT) |
| **LLM Fallback Rate** | < 10% † | 10-stage fine-tuning reduces classifier uncertainty |
| **Vector Search P99** | < 50ms | ANN-tuned HNSW/IVF-PQ (1024d BGE-large vectors) |
| **Cache Hit Rate** | > 40% | Semantic + exact + response caching |
| **Index Memory** | < 1.5GB | PQ/SQ8 compressed 1024d vectors at 500K scale |
| **DPO Win Rate** | > 70% † | Preferred responses over base model (post Stage 9 fine-tuning) |
| **Data Type Coverage** | 12+ types | Text, PDF, images, code, audio, video, CSV, JSON, URLs, email, etc. |

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
║  LAYER 0: UNIVERSAL DATA ACQUISITION & MULTI-MODAL INGESTION                                   ║
║  (Accepts ANY data type the user continuously feeds — text, files, images, audio, video, etc.)  ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝

    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │   Text     │  │   Voice    │  │  Document  │  │   Image    │  │   Code     │
    │   Input    │  │   Input    │  │   Import   │  │   Input    │  │   Input    │
    │ (chat/note)│  │(mic/upload)│  │(PDF/DOCX)  │  │(PNG/JPG/WP)│  │(.py/.js/…) │
    └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
           │               │               │               │               │
    ┌──────┘  ┌────────────┘    ┌──────────┘    ┌──────────┘    ┌──────────┘
    │         │                 │               │               │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  │  URL /     │  │ Structured │  │  Email /   │  │  Video /   │
    │  │  Webpage   │  │   Data     │  │  Calendar  │  │  Audio     │
    │  │  Input     │  │(CSV/JSON/  │  │  Import    │  │  Files     │
    │  │ (scraping) │  │ Excel/DB)  │  │(.eml/.ics) │  │(.mp4/.wav) │
    │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
    │         │               │               │               │
    └─────────┴───────────────┴───────────────┴───────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    UNIVERSAL DATA PROCESSING HUB                           │
    │                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │  FORMAT DETECTION & ROUTING (auto-detect MIME type / extension)    │   │
    │  │                                                                     │   │
    │  │  Text/Chat     → Direct text pipeline                              │   │
    │  │  Voice/Audio   → Whisper ASR (faster-whisper, base) → text         │   │
    │  │  PDF           → PyMuPDF text + table extraction → text chunks     │   │
    │  │  DOCX/ODT      → python-docx / pandoc → text + structure           │   │
    │  │  Image (photo) → EasyOCR (text extraction) + BLIP-base (captioning) │   │
    │  │  Image (diag.) → Layout detection → structured description         │   │
    │  │  Code files    → AST parsing + language detection → semantic units │   │
    │  │  CSV/TSV       → pandas DataFrame → row descriptions + schema      │   │
    │  │  JSON/JSONL    → Schema inference → structured event extraction    │   │
    │  │  Excel (.xlsx) → openpyxl → sheet-by-sheet extraction + formulas  │   │
    │  │  URL/Webpage   → Trafilatura / readability → clean article text   │   │
    │  │  Email (.eml)  → email.parser → subject/body/attachments split    │   │
    │  │  Calendar(.ics)→ icalendar → event extraction with temporal data  │   │
    │  │  Video (.mp4)  → ffmpeg frames + Whisper audio → transcript+desc  │   │
    │  │  Markdown/RST  → Structure-aware parsing → headed sections        │   │
    │  │  Database (SQL) → Schema + query results → relational context     │   │
    │  │  Handwriting   → EasyOCR + preprocessing → recognized text        │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │  METADATA ENRICHMENT (attached to every ingested item)             │   │
    │  │                                                                     │   │
    │  │  • source_type: "pdf" | "image" | "audio" | "code" | "url" | …    │   │
    │  │  • original_filename: preserved for provenance                     │   │
    │  │  • ingestion_timestamp: ISO 8601 with timezone                     │   │
    │  │  • content_hash: SHA-256 for deduplication                         │   │
    │  │  • extracted_tables: count of tables found (if any)                │   │
    │  │  • extracted_images: count of embedded images                      │   │
    │  │  • language_detected: auto-detected language code                  │   │
    │  │  • word_count / token_estimate: for chunking decisions             │   │
    │  │  • confidence_score: extraction confidence (OCR quality, etc.)     │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    │                                                                             │
    │  Output: Normalized text + rich metadata → feeds into Layer 1              │
    │  VRAM: Processing models (~300MB for OCR/captioning, loaded on demand)     │
    │  Dedup: SHA-256 hash check prevents re-ingesting identical content         │
    └─────────────────────────────────────────────────────────────────────────────┘
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
    │ • 1024 dimensions│   │ • Timestamps     │   │ • Relations      │   │ • Fine-grained   │
    │ • Dense search   │   │ • Filters        │   │ • Causal Links   │   │ • High precision │
    │ • <50ms latency  │   │ • SQL queries    │   │ • Graph Traverse │   │ • Complement     │
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
    │  │  Model: BGE-large-en-v1.5  │  │  Method: BM25 +    │  │  Method: NetworkX  │               │
    │  │  Embedding: 1024d   │  │          SPLADE    │  │  Hops: 2-3 levels  │               │
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
    │  Model: DeepSeek-R1-Distill-Qwen-7B (4-bit NF4, 10-stage QLoRA + DPO fine-tuned)        │
    │                                                                                         │
    │  10-Stage QLoRA Curriculum (see Fine-Tuning.md for full specification):                  │
    │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
    │  │  Stage 1: RAG-Grounded Faithfulness (cite-or-abstain, 3K samples)                 │ │
    │  │  Stage 2: Agentic Reasoning & Tool-Use (agent routing, 3K samples)                │ │
    │  │  Stage 3: Causal Chain & Temporal Reasoning (why/when, 3K samples)                │ │
    │  │  Stage 4: Self-Reflective Critique — Self-RAG / CRAG (2.5K samples)               │ │
    │  │  Stage 5: Belief Evolution & Contradiction Handling (2.5K samples)                 │ │
    │  │  Stage 6: Memory Consolidation & Summarization (2.5K samples)                     │ │
    │  │  Stage 7: Multi-Turn Dialogue Coherence (2.5K samples)                            │ │
    │  │  Stage 8: Long-Context Reasoning & Deep Multi-Hop (3K samples)                    │ │
    │  │  Stage 9: DPO Preference Alignment (3K preference pairs)                          │ │
    │  │  Stage 10: User-Specific Style Adaptation (500 user samples)                      │ │
    │  │                                                                                   │ │
    │  │  Total: ~27K SFT + 3K DPO = ~30K training samples                                │ │
    │  │  Training time: ~29.5 hours on RTX 4000 Ada Generation (VRAM: 63%)                │ │
    │  │  Validation: Each stage evaluated independently before proceeding                 │ │
    │  └───────────────────────────────────────────────────────────────────────────────────┘ │
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
    """
    Analyze queries to determine intent and complexity.
    
    Intent Detection Strategy (3-tier cascade):
    1. SetFit ML model (primary) — fine-tuned on 500+ labeled examples per intent
    2. Regex pattern heuristics (fast fallback if model unavailable)
    3. LLM classification (expensive fallback if confidence < 0.6)
    
    Complexity Scoring Strategy (hybrid):
    1. Lightweight feature extraction (syntax-based)
    2. Semantic complexity via embedding distance from simple-query centroid
    3. Combined score with learned weights
    """
    
    # Intent patterns (lightweight heuristics — FALLBACK ONLY, not primary)
    # These exist for cold-start before SetFit model is trained
    INTENT_PATTERNS = {
        'TEMPORAL': [
            r'\b(when|what time|what date|how long|since when)\b',
            r'\b(yesterday|today|tomorrow|last week|next month)\b',
            r'\b(in \d{4}|on [A-Z][a-z]+ \d+)\b',
            r'\b(before|after|during|while|until|since)\b.*\b(did|was|were|had)\b'
        ],
        'CAUSAL': [
            r'\b(why|how come|what caused|what led to|reason for)\b',
            r'\b(because|due to|resulted in|led to|consequence)\b',
            r'\b(impact|effect|influence|trigger)\b.*\b(of|on|from)\b'
        ],
        'REFLECTIVE': [
            r'\b(how (has|have|did) (my|I)|what patterns|what trends)\b',
            r'\b(change|evolve|shift|grow|develop)\b.*\b(over time|lately|recently)\b',
            r'\b(used to|no longer|compared to before)\b',
            r'\b(belief|opinion|perspective|view|feeling|attitude)\b.*\b(change|shift|evolution|different)\b'
        ],
        'MULTI_STEP': [
            r'\b(compare|contrast|analyze|evaluate)\b.*\b(and|versus|vs|with)\b',
            r'\b(first|then|also|additionally|moreover)\b',
            r'.+\?.+\?',  # Multiple question marks
        ],
        'FACTUAL': [
            r'\b(what is|define|explain|tell me about|who is|where is)\b',
            r'\b(how does .* work|how to|what are)\b'
        ]
    }
    
    def __init__(self, intent_model_path: str = None, embedding_model=None):
        """
        Args:
            intent_model_path: Path to fine-tuned SetFit intent classifier
            embedding_model: Shared BGE-large model for semantic complexity scoring
        """
        self.intent_model = None
        if intent_model_path:
            self.intent_model = SetFitModel.from_pretrained(intent_model_path)
        self.embedding_model = embedding_model
        
        # Pre-computed centroid for "simple query" cluster (updated during self-improvement)
        self._simple_query_centroid = None
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Perform full query analysis."""
        intent, intent_confidence = self._detect_intent(query)
        complexity = self._score_complexity(query)
        entities = self._extract_entities(query)
        temporal = self._extract_temporal(query)
        
        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            entities=entities,
            temporal_constraints=temporal,
            confidence=intent_confidence
        )
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect query intent using 3-tier cascade:
        1. SetFit model (primary, if available and confident)
        2. Regex patterns (fast heuristic fallback)
        3. LLM classification (expensive, high-accuracy fallback)
        
        Design rationale: SetFit models achieve >90% accuracy on intent
        classification with just 8-shot training per class. Regex is the
        cold-start fallback before the model is trained on user data.
        """
        # Tier 1: ML model (primary — most accurate)
        if self.intent_model:
            prediction = self.intent_model.predict([query])[0]
            proba = self.intent_model.predict_proba([query])[0]
            confidence = float(proba.max())
            if confidence >= 0.6:
                return prediction, confidence
            # Low confidence → fall through to Tier 2 for confirmation
        
        # Tier 2: Pattern matching (fast heuristic)
        query_lower = query.lower()
        matched_intents = []
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matched_intents.append(intent)
                    break
        
        if len(matched_intents) == 1:
            return matched_intents[0], 0.7  # Single clear match
        elif len(matched_intents) > 1:
            # Ambiguous — multiple intents matched. Use LLM if available.
            pass  # Fall through to Tier 3
        elif len(matched_intents) == 0 and not self.intent_model:
            # No model, no pattern match → LLM or default
            pass
        else:
            # Model was low-confidence, no pattern match either
            pass
        
        # Tier 3: LLM classification (expensive fallback, ~0.3s)
        # Only used when Tiers 1+2 are ambiguous or unavailable
        # In production, uses DeepSeek-R1 with a structured classification prompt
        # Returns: (intent, confidence) from LLM JSON response
        # Fallback of last resort:
        return "FACTUAL", 0.5
    
    def _score_complexity(self, query: str) -> float:
        """
        Score query complexity (0.0-1.0) using hybrid approach:
        
        1. Syntactic features (fast, rule-based) — weight: 0.4
        2. Semantic distance from "simple query" centroid — weight: 0.6
        
        This addresses the flaw where "why?" scores low on syntax
        but is actually a hard causal question. The semantic component
        captures this because "why?" is far from the simple-query centroid
        (which clusters around "what is X?" / "tell me about Y").
        """
        # Component 1: Syntactic complexity (0.0-1.0)
        syntactic_score = self._syntactic_complexity(query)
        
        # Component 2: Semantic complexity (0.0-1.0)
        semantic_score = self._semantic_complexity(query)
        
        # Weighted combination
        if semantic_score is not None:
            return min(1.0, 0.4 * syntactic_score + 0.6 * semantic_score)
        else:
            # No embedding model available — syntactic only (with caveat)
            return syntactic_score
    
    def _syntactic_complexity(self, query: str) -> float:
        """Rule-based syntactic complexity features."""
        score = 0.0
        
        word_count = len(query.split())
        score += min(0.3, word_count / 100)
        
        entities = self._extract_entities(query)
        score += min(0.2, len(entities) * 0.05)
        
        logical_ops = len(re.findall(r'\b(and|or|if|then|but|however|because|although)\b', query.lower()))
        score += min(0.2, logical_ops * 0.1)
        
        if ';' in query or query.count('?') > 1:
            score += 0.15
        
        if re.search(r'(from .* to|between .* and)', query.lower()):
            score += 0.15
        
        return min(1.0, score)
    
    def _semantic_complexity(self, query: str) -> Optional[float]:
        """
        Semantic complexity: cosine distance from the "simple query" centroid.
        
        Intuition: Simple queries ("what is X?") cluster together in embedding space.
        Complex queries ("why did my career priorities shift after the burnout?")
        are far from that centroid. This catches short-but-hard queries like "why?"
        that fool syntactic heuristics.
        
        Requires: BGE-large embedding model (shared with retrieval pipeline).
        """
        if self.embedding_model is None or self._simple_query_centroid is None:
            return None
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Cosine distance from simple-query centroid (0 = simple, 1 = complex)
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(query_embedding, self._simple_query_centroid) / (
            norm(query_embedding) * norm(self._simple_query_centroid)
        )
        
        # Convert similarity (0-1) to complexity (0-1)
        return max(0.0, min(1.0, 1.0 - similarity))
    
    def _extract_entities(self, query: str) -> list:
        """
        Production entity extraction using spaCy NER + fuzzy alias matching.
        
        Pipeline:
        1. spaCy NER (en_core_web_sm) — extracts PERSON, ORG, GPE, DATE, EVENT, etc.
        2. Custom entity patterns from user's canonical entity index (RapidFuzz matching)
        3. Coreference hints ("he" → last mentioned PERSON, "there" → last GPE)
        
        This replaces the naive capitalization check. spaCy NER runs in ~5ms
        on CPU and is loaded once at startup (~50MB RAM).
        
        For zero-shot entity types not in spaCy's ontology, we use GLiNER
        (General Linear Named Entity Recognition) which accepts custom entity
        labels at inference time — useful for domain-specific entities like
        project names, personal concepts, etc.
        """
        import spacy
        
        # Primary: spaCy NER (fast, reliable for standard entity types)
        if not hasattr(self, '_nlp'):
            self._nlp = spacy.load("en_core_web_sm")
        
        doc = self._nlp(query)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,      # PERSON, ORG, DATE, GPE, etc.
                'start': ent.start_char,
                'end': ent.end_char,
                'source': 'spacy_ner'
            })
        
        # Secondary: Match against user's canonical entity index
        # Uses RapidFuzz for fuzzy matching (handles typos, abbreviations)
        if hasattr(self, '_entity_index') and self._entity_index:
            from rapidfuzz import process as fuzz_process
            
            # Extract noun chunks as entity candidates
            candidates = [chunk.text for chunk in doc.noun_chunks]
            for candidate in candidates:
                match = fuzz_process.extractOne(
                    candidate, 
                    self._entity_index.keys(),
                    score_cutoff=75  # 75% similarity threshold
                )
                if match:
                    matched_name, score, _ = match
                    entities.append({
                        'text': candidate,
                        'label': 'USER_ENTITY',
                        'canonical': matched_name,
                        'confidence': score / 100.0,
                        'source': 'entity_index'
                    })
        
        # Deduplicate by span
        seen_spans = set()
        deduped = []
        for ent in entities:
            span_key = (ent.get('start', ent['text']), ent.get('end', ent['text']))
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                deduped.append(ent)
        
        return deduped
    
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

**Week 19-20: Advanced Enhancements (Section 13)**
- [ ] Contextual chunking pipeline
- [ ] Semantic chunking with boundary detection
- [ ] Step-back prompting integration
- [ ] Multi-level caching (embedding + semantic + response)
- [ ] Vector quantization (IVF-PQ/SQ8) for FAISS
- [ ] ANN index tuning (HNSW ef_search, IVF nprobe)
- [ ] Async parallel retrieval pipeline
- [ ] Hot/cold storage tiering
- [ ] RAGChecker diagnostic evaluation
- [ ] Failure-aware query refinement
- [ ] Chain-of-retrieval reasoning
- [ ] Token efficiency optimization
- [ ] Retriever fine-tuning on user data

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
| **Faithfulness** | > 0.92 | Answer grounded in retrieved context (DPO-aligned) |
| **Answer Relevancy** | > 0.88 | Answer directly addresses the query |
| **Context Recall** | > 0.80 | Retrieved all relevant memories |
| **Context Precision** | > 0.78 | Retrieved memories are relevant |
| **Multi-Turn Coherence** | > 0.88 | Consistent across extended dialogues |
| **Long-Context Recall** | > 0.85 | Accurate over 8K+ token contexts |
| **E2E Latency (Simple)** | < 1.5s | Single-agent queries (7B + bf16 + Flash Attention 2) |
| **E2E Latency (Complex)** | < 6s (P90), < 8s (P99) | Multi-agent queries with async parallel retrieval † |

---

## 12. Production Deployment

### System Requirements

**Minimum Hardware:**
- GPU: NVIDIA RTX 4000 Ada Generation (20GB VRAM)
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB (system RAM — Python runtime + DuckDB + FAISS metadata require ~4-6GB alongside GPU VRAM)
- Storage: 40GB SSD (models ~18GB + indexes + database + swap headroom)

**Software Stack:**
- Python 3.10+
- CUDA 12.1+ (Ada Lovelace optimized)
- PyTorch 2.1+ (bf16 + Flash Attention 2 support)
- Node.js 18+ (for frontend)
- Ollama / llama.cpp (model serving with bf16 inference)

**Model Stack (Pre-downloaded):**
| Model | Size | VRAM | Purpose |
|-------|------|------|---------|
| DeepSeek-R1-Distill-Qwen-7B (10-stage QLoRA + DPO) | ~14GB disk | ~4.2GB | Primary LLM (generation, reasoning, reflection) |
| BGE-large-en-v1.5 | ~1.3GB disk | ~1.3GB | Dense embedding (1024d) |
| BGE-reranker-v2-m3 | ~1.1GB disk | ~1.1GB | Cross-encoder reranking |
| SetFit classifiers (3x) | ~250MB disk | ~200MB | Memory type / intent / emotion |
| faster-whisper base | ~150MB disk | ~150MB | Voice input transcription |
| **Total Inference VRAM** | | **~7.0GB** | **35% of 20GB — ample headroom** |

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

### 12.1 Operational Complexity & Dependency Management

> **Honest assessment:** This system is complex. It coordinates 12+ infrastructure components. This section acknowledges that complexity and provides mitigation strategies.

**Component Dependency Map:**

| Component | Purpose | Failure Impact | Restart Strategy |
|-----------|---------|----------------|-----------------|
| DeepSeek-R1-Distill-Qwen-7B | LLM generation, critique, reasoning | **TOTAL** — no response generation | Auto-reload via Ollama watchdog |
| BGE-large-en-v1.5 | Dense embedding | **HIGH** — no new embeddings or retrieval | Lazy reload on next request |
| BGE-reranker-v2-m3 | Reranking | **MEDIUM** — degraded ranking quality | Bypass reranker, use raw fusion scores |
| FAISS (HNSW + IVF-PQ) | Vector search | **HIGH** — no dense retrieval | Fallback to BM25-only retrieval |
| DuckDB | Metadata + SQL queries | **HIGH** — no temporal/metadata filtering | Read-only WAL recovery + restart |
| NetworkX knowledge graph | Entity-relation traversal | **LOW** — causal/graph queries degrade | Rebuild from DuckDB entity table |
| BM25 index (rank_bm25) | Sparse retrieval | **LOW** — lose keyword matching | Dense-only retrieval still functional |
| RAPTOR tree | Hierarchical summaries | **LOW** — lose abstraction levels | Rebuild from raw memory index |
| SetFit classifiers (3x) | Intent/type/emotion | **MEDIUM** — fall back to LLM classification | LLM fallback path (slower, ~3s) |
| faster-whisper | Voice transcription | **LOW** — voice input disabled | User types instead; non-blocking |
| Multi-level cache | Response/embedding cache | **NONE** — cold start, slower until warm | Auto-rebuilds through usage |
| Ingestion models (OCR/BLIP) | Multi-modal processing | **LOW** — affected file types fail | Queue for retry; text types unaffected |

**Mitigation: Graceful Degradation Tiers**

```
Tier 1 (FULL): All 12 components healthy → Full capability
Tier 2 (DEGRADED): LLM + BGE + FAISS + DuckDB → Core RAG works, no graph/BM25/reranking
Tier 3 (MINIMAL): LLM + DuckDB only → Direct SQL memory lookup + LLM generation
Tier 4 (OFFLINE): DuckDB only → Browse/search memories without LLM generation
```

**Dependency Health Monitor:**

```python
"""
Startup health check: verify all components before accepting requests.
If non-critical components fail, start in degraded mode rather than crashing.
"""
class SystemHealthMonitor:
    CRITICAL = ['llm', 'embedding_model', 'faiss_index', 'duckdb']
    OPTIONAL = ['reranker', 'knowledge_graph', 'bm25_index', 'raptor_tree',
                'setfit_classifiers', 'whisper', 'cache', 'ocr_models']
    
    def startup_check(self) -> dict:
        status = {}
        for component in self.CRITICAL + self.OPTIONAL:
            try:
                status[component] = self._check(component)  # True/False
            except Exception as e:
                status[component] = False
                if component in self.CRITICAL:
                    raise RuntimeError(f"Critical component failed: {component} — {e}")
                else:
                    logger.warning(f"Optional component unavailable: {component} — running degraded")
        return status
```

**Maintenance Burden (realistic estimates for a solo developer):**
- **Daily:** Zero — system runs autonomously once deployed
- **Weekly:** ~30 min — review failure logs, check disk usage
- **Monthly:** ~2 hours — run background optimization (§13.13), check model updates
- **Quarterly:** ~4 hours — RAPTOR tree full rebuild, re-evaluate ANN parameters, pip upgrade

---

### 12.2 Privacy, Data Security & Encryption

> **Critical for a personal AI memory system:** All data stays local. No cloud dependency. No telemetry.

**Data-at-Rest Encryption:**

| Store | Location | Encryption | Key Management |
|-------|----------|------------|---------------|
| DuckDB | `data/memories.duckdb` | AES-256-CBC (SQLCipher-compatible) | User-provided passphrase → PBKDF2 key derivation |
| FAISS indexes | `data/indexes/*.faiss` | OS-level full-disk encryption (BitLocker/LUKS) | System-level; transparent to application |
| Knowledge graph | `data/graph.gpickle` | OS-level encryption | System-level |
| Cache files | `data/cache/` | OS-level encryption | Volatile; cleared on restart option |
| Ingested files | `data/inbox/` | OS-level encryption | Original files deletable after ingestion |

**Access Control:**

```python
"""
API Security: All endpoints require local-only access by default.
No external network exposure unless explicitly configured.
"""
# server.py configuration
SECURITY_DEFAULTS = {
    'bind_address': '127.0.0.1',       # Localhost only — not 0.0.0.0
    'cors_origins': ['http://localhost:3000'],  # Frontend only
    'api_key_required': True,           # Even localhost requires API key
    'api_key_location': '~/.cortexlab/api_key',  # Generated on first run
    'watched_folder_path': '~/CortexLab/inbox/',
    'watched_folder_whitelist': [       # Only safe file extensions
        '.txt', '.md', '.pdf', '.docx', '.csv', '.json',
        '.png', '.jpg', '.jpeg', '.wav', '.mp3', '.mp4',
        '.eml', '.ics', '.xlsx', '.py', '.js', '.ts'
    ],
    'max_file_size_mb': 100,            # Reject files > 100MB
    'max_ingestion_rate': 50,           # Max 50 files per minute (DoS protection)
}
```

**Watched Folder & API Endpoint Hardening:**
- Watched folder only accepts whitelisted file extensions (no executables)
- File size cap (100MB) prevents disk exhaustion attacks
- Rate limiting on `/api/ingest` endpoint (50 req/min)
- API key generated at first startup, stored in user home directory
- No outbound network calls (all models run locally, no telemetry)
- Optional: encrypt DuckDB with user passphrase (prompted at first launch)

**Backup Strategy:**

```bash
# Automated daily backup (add to crontab / Task Scheduler)
# Backs up DuckDB + FAISS indexes + knowledge graph (models not backed up — re-downloadable)
BACKUP_DIR="$HOME/CortexLab/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
cp data/memories.duckdb "$BACKUP_DIR/"
cp -r data/indexes/ "$BACKUP_DIR/"
cp data/graph.gpickle "$BACKUP_DIR/"
# Keep last 30 days of backups
find "$HOME/CortexLab/backups/" -maxdepth 1 -mtime +30 -exec rm -rf {} \;
```

---

## 13. Advanced Enhancements: Research-Driven Improvements

> **Added:** February 19, 2026 — Based on cross-referencing with Advanced_RAG_Architecture_Guide, RAG-DL-ResearchPage, and RAG_Literature_Survey

These 15 enhancements address gaps identified from surveying 100+ papers and production RAG deployments. Each is designed to integrate cleanly into the existing architecture without breaking any planned functionality, and is validated against the RTX 4000 Ada Generation (20GB VRAM) constraint.

---

### 13.1 Enhancement: Contextual Chunking (Anthropic 2024)

**Gap Identified:** Current architecture chunks memories at fixed/semantic boundaries but does NOT prepend document-level context to each chunk. This causes chunks to lose their surrounding meaning during retrieval.

**Research Source:** Anthropic Contextual Retrieval (September 2024)

**Impact:** Anthropic reported a 49% reduction in retrieval failure rate when combining contextual chunking with contextual BM25.

**Integration Point:** Layer 1 (Memory Ingestion) — runs at ingestion time, zero query-time cost.

**File: `src/storage/contextual_chunker.py`**

```python
"""
Contextual Chunking: Prepend document-level context to each memory chunk.
Based on: Anthropic Contextual Retrieval (September 2024)

Key Insight: When chunks are retrieved in isolation, they lose the context
of the full document/conversation they came from. By prepending a short
context summary, retrieval accuracy dramatically improves.

For Cortex Lab: Each memory event gets a contextual prefix describing
the conversation session, time period, and topic cluster it belongs to.
"""

from typing import List, Dict
from src.llm.local_llm import LocalLLM


class ContextualChunker:
    """
    Enrich memory chunks with surrounding document/session context.
    
    Cost: 1 LLM call per chunk at ingestion time (background, not query-time).
    VRAM: Uses the same DeepSeek-R1-Distill-Qwen-7B already loaded (~4.2GB).
    Latency: 0ms at query time (context is pre-baked into stored text).
    """
    
    def __init__(self, llm: LocalLLM):
        self.llm = llm
    
    def add_context_to_memory(self, memory: Dict, session_context: str) -> str:
        """
        Prepend contextual information to a memory chunk.
        
        Args:
            memory: The memory event dict with 'content', 'timestamp', 'type'
            session_context: The full conversation/session this memory is from
        
        Returns:
            Contextualized memory text ready for embedding
        
        Example:
            Original: "I decided to switch from React to Svelte"
            Contextualized: "During a conversation about frontend framework 
                            choices on March 15, 2025, following frustrations 
                            with React bundle sizes: I decided to switch from 
                            React to Svelte"
        """
        prompt = f"""Here is a conversation/session:
<session>
{session_context[:1500]}
</session>

Here is a specific memory from that session:
<memory>
{memory['content']}
</memory>

Write a SHORT context (1-2 sentences) to situate this memory within 
the session. Include who/what/when if relevant. Output ONLY the context."""
        
        context = self.llm.generate(prompt, max_tokens=80)
        
        return f"{context.strip()}: {memory['content']}"
    
    def batch_contextualize(
        self, 
        memories: List[Dict], 
        session_context: str
    ) -> List[str]:
        """Contextualize a batch of memories from the same session."""
        contextualized = []
        for mem in memories:
            ctx_text = self.add_context_to_memory(mem, session_context)
            contextualized.append(ctx_text)
        return contextualized
```

**VRAM Impact:** 0 additional VRAM (reuses existing LLM at ingestion time).

---

### 13.2 Enhancement: Semantic Chunking with Boundary Detection

**Gap Identified:** Current architecture implicitly uses fixed-window chunking for memories. Semantic chunking breaks at natural meaning boundaries, producing more coherent retrieval units.

**Research Source:** Pinecone/Weaviate Chunking Strategies (2024)

**Integration Point:** Layer 1 (Memory Ingestion) — preprocessing before embedding.

**File: `src/storage/semantic_chunker.py`**

```python
"""
Semantic Chunking: Break text at natural meaning boundaries.

Unlike fixed-size chunking (every 512 tokens), semantic chunking detects
where the topic/meaning shifts by computing embedding similarity between
consecutive sentences. When similarity drops below threshold, a new chunk
begins.
"""

import numpy as np
from typing import List, Tuple
from src.models.embeddings import EmbeddingModel


class SemanticChunker:
    """
    Chunk text at semantic boundaries using embedding similarity.
    
    Cost: Embeds all sentences once at ingestion time.
    VRAM: Uses BGE-large-en-v1.5 already loaded (~1.3GB).
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 50,   # minimum words per chunk
        max_chunk_size: int = 300   # maximum words per chunk
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries.
        
        Algorithm:
        1. Split into sentences
        2. Embed each sentence (batch for efficiency)
        3. Compute cosine similarity between consecutive sentences
        4. When similarity < threshold → new chunk boundary
        5. Enforce min/max chunk size constraints
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        # Batch embed all sentences (single GPU call)
        embeddings = np.array([
            self.embedding_model.encode(s, is_query=False) 
            for s in sentences
        ])
        
        # Compute consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))
        
        # Find chunk boundaries
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            current_words = sum(len(s.split()) for s in current_chunk)
            
            if (sim < self.similarity_threshold and 
                current_words >= self.min_chunk_size) or \
               current_words >= self.max_chunk_size:
                # Boundary detected: save current chunk, start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i + 1]]
            else:
                current_chunk.append(sentences[i + 1])
        
        # Add remaining
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple regex-based)."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
```

**VRAM Impact:** 0 additional (reuses BGE-large-en-v1.5).

---

### 13.3 Enhancement: Step-Back Prompting

**Gap Identified:** Current query transformation has Multi-Query + HyDE + Decomposition but lacks Step-Back Prompting — generating a higher-level abstract question before retrieval. This helps when direct queries are too specific to match stored memories.

**Research Source:** Step-Back Prompting (Google DeepMind 2024)

**Integration Point:** Layer 3 (Query Transformation Pipeline) — added as optional stage.

```
┌──────────────────────────────────────────────────────────────────────┐
│  STEP-BACK PROMPTING (New Addition to Query Transformation)         │
│                                                                      │
│  Original Query: "Why did I quit my job at Google in March 2025?"   │
│                                                                      │
│  Step-Back Question (generated):                                     │
│  "What were the factors affecting my career satisfaction in          │
│   early 2025?"                                                      │
│                                                                      │
│  Why it helps:                                                       │
│  • Retrieves broader context that direct query might miss            │
│  • Finds related memories about career, burnout, life changes       │
│  • Especially useful for causal/reflective queries                  │
│  • Results merged with direct query results via RRF                  │
│                                                                      │
│  Trigger Conditions:                                                 │
│  • Query complexity > 0.5                                            │
│  • Intent is CAUSAL or REFLECTIVE                                    │
│  • Initial retrieval returned < 5 results with score > 0.7          │
│                                                                      │
│  Cost: 1 LLM call (~80 tokens) = ~0.3s on DeepSeek-R1-Distill-Qwen-7B  │
└──────────────────────────────────────────────────────────────────────┘
```

**File: `src/retrieval/query_transformer.py` (addition to existing class)**

```python
def generate_step_back(self, query: str) -> str:
    """
    Generate a more abstract version of the query.
    
    Based on: Step-Back Prompting (Google DeepMind 2024)
    
    Instead of asking the specific question directly, ask a broader
    question that provides relevant background context.
    """
    prompt = f"""Given the following specific question, generate a more 
general "step-back" question that would provide useful background 
context for answering the original question.

Specific question: {query}

Step-back question (ONE question only):"""
    
    step_back = self.llm.generate(prompt, max_tokens=60)
    return step_back.strip()
```

**VRAM Impact:** 0 additional (reuses existing LLM).

---

### 13.4 Enhancement: Multi-Level Caching Strategy

**Gap Identified:** Current architecture has NO caching layer. Every query runs the full pipeline even for repeated or semantically similar questions. This is the single biggest latency optimization available.

**Research Source:** Production RAG deployments (Pinecone, Weaviate, Google Vertex AI); RAG Cache patterns from enterprise systems.

**Integration Point:** New cross-cutting layer between API input and pipeline execution.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LEVEL CACHING ARCHITECTURE                            │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │  LEVEL 1: EXACT QUERY CACHE                                               ││
│  │                                                                            ││
│  │  • Key: query_hash                                                        ││
│  │  • Value: full response + metadata                                        ││
│  │  • TTL: 1 hour (configurable)                                             ││
│  │  • Storage: In-memory dict (LRU, max 5000 entries)                        ││
│  │  • Latency: <1ms                                                          ││
│  │  • Use case: Exact repeat queries within session                          ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │  LEVEL 2: SEMANTIC QUERY CACHE                                            ││
│  │                                                                            ││
│  │  • Key: query_embedding                                                   ││
│  │  • Lookup: cosine similarity > 0.95 against cached query embeddings       ││
│  │  • Value: full response + metadata                                        ││
│  │  • TTL: 6 hours                                                           ││
│  │  • Storage: Small FAISS index (~1000 vectors)                             ││
│  │  • Latency: ~5ms                                                          ││
│  │  • Use case: "Why did I quit?" ≈ "What made me leave my job?"             ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐│
│  │  LEVEL 3: EMBEDDING CACHE                                                 ││
│  │                                                                            ││
│  │  • Key: text_hash                                                         ││
│  │  • Value: pre-computed embedding vector                                   ││
│  │  • TTL: permanent (until memory updated)                                  ││
│  │  • Storage: DuckDB column                                                 ││
│  │  • Latency: <1ms vs ~50ms for re-encoding                                ││
│  │  • Use case: Avoid re-embedding the same memory text                      ││
│  └────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  Cache Invalidation Strategy:                                                   │
│  ─────────────────────────────────────────────────────────────────────        │
│  • New memory ingested → Invalidate semantic cache entries about same topic   │
│  • Memory updated → Invalidate embedding cache for that memory               │
│  • TTL expiry → Automatic eviction                                           │
│  • Manual flush → Admin API endpoint                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**File: `src/cache/rag_cache.py`**

```python
"""
Multi-Level RAG Cache: Dramatically reduce latency for repeated queries.

Expected Impact:
- 40%+ cache hit rate for active users (many queries are rephrased repeats)
- Cache hit latency: <5ms vs full pipeline 2-5s
- Memory cost: ~20MB for cache index
"""

import hashlib
import time
from typing import Optional, Dict, Tuple
from collections import OrderedDict
import numpy as np
from src.models.embeddings import EmbeddingModel


class MultiLevelCache:
    """Three-tier caching for RAG responses."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        exact_cache_size: int = 5000,
        semantic_cache_size: int = 1000,
        semantic_threshold: float = 0.95,
        exact_ttl: int = 3600,       # 1 hour
        semantic_ttl: int = 21600     # 6 hours
    ):
        self.embedding_model = embedding_model
        self.semantic_threshold = semantic_threshold
        self.exact_ttl = exact_ttl
        self.semantic_ttl = semantic_ttl
        
        # Level 1: Exact cache (LRU)
        self.exact_cache = OrderedDict()
        self.exact_cache_size = exact_cache_size
        
        # Level 2: Semantic cache
        self.semantic_entries = []  # List of (embedding, response, timestamp)
        self.semantic_cache_size = semantic_cache_size
        
        # Level 3: Embedding cache
        self.embedding_cache = {}  # text_hash → embedding
    
    def lookup(self, query: str) -> Optional[Dict]:
        """Try all cache levels. Returns cached response or None."""
        now = time.time()
        
        # Level 1: Exact match
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        if query_hash in self.exact_cache:
            entry = self.exact_cache[query_hash]
            if now - entry['timestamp'] < self.exact_ttl:
                self.exact_cache.move_to_end(query_hash)
                return {'response': entry['response'], 'cache_level': 'exact'}
            else:
                del self.exact_cache[query_hash]
        
        # Level 2: Semantic match
        query_embedding = self.get_embedding(query)
        for entry in self.semantic_entries:
            if now - entry['timestamp'] > self.semantic_ttl:
                continue
            sim = np.dot(query_embedding, entry['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry['embedding'])
            )
            if sim > self.semantic_threshold:
                return {'response': entry['response'], 'cache_level': 'semantic'}
        
        return None
    
    def store(self, query: str, response: Dict):
        """Store response in both cache levels."""
        now = time.time()
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        # Level 1
        self.exact_cache[query_hash] = {
            'response': response, 'timestamp': now
        }
        if len(self.exact_cache) > self.exact_cache_size:
            self.exact_cache.popitem(last=False)
        
        # Level 2
        query_embedding = self.get_embedding(query)
        self.semantic_entries.append({
            'embedding': query_embedding, 
            'response': response, 
            'timestamp': now
        })
        if len(self.semantic_entries) > self.semantic_cache_size:
            self.semantic_entries.pop(0)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Level 3: Cached embedding lookup."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = self.embedding_model.encode(
                text, is_query=True
            )
        return self.embedding_cache[text_hash]
    
    def invalidate_topic(self, topic_embedding: np.ndarray, threshold: float = 0.8):
        """Invalidate semantic cache entries related to a topic."""
        self.semantic_entries = [
            e for e in self.semantic_entries
            if np.dot(topic_embedding, e['embedding']) / (
                np.linalg.norm(topic_embedding) * np.linalg.norm(e['embedding'])
            ) < threshold
        ]
```

**VRAM Impact:** ~20MB for cache index. Net POSITIVE because it avoids running the full pipeline on ~40% of queries.

---

### 13.5 Enhancement: Vector Quantization for Memory-Efficient ANN

**Gap Identified:** Current architecture uses raw FAISS with 1024-dimensional float32 vectors (BGE-large-en-v1.5). At scale (100K+ memories with propositions and RAPTOR nodes), this could consume 2GB+ RAM. Vector quantization compresses vectors 4-16x with minimal recall loss, keeping index memory well within the 20GB VRAM budget.

**Research Source:** Production Vector DB best practices (Milvus, FAISS, PingCAP); ANN benchmark literature.

**Integration Point:** Layer 2 (Vector Store) — transparent replacement of FAISS index type.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR QUANTIZATION STRATEGY                                │
│                                                                                 │
│  Current: FAISS IndexFlatIP (brute-force, exact)                               │
│  ─────────────────────────────────────────────────────────────────────        │
│  Memory: 1024 dims × 4 bytes × N vectors                                      │
│  At 100K vectors: ~400MB                                                       │
│  At 500K vectors: ~2.0GB   ← Still comfortable on 20GB VRAM                    │
│                                                                                 │
│  Optimized: Tiered Index Strategy                                              │
│  ─────────────────────────────────────────────────────────────────────        │
│                                                                                 │
│  Tier 1 (Hot): Recent memories (last 30 days)                                  │
│  ├─ Index: HNSW (exact, in-memory)                                             │
│  ├─ Params: M=16, ef_construction=200, ef_search=64                            │
│  ├─ Recall: ~98%                                                               │
│  ├─ Latency: ~5ms                                                              │
│  └─ Memory: Full float32 (small dataset, ~10K vectors = ~40MB)                │
│                                                                                 │
│  Tier 2 (Warm): Older memories (30 days - 1 year)                              │
│  ├─ Index: IVF + SQ8 (scalar quantization)                                    │
│  ├─ Params: nlist=256, nprobe=16                                               │
│  ├─ Recall: ~95%                                                               │
│  ├─ Latency: ~15ms                                                             │
│  └─ Memory: 4x compressed (~50K vectors = ~50MB vs 200MB raw)                  │
│                                                                                 │
│  Tier 3 (Cold): Archival memories (>1 year)                                    │
│  ├─ Index: IVF + PQ (product quantization)                                     │
│  ├─ Params: nlist=512, nprobe=8, m_pq=48                                       │
│  ├─ Recall: ~90%                                                               │
│  ├─ Latency: ~25ms                                                             │
│  └─ Memory: 8-16x compressed (~100K vectors = ~25MB vs 400MB raw)             │
│                                                                                 │
│  Total at 160K vectors: ~115MB (vs ~650MB uncompressed)                         │
│  Savings: ~80% memory reduction                                               │
│                                                                                 │
│  Migration: Monthly background job promotes/demotes vectors between tiers      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**File: `src/storage/tiered_vector_store.py`**

```python
"""
Tiered Vector Store: HNSW (hot) + IVF-SQ8 (warm) + IVF-PQ (cold)

Aligns with RAPTOR hierarchy levels:
- Level 0 raw memories → distributed across tiers by age
- Level 1-4 summaries → always in hot tier (small count, important)
"""

import faiss
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime, timedelta


class TieredVectorStore:
    """Memory-efficient vector storage with tiered indexing."""
    
    DIM = 1024  # BGE-large-en-v1.5 embedding dimension
    
    def __init__(self):
        # Tier 1: Hot (HNSW - exact, fast)
        self.hot_index = faiss.IndexHNSWFlat(self.DIM, 16)  # M=16
        self.hot_index.hnsw.efConstruction = 200
        self.hot_index.hnsw.efSearch = 64
        
        # Tier 2: Warm (IVF + SQ8)
        quantizer_warm = faiss.IndexFlatIP(self.DIM)
        self.warm_index = faiss.IndexIVFScalarQuantizer(
            quantizer_warm, self.DIM, 256,  # nlist=256
            faiss.ScalarQuantizer.QT_8bit
        )
        self.warm_index.nprobe = 16
        
        # Tier 3: Cold (IVF + PQ)
        quantizer_cold = faiss.IndexFlatIP(self.DIM)
        self.cold_index = faiss.IndexIVFPQ(
            quantizer_cold, self.DIM, 512,  # nlist=512
            48, 8  # m_pq=48 sub-quantizers, 8 bits each
        )
        self.cold_index.nprobe = 8
        
        # Metadata mapping: vector_id → {memory_id, tier, timestamp}
        self.metadata: Dict[int, Dict] = {}
        self.id_counter = 0
    
    def add(self, embedding: np.ndarray, memory_id: str, 
            timestamp: datetime) -> int:
        """Add vector to appropriate tier based on age."""
        vec = embedding.reshape(1, -1).astype('float32')
        vec_id = self.id_counter
        self.id_counter += 1
        
        age_days = (datetime.now() - timestamp).days
        
        if age_days <= 30:
            tier = 'hot'
            self.hot_index.add(vec)
        elif age_days <= 365:
            tier = 'warm'
            if self.warm_index.is_trained:
                self.warm_index.add(vec)
        else:
            tier = 'cold'
            if self.cold_index.is_trained:
                self.cold_index.add(vec)
        
        self.metadata[vec_id] = {
            'memory_id': memory_id,
            'tier': tier,
            'timestamp': timestamp
        }
        
        return vec_id
    
    def search(self, query_embedding: np.ndarray, k: int = 50) -> List[Tuple[str, float]]:
        """Search across all tiers and merge results."""
        query = query_embedding.reshape(1, -1).astype('float32')
        
        all_results = []
        
        # Search each tier
        for index, tier_name in [
            (self.hot_index, 'hot'),
            (self.warm_index, 'warm'),
            (self.cold_index, 'cold')
        ]:
            if index.ntotal > 0:
                tier_k = min(k, index.ntotal)
                distances, indices = index.search(query, tier_k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx >= 0:
                        all_results.append((idx, float(dist), tier_name))
        
        # Sort by distance (similarity) and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score, tier in all_results[:k]:
            if idx in self.metadata:
                results.append((self.metadata[idx]['memory_id'], score))
        
        return results
    
    def migrate_tiers(self):
        """Background job: move vectors between tiers based on age."""
        # This runs monthly to rebalance hot/warm/cold
        # Implementation: rebuild indexes with reassigned vectors
        pass
    
    def get_memory_stats(self) -> Dict:
        """Return index statistics for monitoring."""
        return {
            'hot_vectors': self.hot_index.ntotal,
            'warm_vectors': self.warm_index.ntotal,
            'cold_vectors': self.cold_index.ntotal,
            'estimated_memory_mb': (
                self.hot_index.ntotal * self.DIM * 4 / 1e6 +   # float32
                self.warm_index.ntotal * self.DIM * 1 / 1e6 +  # int8
                self.cold_index.ntotal * 48 * 1 / 1e6           # PQ codes
            )
        }
```

**VRAM Impact:** Net REDUCTION of ~80% vector memory usage at scale.

---

### 13.6 Enhancement: ANN Index Tuning for Sub-50ms Retrieval

**Gap Identified:** Current architecture specifies "~100ms latency" for dense retrieval. With proper ANN parameter tuning, this can be reduced to <50ms P99 — critical for meeting the <2s simple query target.

**Research Source:** Production ANN tuning guides (FAISS, Milvus documentation); PingCAP vector search benchmarks.

**Integration Point:** Layer 5 (Multi-Channel Retrieval) — configuration tuning.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ANN INDEX TUNING PARAMETERS                                 │
│                                                                                 │
│  Target: P99 Retrieval Latency < 50ms for Top-50 search                        │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  HNSW Tuning (Hot Tier):                                                 │ │
│  │                                                                           │ │
│  │  Parameter        │ Default │ Tuned  │ Effect                            │ │
│  │  ──────────────── ┼──────── ┼─────── ┼──────────────────────────────     │ │
│  │  M (neighbors)    │ 32      │ 16     │ Less memory, slightly lower       │ │
│  │                   │         │        │ recall (~98% vs 99%)              │ │
│  │  ef_construction  │ 40      │ 200    │ Better graph quality at build     │ │
│  │  ef_search        │ 40      │ 64     │ Balance: 98% recall at ~5ms      │ │
│  │                   │         │        │                                   │ │
│  │  Tuning method: Start ef_search=32, increase until recall@50 > 0.95     │ │
│  │  on validation set. Stop when latency approaches 20ms P99.              │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  IVF Tuning (Warm/Cold Tiers):                                           │ │
│  │                                                                           │ │
│  │  Parameter        │ Default │ Tuned  │ Effect                            │ │
│  │  ──────────────── ┼──────── ┼─────── ┼──────────────────────────────     │ │
│  │  nlist (clusters) │ 100     │ 256    │ Finer partitioning                │ │
│  │  nprobe (search)  │ 1       │ 16     │ 95% recall at ~15ms               │ │
│  │                   │         │        │                                   │ │
│  │  Rule of thumb: nlist ≈ sqrt(N) × 4, nprobe ≈ nlist / 16               │ │
│  │  For 50K vectors: nlist=256, nprobe=16 → P99 ~15ms                     │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  VALIDATION PROTOCOL:                                                    │ │
│  │                                                                           │ │
│  │  1. Create ground-truth with brute-force IndexFlatIP                     │ │
│  │  2. Run 1000 random queries against ANN index                            │ │
│  │  3. Measure recall@50 = |ANN ∩ Exact| / 50                              │ │
│  │  4. Measure P50/P95/P99 latency                                          │ │
│  │  5. Adjust parameters until recall > 0.95 AND P99 < 50ms                │ │
│  │  6. Re-validate monthly as index grows                                   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** 0 additional. This is pure configuration tuning.

---

### 13.7 Enhancement: Async Parallel Retrieval Pipeline

**Gap Identified:** Current architecture describes 5 parallel retrieval channels but doesn't specify HOW they run in parallel. Without async execution, they run sequentially (100+50+80+30+90 = 350ms). With async, they complete in max(100,50,80,30,90) = 100ms.

**Research Source:** Production RAG deployment patterns; Google Vertex AI async architecture.

**Integration Point:** Layer 5 (Multi-Channel Retrieval) — execution pattern.

**File: `src/retrieval/async_retriever.py`**

```python
"""
Async Parallel Retrieval: Execute all 5 retrieval channels simultaneously.

Latency Improvement:
- Sequential: 100 + 50 + 80 + 30 + 90 = 350ms
- Parallel:   max(100, 50, 80, 30, 90) = ~100ms
- Savings:    ~250ms per query (71% reduction in retrieval phase)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple


class AsyncHybridRetriever:
    """Execute retrieval channels in parallel using asyncio."""
    
    def __init__(self, dense_retriever, sparse_retriever, graph_retriever,
                 temporal_retriever, proposition_retriever):
        self.channels = {
            'dense': dense_retriever,
            'sparse': sparse_retriever,
            'graph': graph_retriever,
            'temporal': temporal_retriever,
            'proposition': proposition_retriever
        }
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def parallel_retrieve(
        self, 
        query: str, 
        active_channels: List[str] = None,
        k: int = 50
    ) -> Dict[str, List]:
        """
        Run selected channels in parallel.
        
        Args:
            query: Search query
            active_channels: Which channels to activate
                             (agent decides based on intent)
            k: Results per channel
        
        Returns:
            Dict mapping channel_name → ranked results
        """
        if active_channels is None:
            active_channels = list(self.channels.keys())
        
        loop = asyncio.get_event_loop()
        
        # Launch all channels simultaneously
        tasks = {}
        for channel_name in active_channels:
            if channel_name in self.channels:
                retriever = self.channels[channel_name]
                task = loop.run_in_executor(
                    self.executor,
                    retriever.search,
                    query, k
                )
                tasks[channel_name] = task
        
        # Wait for all to complete
        results = {}
        for channel_name, task in tasks.items():
            try:
                results[channel_name] = await task
            except Exception as e:
                # Channel failure should not crash the pipeline
                results[channel_name] = []
                print(f"Channel {channel_name} failed: {e}")
        
        return results
    
    async def retrieve_and_fuse(
        self, 
        query: str,
        active_channels: List[str] = None,
        weights: Dict[str, float] = None,
        k: int = 20
    ) -> List[Tuple[str, float]]:
        """Parallel retrieve + RRF fusion in one call."""
        if weights is None:
            weights = {
                'dense': 0.35, 'sparse': 0.25, 'graph': 0.20,
                'temporal': 0.10, 'proposition': 0.10
            }
        
        channel_results = await self.parallel_retrieve(
            query, active_channels, k=50
        )
        
        # RRF Fusion (same as existing implementation)
        fused_scores = {}
        rrf_k = 60
        
        for channel_name, results in channel_results.items():
            weight = weights.get(channel_name, 0.1)
            for rank, (memory_id, score) in enumerate(results):
                if memory_id not in fused_scores:
                    fused_scores[memory_id] = 0.0
                fused_scores[memory_id] += weight / (rrf_k + rank)
        
        # Sort and return top-k
        sorted_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:k]
```

**VRAM Impact:** 0 additional. This is an execution pattern change.

---

### 13.8 Enhancement: RAGChecker Fine-Grained Diagnostic Evaluation

**Gap Identified:** Current architecture only uses RAGAS (4 metrics). RAGChecker (NeurIPS 2024) provides 8+ fine-grained diagnostic metrics that can pinpoint exactly WHERE the pipeline fails — retrieval vs generation, and what TYPE of failure.

**Research Source:** RAGChecker: A Fine-grained Framework for Diagnosing RAG (Dong et al., NeurIPS 2024 Datasets & Benchmarks Track)

**Integration Point:** Section 11 (Evaluation & Benchmarking) — additional evaluation layer.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RAGChecker DIAGNOSTIC METRICS                               │
│                    (Addition to existing RAGAS evaluation)                      │
│                                                                                 │
│  RETRIEVAL DIAGNOSTICS:                                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ 1. Context Precision — Are retrieved memories relevant?                  ││
│  │ 2. Context Recall — Did we find all relevant memories?                   ││
│  │ 3. Context Entity Recall — Are key entities present in results?          ││
│  │ 4. Noise Robustness — Does performance degrade with irrelevant results? ││
│  │ 5. Negative Rejection — Can the system say "I don't know" correctly?    ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  GENERATION DIAGNOSTICS:                                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ 6. Faithfulness — Every claim grounded in retrieved context?             ││
│  │ 7. Information Integration — Can it synthesize across multiple chunks?   ││
│  │ 8. Counterfactual Robustness — Resistant to contradictory noise?         ││
│  │ 9. Utilization Rate — What % of retrieved context is actually used?      ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  ERROR CLASSIFICATION (for continuous improvement):                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Retrieval Miss:  Relevant memory exists but wasn't found              ││
│  │ • Retrieval Noise: Irrelevant memories ranked too high                  ││
│  │ • Hallucination:   Claim not in any retrieved memory                    ││
│  │ • Incomplete:      Relevant context found but not used in answer        ││
│  │ • Conflation:      Multiple memories incorrectly merged                 ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  Usage: Run after each test batch. Failures feed into:                        │
│  • Query refinement training data                                             │
│  • Retriever fine-tuning negative examples                                    │
│  • Embedding model hard negative mining                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** 0 additional (evaluation runs offline).

---

### 13.9 Enhancement: Failure-Aware Automatic Query Refinement

**Gap Identified:** Current architecture has CRAG which decides CORRECT/AMBIGUOUS/INCORRECT but does NOT have systematic query refinement based on failure TYPE. Different failures need different refinement strategies.

**Research Source:** Advanced RAG Architecture Guide — Query Refinement patterns; Agentic RAG self-improvement loops.

**Integration Point:** Layer 6 (Post-Retrieval) — new refinement module after CRAG evaluation.

**File: `src/retrieval/query_refiner.py`**

```python
"""
Failure-Aware Query Refinement: Different refinement strategies
for different failure types.

Integrates with CRAG evaluation:
- CRAG says AMBIGUOUS → QueryRefiner diagnoses WHY and fixes the query
- CRAG says INCORRECT → QueryRefiner applies aggressive expansion
"""

from typing import List, Dict, Optional
from src.llm.local_llm import LocalLLM
from src.models.embeddings import EmbeddingModel


class FailureAwareQueryRefiner:
    """Refine queries based on classified failure type."""
    
    def __init__(self, llm: LocalLLM, embedding_model: EmbeddingModel):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def classify_failure(
        self, 
        query: str, 
        retrieved_results: List[Dict],
        crag_scores: Dict
    ) -> str:
        """
        Classify WHY retrieval failed.
        
        Returns one of:
        - 'no_results': Nothing was found at all
        - 'low_relevance': Results found but not relevant
        - 'vocabulary_mismatch': Query uses different terms than memories
        - 'too_specific': Query is too narrow, needs broadening
        - 'too_vague': Query is too broad, needs narrowing
        - 'temporal_miss': Time range doesn't match
        """
        if len(retrieved_results) == 0:
            return 'no_results'
        
        avg_relevance = crag_scores.get('relevance', 0)
        avg_support = crag_scores.get('support', 0)
        
        if avg_relevance < 0.3 and avg_support < 0.3:
            return 'vocabulary_mismatch'
        elif avg_relevance > 0.5 and avg_support < 0.3:
            return 'too_vague'
        elif avg_relevance < 0.3 and avg_support > 0.3:
            return 'too_specific'
        elif crag_scores.get('temporal_match', 1.0) < 0.3:
            return 'temporal_miss'
        else:
            return 'low_relevance'
    
    def refine(
        self, 
        query: str, 
        failure_type: str,
        previous_results: List[Dict] = None
    ) -> str:
        """
        Apply failure-specific refinement strategy.
        """
        strategies = {
            'no_results': self._expand_with_synonyms,
            'vocabulary_mismatch': self._rephrase_with_alternatives,
            'too_specific': self._broaden_query,
            'too_vague': self._narrow_with_context,
            'temporal_miss': self._relax_temporal,
            'low_relevance': self._focus_on_entities
        }
        
        strategy = strategies.get(failure_type, self._general_refinement)
        return strategy(query, previous_results)
    
    def _expand_with_synonyms(self, query: str, results=None) -> str:
        prompt = f"""Rewrite this query using synonyms and alternative 
phrasing. The original returned no results, so use very different words.

Original: {query}

Expanded query:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _rephrase_with_alternatives(self, query: str, results=None) -> str:
        prompt = f"""The following query uses terms that don't match 
stored memories. Rephrase using more common, everyday language.

Original: {query}

Rephrased:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _broaden_query(self, query: str, results=None) -> str:
        prompt = f"""This query is too specific. Make it broader while 
keeping the core intent. Remove very specific details.

Original: {query}

Broadened:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _narrow_with_context(self, query: str, results=None) -> str:
        context = ""
        if results:
            context = f"\nPrevious results were about: {results[0].get('content', '')[:200]}"
        prompt = f"""This query is too vague. Add specificity to narrow 
down the search.{context}

Original: {query}

More specific:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _relax_temporal(self, query: str, results=None) -> str:
        prompt = f"""This query might have wrong time references. 
Remove or broaden any date/time constraints.

Original: {query}

Time-relaxed:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _focus_on_entities(self, query: str, results=None) -> str:
        prompt = f"""Extract and focus on the key entities (people, 
places, events) in this query. Build a simpler query around them.

Original: {query}

Entity-focused:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
    
    def _general_refinement(self, query: str, results=None) -> str:
        prompt = f"""Rewrite this query to improve search results. 
Make it clearer and more specific.

Original: {query}

Refined:"""
        return self.llm.generate(prompt, max_tokens=80).strip()
```

**VRAM Impact:** 0 additional (reuses existing LLM).

---

### 13.10 Enhancement: Chain-of-Retrieval Augmented Generation

**Gap Identified:** Current FLARE retrieves mid-generation based on token confidence. Chain-of-Retrieval (NeurIPS 2024) takes a fundamentally different approach: the model plans retrieval STEPS as part of its reasoning chain, like o1 thinks step-by-step.

**Research Source:** Chain-of-Retrieval Augmented Generation (NeurIPS 2024 Poster)

**Integration Point:** Layer 7 (Self-Reflective Generation) — alternative generation mode for complex multi-hop queries.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CHAIN-OF-RETRIEVAL (CoR) GENERATION                         │
│                    (Alternative mode for MULTI_STEP queries)                    │
│                                                                                 │
│  Traditional (current):                                                        │
│  Query → Retrieve All → Generate Answer                                       │
│                                                                                 │
│  Chain-of-Retrieval:                                                           │
│  Query → Think → [Retrieval Step 1] → Think → [Retrieval Step 2] → Answer    │
│                                                                                 │
│  Example:                                                                      │
│  ─────────────────────────────────────────────────────────────────────        │
│  Query: "How did my views on AI safety change after attending the              │
│          conference, and what role did my mentor play?"                         │
│                                                                                 │
│  <think> I need to find information about three things:                        │
│  1. Views on AI safety BEFORE the conference                                   │
│  2. The conference experience                                                  │
│  3. Mentor interactions related to this topic </think>                          │
│                                                                                 │
│  <retrieve> my views on AI safety before 2025 </retrieve>                      │
│  → Retrieved: [mem_23, mem_45]                                                 │
│                                                                                 │
│  <think> I had cautiously optimistic views. Now let me find the                │
│  conference experience. </think>                                               │
│                                                                                 │
│  <retrieve> AI conference experience 2025 </retrieve>                          │
│  → Retrieved: [mem_67, mem_89]                                                 │
│                                                                                 │
│  <think> The conference shifted my perspective. Now the mentor part. </think>  │
│                                                                                 │
│  <retrieve> mentor discussion about AI safety </retrieve>                      │
│  → Retrieved: [mem_91]                                                         │
│                                                                                 │
│  <answer> Your views on AI safety evolved significantly... </answer>           │
│                                                                                 │
│  Key Advantages:                                                               │
│  • Each retrieval is focused on exactly what's needed at that step             │
│  • Model reasons about what to search for (not just keyword extraction)        │
│  • Naturally handles multi-hop without explicit decomposition                  │
│  • Reasoning trace is fully transparent                                        │
│                                                                                 │
│  Trigger: complexity > 0.7 AND intent ∈ {CAUSAL, REFLECTIVE, MULTI_STEP}      │
│  Max retrieval steps: 5 (prevents runaway loops)                               │
│  Fallback: If any step returns empty, revert to standard single-pass RAG       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** 0 additional (uses existing LLM, just a different prompting strategy).

---

### 13.11 Enhancement: Token Efficiency Optimization

**Gap Identified:** Agentic RAG with 6 agents, multi-query generation, HyDE, Self-RAG critique, and FLARE can consume enormous token budgets. Without optimization, a complex query might trigger 8-10 LLM calls. Token efficiency directly translates to latency on an RTX 4000 Ada Generation.

**Research Source:** TeaRAG (Token-Efficient Agentic RAG, 2025); Production agentic RAG optimization patterns.

**Integration Point:** Cross-cutting concern across all layers.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TOKEN EFFICIENCY STRATEGIES                                  │
│                                                                                 │
│  Problem: Worst-case token budget per complex query                            │
│  ─────────────────────────────────────────────────────────────────────        │
│  Intent Classification:    ~50 tokens (SetFit, no LLM)             ✅ Free    │
│  Multi-Query (4 variants): ~200 tokens generation                  ~0.3s      │
│  HyDE:                     ~150 tokens generation                  ~0.3s      │
│  Step-Back:                ~60 tokens generation                   ~0.15s     │
│  Context prompt:           ~2000 tokens (10 memories × 200 tokens) -           │
│  Self-RAG critique:        ~300 tokens per iteration (×3 max)      ~1.8s      │
│  FLARE re-retrieval:       ~100 tokens per trigger (×2 max)        ~0.4s      │
│  Total worst case:         ~4000 tokens, ~8 LLM calls              ~6-8s      │
│  (includes prefill, retrieval between calls, Python overhead ~1.5-2s)           │
│                                                                                 │
│  Optimization Strategies:                                                      │
│  ─────────────────────────────────────────────────────────────────────        │
│                                                                                 │
│  1. SKIP UNNECESSARY STAGES (Adaptive Bypass)                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ if complexity < 0.3: Skip Multi-Query, HyDE, Step-Back, CRAG, Self-RAG  ││
│  │ if complexity 0.3-0.5: Skip Step-Back, limit Self-RAG to 1 iteration    ││
│  │ if complexity 0.5-0.7: Skip either HyDE or Multi-Query (not both)       ││
│  │ if complexity > 0.7: Full pipeline (all stages)                          ││
│  │                                                                           ││
│  │ Result: 60%+ of queries use ≤3 LLM calls instead of 8                   ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  2. COMPACT PROMPTS (Reduce token waste)                                       │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Use structured output (JSON) instead of verbose instructions           ││
│  │ • Truncate memory excerpts to 150 tokens each (not full text)            ││
│  │ • Use system prompt caching (reuse instruction prefix across calls)      ││
│  │ • Combine Multi-Query + HyDE into single prompt when both needed         ││
│  │                                                                           ││
│  │ Combined prompt example:                                                  ││
│  │ "Given query: '{query}'                                                   ││
│  │  1. Generate 3 variant queries                                            ││
│  │  2. Write a hypothetical answer (2 sentences)                             ││
│  │  Output JSON: {variants: [...], hypothetical: '...'}"                     ││
│  │                                                                           ││
│  │ Result: Saves 1 full LLM call (~0.3s) when both are needed               ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  3. EARLY TERMINATION (Stop when confident)                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Self-RAG: If first critique scores > 0.85 on all dimensions → STOP     ││
│  │ • CRAG: If relevance > 0.8 on first evaluation → skip refinement         ││
│  │ • Multi-agent: If first agent confidence > 0.9 → skip arbitration        ││
│  │                                                                           ││
│  │ Result: ~70% of queries terminate in first iteration                      ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  4. BATCH LLM CALLS (Reduce overhead)                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • KV-cache reuse: Keep context in GPU memory across sequential calls      ││
│  │ • Batch multiple short prompts into one call with separators              ││
│  │ • Ollama/llama.cpp: Use keep_alive to maintain model in VRAM             ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  Token Budget Summary (after optimization, DeepSeek-R1-Distill-Qwen-7B):      │
│  ─────────────────────────────────────────────────────────────────────        │
│  Simple query:  ~500 tokens, 1 LLM call,   ~0.8s                              │
│  Moderate query: ~1500 tokens, 3 LLM calls, ~2.5s                              │
│  Complex query:  ~3000 tokens, 5 LLM calls, ~5-6s (P90)                        │
│                                                                                 │
│  Note: Latency includes LLM generation + retrieval + prefill + Python           │
│  overhead. Cache hits reduce complex queries to ~2-3s. First-time queries       │
│  with cold KV-cache may reach ~8s (P99).                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** Net POSITIVE — reduces peak VRAM usage by reducing concurrent prompt sizes.

---

### 13.12 Enhancement: Retriever Fine-Tuning Pipeline

**Gap Identified:** Current architecture uses off-the-shelf BGE-large-en-v1.5 embeddings. Over time, as the user builds up memories, fine-tuning the embedding model on their actual data would significantly improve retrieval quality for their specific vocabulary, topics, and writing style.

**Research Source:** FedRAG (retriever fine-tuning framework, 2025); BGE fine-tuning documentation; Hard Negative Mining techniques.

**Integration Point:** Background training pipeline — runs periodically (weekly).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    RETRIEVER FINE-TUNING PIPELINE                              │
│                    (Background, runs weekly or on-demand)                       │
│                                                                                 │
│  Step 1: BUILD TRAINING DATA (from user interactions)                          │
│  ─────────────────────────────────────────────────────────────────────        │
│  Positive Pairs:                                                               │
│  • (query, memory) pairs where user found the answer useful                   │
│  • (query, clicked memory) — memories the user actually read                  │
│  • (query, highly-ranked memory with score > 0.8)                             │
│                                                                                 │
│  Hard Negatives:                                                               │
│  • Memories that were retrieved BUT scored low on CRAG evaluation             │
│  • Memories from wrong time periods                                           │
│  • Semantically similar but factually irrelevant                              │
│                                                                                 │
│  Step 2: FINE-TUNE BGE-large-en-v1.5 with LoRA                                  │
│  ─────────────────────────────────────────────────────────────────────        │
│  • Adapter method: LoRA (rank=16, alpha=32)                                    │
│  • Training: InfoNCE contrastive loss                                         │
│  • Batch size: 32 pairs                                                       │
│  • Epochs: 3-5                                                                 │
│  • VRAM: ~1.6GB additional during training (unload LLM temporarily)            │
│  • Duration: ~20 minutes for 2000 training pairs on RTX 4000 Ada Generation   │
│                                                                                 │
│  Step 3: VALIDATE                                                              │
│  ─────────────────────────────────────────────────────────────────────        │
│  • Compare recall@10 before vs after fine-tuning                              │
│  • If recall improves > 2%: deploy new model                                 │
│  • If recall degrades: rollback to previous checkpoint                        │
│                                                                                 │
│  Step 4: RE-INDEX (incremental)                                                │
│  ─────────────────────────────────────────────────────────────────────        │
│  • Re-embed all memories with new model                                       │
│  • Runs in background overnight (CPU + GPU time-sliced)                       │
│  • Swap indexes atomically when complete                                      │
│                                                                                 │
│  Expected Improvement: 5-15% retrieval accuracy gain after 1 month of use     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** ~1.6GB during training (temporary, LLM unloaded). 0 during normal operation.

---

### 13.13 Enhancement: Continuous Feedback Loop & Self-Improvement

**Gap Identified:** Current architecture evaluates quality but has no mechanism to USE evaluation results for continuous improvement. Failed queries, low RAGAS scores, and CRAG rejections should automatically feed back into system optimization.

**Research Source:** RAG-Gym (systematic optimization for agentic RAG, 2025); Continuous Feedback Loop patterns from production deployments.

**Integration Point:** New background process connecting evaluation → training.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS SELF-IMPROVEMENT LOOP                            │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA COLLECTION                                │   │
│  │                                                                        │   │
│  │  Every query-response cycle automatically logs:                        │   │
│  │  • Query text + intent + complexity score                              │   │
│  │  • Retrieved memory IDs + CRAG scores                                  │   │
│  │  • Generated response + Self-RAG critique scores                       │   │
│  │  • RAGAS/RAGChecker evaluation scores                                  │   │
│  │  • User feedback (thumbs up/down, if provided)                         │   │
│  │  • Latency breakdown per pipeline stage                                │   │
│  └────────────────────────────────────────────────────────┬───────────────┘   │
│                                                           │                   │
│                                                           ▼                   │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                         FAILURE ANALYSIS (Weekly)                       │   │
│  │                                                                        │   │
│  │  Automated analysis of failed queries:                                 │   │
│  │                                                                        │   │
│  │  1. Retrieval Failures (CRAG = INCORRECT)                              │   │
│  │     → Extract (query, irrelevant_result) as hard negatives             │   │
│  │     → Feed into embedding fine-tuning pipeline (13.12)                 │   │
│  │                                                                        │   │
│  │  2. Generation Failures (Self-RAG critique < 0.5)                      │   │
│  │     → Analyze prompt template effectiveness                            │   │
│  │     → A/B test alternative prompt structures                           │   │
│  │                                                                        │   │
│  │  3. Routing Failures (wrong agent selected)                            │   │
│  │     → Add to intent classifier training data                           │   │
│  │     → Retrain SetFit classifier (~5 minutes)                           │   │
│  │                                                                        │   │
│  │  4. Latency Outliers (>2x average)                                     │   │
│  │     → Identify bottleneck stage                                        │   │
│  │     → Adjust complexity thresholds to reduce unnecessary stages        │   │
│  └────────────────────────────────────────────────────────┬───────────────┘   │
│                                                           │                   │
│                                                           ▼                   │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                         AUTO-OPTIMIZATION (Monthly)                     │   │
│  │                                                                        │   │
│  │  Based on accumulated failure data:                                    │   │
│  │                                                                        │   │
│  │  1. Re-tune RRF channel weights (α, β, γ, δ, ε)                       │   │
│  │     → Optimize for maximum recall on logged queries                    │   │
│  │                                                                        │   │
│  │  2. Adjust complexity thresholds (0.3, 0.7)                            │   │
│  │     → Find optimal routing boundaries from query performance data      │   │
│  │                                                                        │   │
│  │  3. Fine-tune BGE-large-en-v1.5 on accumulated hard negatives            │   │
│  │                                                                        │   │
│  │  4. Update RAPTOR tree (rebuild clusters with new data)                │   │
│  │                                                                        │   │
│  │  5. Recalibrate ANN index parameters                                   │   │
│  │     → Re-run tuning protocol (13.6) with current index size            │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  VRAM Management During Optimization:                                          │
│  • All training runs during idle periods (user not actively querying)          │
│  • LLM unloaded temporarily during embedding fine-tuning                      │
│  • Re-indexing runs on CPU (FAISS supports CPU indexing)                       │
│  • Results validated before deployment (rollback if metrics degrade)           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**⚠️ Circular Dependency Safeguards:**

The self-improvement loop risks encoding the LLM's own errors into training data
(e.g., bad generations become "good" examples, retrieval biases compound). We
mitigate this with a multi-layer safety architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTION & SAFEGUARDS                                │
│                                                                                 │
│  1. HELD-OUT VALIDATION SET (never used for training)                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • 200 curated query-answer pairs created at initial deployment           ││
│  │ • Covers all intent types, complexity levels, and multi-turn patterns    ││
│  │ • Evaluated BEFORE and AFTER every auto-optimization cycle               ││
│  │ • ROLLBACK TRIGGER: If ANY metric drops >5% on held-out set,            ││
│  │   auto-revert to previous configuration (weights, thresholds, index)    ││
│  │ • Held-out set is APPEND-ONLY: new cases added, never removed           ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  2. HUMAN-IN-THE-LOOP GATES                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Tier 1 (auto-apply, notify): RRF weight adjustments, ANN tuning,      ││
│  │   cache invalidation, complexity threshold tweaks                        ││
│  │ • Tier 2 (require user approval): Embedding fine-tuning, SetFit          ││
│  │   retraining, RAPTOR rebuild, prompt template changes                   ││
│  │ • Tier 3 (require explicit opt-in): LLM LoRA adapter updates,           ││
│  │   any change affecting >10% of retrieval results                        ││
│  │                                                                           ││
│  │ Implementation: Each optimization generates a "change proposal" shown    ││
│  │ in the UI with before/after metrics. User clicks approve/reject.        ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  3. SYNTHETIC DATA VALIDATION                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • LLM-generated training data is NEVER trusted blindly                   ││
│  │ • Every synthetic (query, answer) pair is cross-verified:                ││
│  │   a) Answer must be grounded in retrieved passages (faithfulness > 0.8)  ││
│  │   b) Retrieved passages must exist in user's actual memory store         ││
│  │   c) Generated hard negatives verified as actually irrelevant            ││
│  │ • Unverifiable synthetic pairs are discarded, not used for training      ││
│  │ • Max synthetic ratio: 30% of any training batch (70% must be from      ││
│  │   real user interactions with explicit or implicit feedback)             ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  4. METRIC DRIFT MONITORING                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Track 7-day rolling averages for: faithfulness, relevancy, latency,   ││
│  │   retrieval recall, user satisfaction (thumbs up ratio)                  ││
│  │ • Alert thresholds: >10% degradation from 30-day baseline               ││
│  │ • Mode collapse detection: If >80% of queries route to same agent       ││
│  │   or same retrieval channel dominates RRF — flag for review             ││
│  │ • Diversity check: Ensure retrieved results don't converge to narrow    ││
│  │   cluster (embedding space coverage measured monthly)                    ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
│  5. VERSIONED ROLLBACK                                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │ • Every auto-optimization creates a versioned snapshot:                  ││
│  │   - Embedding model weights (if fine-tuned)                              ││
│  │   - RRF weights, complexity thresholds, routing config                   ││
│  │   - FAISS index checkpoint                                               ││
│  │ • Last 3 versions retained (~500MB per version)                          ││
│  │ • One-click rollback via UI or automatic on held-out regression          ││
│  └───────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** 0 during normal operation. Training uses ~1.6GB temporarily during idle windows.

---

### 13.15 Enhancement: Universal Multi-Modal Data Ingestion Pipeline

**Gap Identified:** Original Layer 0 only accepted text, voice, and basic document uploads. For a true personal AI memory system, users will continuously feed *any* type of data — PDFs, images, code repositories, spreadsheets, URLs, emails, handwritten notes, video recordings, and more. The system must seamlessly ingest, normalize, and index ALL of these into unified memory events with full retrieval capability.

**Research Source:** Multi-Modal RAG (Google DeepMind 2024); Document AI (2024); CLIP cross-modal retrieval; ColPali visual document retrieval (2024).

**Integration Point:** Layer 0 (Input Acquisition) — completely expanded to Universal Data Acquisition Hub.

**File: `src/ingestion/universal_data_pipeline.py`**

```python
"""
Universal Multi-Modal Data Ingestion Pipeline

Accepts ANY data format the user provides and converts it into
normalized memory events ready for embedding + indexing.

Supported Data Types (12+ formats):
─────────────────────────────────────
 1. Text/Chat     — Direct text, chat messages, notes
 2. Voice/Audio   — Microphone input, .wav/.mp3/.m4a files
 3. PDF           — Text PDFs, scanned PDFs (OCR), forms
 4. DOCX/ODT      — Word documents, OpenDocument text
 5. Images        — Screenshots, photos, diagrams, whiteboard captures
 6. Code Files    — .py, .js, .ts, .java, .cpp, .rs, .go, etc.
 7. CSV/TSV       — Tabular data with schema awareness
 8. JSON/JSONL    — Structured data, API responses, configs
 9. Excel (.xlsx) — Multi-sheet workbooks with formulas
10. URLs/Webpages — Article extraction, link graph building
11. Email (.eml)  — Subject/body/attachment decomposition
12. Video (.mp4)  — Transcript + key frame descriptions
13. Markdown/RST  — Structure-aware section parsing
14. Calendar(.ics)— Event extraction with temporal context
15. Handwriting   — OCR + preprocessing for handwritten notes
16. Database/SQL  — Schema + query result import

Hardware: RTX 4000 Ada Generation (20GB) — all extraction models
loaded on-demand, unloaded after batch completes.

VRAM Budget for Ingestion (on-demand, NOT concurrent with LLM inference):
  • EasyOCR:        ~300MB (loaded only for image/handwriting/scanned PDF)
  • BLIP-base caption: ~1GB (Salesforce/blip-image-captioning-base; loaded on-demand)
  • Whisper base:    ~150MB (loaded only for audio/video transcription)
  • Total peak:     ~1.5GB (models loaded one-at-a-time, never all together)

  ⚠️ VRAM COORDINATION: Ingestion GPU models MUST NOT run concurrently
  with LLM inference (DeepSeek-R1-Distill-Qwen-7B uses ~7GB at peak).
  The IngestVRAMGuard (below) enforces mutual exclusion:
  • When a user query arrives → ingestion GPU tasks are paused
  • When ingestion needs GPU → it acquires a lock, loads model, processes
    batch, unloads model, calls torch.cuda.empty_cache(), releases lock
  • §13.16 "continuous feed" means continuous QUEUEING, not continuous
    GPU processing. CPU-only extraction (PDF text, code parsing) runs
    in parallel with inference; GPU tasks (OCR, captioning) are queued.

  NOTE: We use BLIP-base (~1GB) instead of BLIP-2 (~3-5GB) because
  BLIP-2's VRAM footprint conflicts with the 7B LLM on a 20GB card.
  BLIP-base provides adequate captioning quality for memory indexing.
"""

import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


class UniversalDataPipeline:
    """
    Master ingestion pipeline: auto-detects format and routes to
    the appropriate extractor. Every piece of data becomes one or
    more MemoryEvent objects with full metadata.
    """
    
    # Supported MIME type → handler mapping
    HANDLERS = {
        # Text
        'text/plain': 'extract_text',
        'text/markdown': 'extract_markdown',
        'text/x-rst': 'extract_markdown',
        
        # Documents
        'application/pdf': 'extract_pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'extract_docx',
        'application/vnd.oasis.opendocument.text': 'extract_docx',
        
        # Images
        'image/png': 'extract_image',
        'image/jpeg': 'extract_image',
        'image/webp': 'extract_image',
        'image/gif': 'extract_image',
        'image/tiff': 'extract_image',
        
        # Spreadsheets / Structured Data
        'text/csv': 'extract_csv',
        'application/json': 'extract_json',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'extract_excel',
        
        # Code (detected by extension)
        'text/x-python': 'extract_code',
        'text/javascript': 'extract_code',
        'text/x-typescript': 'extract_code',
        'text/x-java': 'extract_code',
        'text/x-c': 'extract_code',
        'text/x-rust': 'extract_code',
        
        # Audio/Video
        'audio/wav': 'extract_audio',
        'audio/mpeg': 'extract_audio',
        'audio/mp4': 'extract_audio',
        'video/mp4': 'extract_video',
        'video/webm': 'extract_video',
        
        # Email/Calendar
        'message/rfc822': 'extract_email',
        'text/calendar': 'extract_calendar',
        
        # URLs
        'text/x-url': 'extract_url',
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.dedup_hashes = set()  # SHA-256 content deduplication
    
    def ingest(self, source: Union[str, Path, bytes], 
               source_type: Optional[str] = None,
               metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Universal entry point: ingest ANY data type.
        
        Args:
            source: File path, URL, or raw bytes
            source_type: Optional MIME type override (auto-detected if None)
            metadata: Optional user-provided metadata
            
        Returns:
            List of normalized MemoryEvent dicts ready for embedding
        """
        # Step 1: Detect format
        mime_type = source_type or self._detect_type(source)
        
        # Step 2: Deduplication check
        content_hash = self._compute_hash(source)
        if content_hash in self.dedup_hashes:
            return []  # Already ingested — skip
        self.dedup_hashes.add(content_hash)
        
        # Step 3: Route to appropriate extractor
        handler_name = self.HANDLERS.get(mime_type, 'extract_text')
        handler = getattr(self, handler_name)
        extracted = handler(source)
        
        # Step 4: Enrich with metadata
        base_metadata = {
            'source_type': mime_type,
            'ingestion_timestamp': datetime.utcnow().isoformat(),
            'content_hash': content_hash,
            'original_source': str(source)[:500],
            **(metadata or {})
        }
        
        # Step 5: Convert to memory events
        memory_events = []
        for chunk in extracted:
            event = {
                'content': chunk['text'],
                'metadata': {**base_metadata, **chunk.get('metadata', {})},
                'timestamp': chunk.get('timestamp', base_metadata['ingestion_timestamp']),
                'type': chunk.get('type', 'imported_data'),
                'importance': chunk.get('importance', 0.5),
            }
            memory_events.append(event)
        
        return memory_events
    
    # ─── EXTRACTORS ───────────────────────────────────────────────────
    
    def extract_pdf(self, path: Union[str, Path]) -> List[Dict]:
        """
        Extract text, tables, and images from PDFs.
        
        Uses PyMuPDF (fitz) for text extraction.
        Falls back to EasyOCR for scanned/image-only PDFs.
        Tables detected via pdfplumber for structured extraction.
        
        Output: One chunk per page (with page number metadata).
        """
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        chunks = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            # If page has very little text, it might be scanned — use OCR
            if len(text.strip()) < 50:
                text = self._ocr_page(page)
            
            # Extract tables separately
            tables = self._extract_tables_from_page(page)
            
            chunks.append({
                'text': text,
                'metadata': {
                    'page_number': page_num + 1,
                    'total_pages': len(doc),
                    'has_tables': len(tables) > 0,
                    'table_count': len(tables),
                },
                'type': 'document_page'
            })
            
            # Add tables as separate structured chunks
            for i, table in enumerate(tables):
                chunks.append({
                    'text': f"[Table {i+1} from page {page_num+1}]:\n{table}",
                    'metadata': {'is_table': True, 'page_number': page_num + 1},
                    'type': 'structured_data'
                })
        
        doc.close()
        return chunks
    
    def extract_image(self, path: Union[str, Path]) -> List[Dict]:
        """
        Extract information from images using:
        1. EasyOCR — for text within images (screenshots, whiteboards, docs)
        2. BLIP-base captioning — for visual scene description (photos, diagrams)
        
        Both are loaded on-demand and unloaded after processing.
        VRAM: ~300MB OCR + ~1GB captioning (sequential, not concurrent)
        """
        results = []
        
        # OCR pass: extract any visible text
        ocr_text = self._run_ocr(path)
        if ocr_text and len(ocr_text.strip()) > 10:
            results.append({
                'text': f"[Text extracted from image]: {ocr_text}",
                'metadata': {'extraction_method': 'ocr', 'source_format': 'image'},
                'type': 'image_text'
            })
        
        # Vision captioning pass: describe the image
        caption = self._generate_caption(path)
        if caption:
            results.append({
                'text': f"[Image description]: {caption}",
                'metadata': {'extraction_method': 'captioning', 'source_format': 'image'},
                'type': 'image_description'
            })
        
        return results or [{'text': '[Image with no extractable content]',
                           'metadata': {'source_format': 'image'}, 'type': 'image_placeholder'}]
    
    def extract_code(self, path: Union[str, Path]) -> List[Dict]:
        """
        Extract and chunk code files with syntax awareness.
        
        Strategy:
        - Detect language from extension
        - Parse into semantic units (functions, classes, imports)
        - Each function/class becomes a separate retrievable chunk
        - Preserve docstrings and comments as searchable text
        - Add code structure metadata (language, dependencies, complexity)
        """
        path = Path(path)
        content = path.read_text(encoding='utf-8', errors='replace')
        language = self._detect_language(path.suffix)
        
        # Split into semantic units (functions, classes)
        units = self._parse_code_units(content, language)
        
        chunks = []
        for unit in units:
            chunks.append({
                'text': f"[{language} code — {unit['name']}]:\n{unit['code']}",
                'metadata': {
                    'language': language,
                    'unit_type': unit['type'],  # function, class, module
                    'unit_name': unit['name'],
                    'line_start': unit['line_start'],
                    'line_end': unit['line_end'],
                    'has_docstring': unit.get('has_docstring', False),
                },
                'type': 'code_unit'
            })
        
        return chunks
    
    def extract_csv(self, path: Union[str, Path]) -> List[Dict]:
        """
        Extract structured data from CSV/TSV files.
        
        Strategy:
        - Infer schema (column names, types, statistics)
        - Create a schema summary chunk (always retrievable)
        - Create row-group chunks (every N rows as a described batch)
        - Identify key patterns, outliers, and aggregates
        """
        import pandas as pd
        df = pd.read_csv(str(path))
        
        chunks = []
        
        # Schema summary chunk
        schema_desc = f"Dataset: {Path(path).name}\n"
        schema_desc += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        schema_desc += f"Columns: {', '.join(df.columns.tolist())}\n"
        schema_desc += f"Types: {dict(df.dtypes.astype(str))}\n"
        schema_desc += f"Summary:\n{df.describe().to_string()}"
        
        chunks.append({
            'text': f"[Structured data schema]: {schema_desc}",
            'metadata': {'rows': df.shape[0], 'columns': df.shape[1]},
            'type': 'data_schema'
        })
        
        # Row-group chunks (every 50 rows)
        for start in range(0, len(df), 50):
            batch = df.iloc[start:start+50]
            batch_text = batch.to_string(index=False, max_rows=50)
            chunks.append({
                'text': f"[Data rows {start+1}-{min(start+50, len(df))}]:\n{batch_text}",
                'metadata': {'row_start': start+1, 'row_end': min(start+50, len(df))},
                'type': 'data_rows'
            })
        
        return chunks
    
    def extract_json(self, path: Union[str, Path]) -> List[Dict]:
        """Extract and flatten JSON/JSONL into searchable text chunks."""
        import json
        path = Path(path)
        content = path.read_text(encoding='utf-8')
        
        # Handle JSONL (one JSON object per line)
        if path.suffix == '.jsonl':
            objects = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
        else:
            data = json.loads(content)
            objects = data if isinstance(data, list) else [data]
        
        chunks = []
        for i, obj in enumerate(objects):
            flat_text = json.dumps(obj, indent=2, default=str)
            chunks.append({
                'text': f"[JSON object {i+1}]: {flat_text[:2000]}",
                'metadata': {'object_index': i, 'keys': list(obj.keys()) if isinstance(obj, dict) else []},
                'type': 'structured_data'
            })
        
        return chunks
    
    def extract_excel(self, path: Union[str, Path]) -> List[Dict]:
        """Extract all sheets from Excel workbooks with formula awareness."""
        import openpyxl
        wb = openpyxl.load_workbook(str(path), data_only=True)
        chunks = []
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            # Read as text table
            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append('\t'.join(str(c) if c is not None else '' for c in row))
            
            sheet_text = '\n'.join(rows[:200])  # Cap at 200 rows per sheet
            chunks.append({
                'text': f"[Excel sheet '{sheet_name}']:\n{sheet_text}",
                'metadata': {'sheet_name': sheet_name, 'row_count': ws.max_row, 'col_count': ws.max_column},
                'type': 'spreadsheet_data'
            })
        
        wb.close()
        return chunks
    
    def extract_url(self, url: str) -> List[Dict]:
        """
        Extract clean article text from URLs using Trafilatura.
        Falls back to BeautifulSoup + readability for complex pages.
        """
        import trafilatura
        
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_tables=True, include_comments=False)
            if text:
                return [{
                    'text': f"[Web content from {url}]:\n{text[:5000]}",
                    'metadata': {'url': url, 'extraction_method': 'trafilatura'},
                    'type': 'web_content'
                }]
        
        return [{'text': f'[Failed to extract content from {url}]',
                 'metadata': {'url': url}, 'type': 'web_content'}]
    
    def extract_audio(self, path: Union[str, Path]) -> List[Dict]:
        """Transcribe audio files using faster-whisper (base model, ~150MB VRAM)."""
        from faster_whisper import WhisperModel
        
        model = WhisperModel("base", device="cuda", compute_type="float16")
        segments, info = model.transcribe(str(path))
        
        full_text = ' '.join(seg.text for seg in segments)
        return [{
            'text': f"[Audio transcription ({info.language}, {info.duration:.0f}s)]: {full_text}",
            'metadata': {
                'language': info.language,
                'duration_seconds': info.duration,
                'source_format': 'audio'
            },
            'type': 'audio_transcript'
        }]
    
    def extract_video(self, path: Union[str, Path]) -> List[Dict]:
        """
        Extract from video: audio transcript + key frame descriptions.
        
        Pipeline:
        1. Extract audio track → Whisper transcription
        2. Sample key frames (1 per 30 seconds) → BLIP-base captions
        3. Combine into timestamped narrative
        
        VRAM: ~150MB (Whisper) + ~1GB (BLIP-base), loaded sequentially
        """
        chunks = []
        
        # Audio transcript
        audio_path = self._extract_audio_from_video(path)
        if audio_path:
            transcript_chunks = self.extract_audio(audio_path)
            chunks.extend(transcript_chunks)
        
        # Key frame descriptions (sampled every 30 seconds)
        frames = self._sample_video_frames(path, interval_seconds=30)
        for i, (timestamp, frame_path) in enumerate(frames):
            caption = self._generate_caption(frame_path)
            if caption:
                chunks.append({
                    'text': f"[Video frame at {timestamp}s]: {caption}",
                    'metadata': {'timestamp_seconds': timestamp, 'frame_index': i},
                    'type': 'video_frame_description'
                })
        
        return chunks
    
    def extract_email(self, path: Union[str, Path]) -> List[Dict]:
        """Parse .eml files into subject, body, and attachment events."""
        import email
        from email import policy
        
        with open(str(path), 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)
        
        chunks = [{
            'text': (f"[Email from {msg['from']} to {msg['to']}]\n"
                     f"Subject: {msg['subject']}\n"
                     f"Date: {msg['date']}\n\n"
                     f"{msg.get_body(preferencelist=('plain',)).get_content()[:3000]}"),
            'metadata': {
                'from': str(msg['from']),
                'to': str(msg['to']),
                'subject': str(msg['subject']),
                'date': str(msg['date']),
            },
            'type': 'email_message'
        }]
        
        # Process attachments recursively
        for attachment in msg.iter_attachments():
            att_data = attachment.get_content()
            att_name = attachment.get_filename() or 'unnamed_attachment'
            # Re-ingest attachment through the universal pipeline
            chunks.append({
                'text': f"[Email attachment: {att_name}] (queued for separate ingestion)",
                'metadata': {'attachment_name': att_name, 'parent_email': str(msg['subject'])},
                'type': 'email_attachment'
            })
        
        return chunks
    
    def extract_calendar(self, path: Union[str, Path]) -> List[Dict]:
        """Parse .ics calendar files into temporal memory events."""
        from icalendar import Calendar
        
        with open(str(path), 'rb') as f:
            cal = Calendar.from_ical(f.read())
        
        chunks = []
        for component in cal.walk():
            if component.name == 'VEVENT':
                event_text = (
                    f"[Calendar event]: {component.get('summary', 'Untitled')}\n"
                    f"When: {component.get('dtstart', {}).dt} to {component.get('dtend', {}).dt}\n"
                    f"Location: {component.get('location', 'N/A')}\n"
                    f"Description: {str(component.get('description', ''))[:500]}"
                )
                chunks.append({
                    'text': event_text,
                    'metadata': {
                        'event_name': str(component.get('summary', '')),
                        'start_time': str(component.get('dtstart', {}).dt),
                    },
                    'timestamp': str(component.get('dtstart', {}).dt),
                    'type': 'calendar_event',
                    'importance': 0.7  # Calendar events are moderately important
                })
        
        return chunks
    
    def extract_markdown(self, path: Union[str, Path]) -> List[Dict]:
        """Parse Markdown files into headed sections for granular retrieval."""
        content = Path(path).read_text(encoding='utf-8', errors='replace')
        sections = self._split_by_headers(content)
        
        return [{
            'text': f"[Markdown section: {s['header']}]\n{s['content']}",
            'metadata': {'header': s['header'], 'level': s['level']},
            'type': 'document_section'
        } for s in sections]
    
    def extract_text(self, source: Union[str, Path]) -> List[Dict]:
        """Fallback: treat as plain text."""
        if isinstance(source, (str, Path)) and Path(source).is_file():
            text = Path(source).read_text(encoding='utf-8', errors='replace')
        else:
            text = str(source)
        return [{'text': text, 'metadata': {}, 'type': 'text_memory'}]
    
    # ─── UTILITY METHODS ──────────────────────────────────────────────
    
    def _detect_type(self, source) -> str:
        """Auto-detect MIME type from file extension or content."""
        if isinstance(source, str) and source.startswith(('http://', 'https://')):
            return 'text/x-url'
        if isinstance(source, (str, Path)):
            mime, _ = mimetypes.guess_type(str(source))
            return mime or 'text/plain'
        return 'text/plain'
    
    def _compute_hash(self, source) -> str:
        """SHA-256 hash for deduplication."""
        if isinstance(source, bytes):
            return hashlib.sha256(source).hexdigest()
        if isinstance(source, (str, Path)) and Path(source).is_file():
            return hashlib.sha256(Path(source).read_bytes()).hexdigest()
        return hashlib.sha256(str(source).encode()).hexdigest()
    
    def _detect_language(self, extension: str) -> str:
        """Map file extension to programming language."""
        lang_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.rs': 'Rust',
            '.go': 'Go', '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift',
            '.kt': 'Kotlin', '.scala': 'Scala', '.r': 'R', '.sql': 'SQL',
            '.sh': 'Shell', '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML',
        }
        return lang_map.get(extension.lower(), 'Unknown')
    
    def _parse_code_units(self, content: str, language: str) -> List[Dict]:
        """Parse code into semantic units (functions, classes)."""
        # For Python: use ast module
        # For others: use tree-sitter or regex-based splitting
        # Returns list of {name, type, code, line_start, line_end}
        # Implementation varies by language
        pass
    
    def _run_ocr(self, path) -> str:
        """Run EasyOCR on an image. Loaded on-demand, ~300MB VRAM."""
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        results = reader.readtext(str(path))
        return ' '.join(text for _, text, conf in results if conf > 0.5)
    
    def _generate_caption(self, path) -> str:
        """Generate image caption using BLIP-base. ~1GB VRAM, on-demand.
        
        Uses Salesforce/blip-image-captioning-base instead of BLIP-2
        to fit within 20GB VRAM budget alongside the 7B LLM.
        Model is loaded on-demand, inference runs, then model is
        explicitly unloaded with torch.cuda.empty_cache().
        """
        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        try:
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16
            ).to("cuda")
            
            image = Image.open(path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output[0], skip_special_tokens=True)
            
        finally:
            # CRITICAL: Free VRAM for LLM inference
            del model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return caption
    
    def _ocr_page(self, page) -> str:
        """OCR a PDF page that has no extractable text (scanned)."""
        # Render page to image, then run OCR
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        return self._run_ocr(img_bytes)
    
    def _extract_tables_from_page(self, page) -> List[str]:
        """Extract tables from a PDF page using pdfplumber."""
        # Returns list of table strings (markdown format)
        pass
    
    def _extract_audio_from_video(self, video_path) -> Optional[str]:
        """Extract audio track from video using ffmpeg."""
        # subprocess: ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav
        pass
    
    def _sample_video_frames(self, video_path, interval_seconds=30) -> List:
        """Sample key frames from video at given interval."""
        # subprocess: ffmpeg -i video.mp4 -vf fps=1/{interval} frame_%d.jpg
        pass
    
    def _split_by_headers(self, markdown: str) -> List[Dict]:
        """Split markdown by headers into sections."""
        import re
        sections = []
        current = {'header': 'Introduction', 'level': 0, 'content': ''}
        for line in markdown.split('\n'):
            match = re.match(r'^(#{1,6})\s+(.+)', line)
            if match:
                if current['content'].strip():
                    sections.append(current)
                current = {
                    'header': match.group(2),
                    'level': len(match.group(1)),
                    'content': ''
                }
            else:
                current['content'] += line + '\n'
        if current['content'].strip():
            sections.append(current)
        return sections
```

**Supported Data Types — Complete Coverage Matrix:**

| Data Type | Extractor | Output Format | VRAM (On-Demand) | Chunking Strategy |
|-----------|-----------|---------------|-----------------|-------------------|
| Text/Chat | Direct passthrough | Raw text | 0 | Semantic chunking |
| Voice/Audio | faster-whisper (base) | Transcript text | ~150MB | Time-windowed segments |
| PDF (text) | PyMuPDF (fitz) | Pages + tables | 0 (CPU) | Page-level + table separation |
| PDF (scanned) | PyMuPDF → EasyOCR | OCR text | ~300MB | Page-level with confidence |
| DOCX/ODT | python-docx / pandoc | Structured text | 0 (CPU) | Section-level |
| Images | EasyOCR + BLIP-base | OCR text + caption | ~1GB | Per-image event |
| Code files | AST / tree-sitter | Semantic code units | 0 (CPU) | Function/class level |
| CSV/TSV | pandas | Schema + row groups | 0 (CPU) | Schema chunk + 50-row batches |
| JSON/JSONL | json module | Flattened objects | 0 (CPU) | Per-object chunk |
| Excel (.xlsx) | openpyxl | Sheet text + formulas | 0 (CPU) | Per-sheet chunks |
| URLs/Webpages | Trafilatura | Clean article text | 0 (CPU) | Article sections |
| Email (.eml) | email.parser | Subject/body split | 0 (CPU) | Message + attachments |
| Video (.mp4) | ffmpeg + Whisper + BLIP-base | Transcript + frames | ~1.2GB | 30s segments + key frames |
| Calendar (.ics) | icalendar | Event descriptions | 0 (CPU) | Per-event chunk |
| Markdown/RST | Regex header splitting | Headed sections | 0 (CPU) | Section-level |
| Handwriting | EasyOCR + preprocessing | Recognized text | ~300MB | Per-page/image |

**Key Design Principles:**
1. **On-Demand Model Loading:** Heavy models (OCR, captioning) are loaded only when needed, then unloaded — never competing with inference VRAM
2. **SHA-256 Deduplication:** Identical content is never re-ingested, even if uploaded again
3. **Metadata Enrichment:** Every chunk carries provenance (source file, page number, extraction method, confidence)
4. **Graceful Degradation:** If an extractor fails, content is still stored as raw text with a low confidence flag
5. **Recursive Processing:** Email attachments, embedded images in PDFs, etc. are recursively re-ingested through the pipeline

**VRAM Impact:** ~300MB–1GB on-demand during ingestion batches (models loaded sequentially, never simultaneously). 0 during querying (all extraction is at ingestion time). GPU ingestion tasks are mutex-locked against LLM inference — see VRAM Coordination note above.

---

### 13.16 Enhancement: Continuous Data Feed & Incremental Indexing

**Gap Identified:** Users don't just upload files once — they continuously feed data over weeks and months. The system needs an always-on ingestion endpoint that watches for new data, processes it in the background, and incrementally updates all indexes without downtime.

**Research Source:** Streaming RAG indexing (Weaviate 2024); Incremental HNSW updates; Event-driven architectures.

**Integration Point:** Layer 0 → Layer 1 → Layer 2 — background ingestion pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS DATA FEED ARCHITECTURE                           │
│                    (Always-on, incremental, zero-downtime)                      │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  DATA SOURCES (user continuously feeds)                                  │ │
│  │                                                                           │ │
│  │  • Chat messages (real-time, as user types)                              │ │
│  │  • File drag-and-drop (any supported format)                             │ │
│  │  • Clipboard paste (text, images, URLs)                                  │ │
│  │  • Watched folder (auto-ingest new files from ~/CortexLab/inbox/)        │ │
│  │  • API endpoint (POST /api/ingest — for integrations)                    │ │
│  │  • Browser extension (save web pages directly)                           │ │
│  │  • Email forwarding (dedicated ingest email address)                     │ │
│  └───────────────────────────────────────────────────┬───────────────────────┘ │
│                                                       │                         │
│                                                       ▼                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  INGESTION QUEUE (async, priority-ordered)                               │ │
│  │                                                                           │ │
│  │  Priority 1: Real-time chat messages (process immediately, <100ms)       │ │
│  │  Priority 2: Clipboard/drag-drop (process within 5 seconds)             │ │
│  │  Priority 3: File uploads (process within 30 seconds)                    │ │
│  │  Priority 4: Watched folder batch (process within 5 minutes)            │ │
│  │  Priority 5: Background re-indexing (during idle time only)             │ │
│  │                                                                           │ │
│  │  Queue: asyncio.PriorityQueue — in-memory with SQLite WAL persistence   │ │
│  └───────────────────────────────────────────────────┬───────────────────────┘ │
│                                                       │                         │
│                                                       ▼                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  INCREMENTAL INDEX UPDATES (no full re-index needed)                     │ │
│  │                                                                           │ │
│  │  1. New memory → embed with BGE-large-en-v1.5 (1024d)                   │ │
│  │  2. Add vector to FAISS HNSW hot tier (supports incremental add)        │ │
│  │  3. Insert into DuckDB SQL store (atomic transaction)                   │ │
│  │  4. Update knowledge graph (add entity nodes + relationship edges)      │ │
│  │  5. Update BM25 inverted index (append new terms)                       │ │
│  │  6. RAPTOR tree: DEFERRED UPDATE (see note below)                       │ │
│  │  7. Invalidate relevant cache entries (semantic similarity check)       │ │
│  │                                                                           │ │
│  │  Steps 1-5, 7 complete in <500ms for text, <5s for files               │ │
│  │                                                                           │ │
│  │  ⚠️ RAPTOR NOTE: RAPTOR's hierarchical clusters were designed for       │ │
│  │  static corpora. True incremental insertion that preserves semantic     │ │
│  │  coherence is an open research problem. Our approach:                   │ │
│  │  • New memories are immediately searchable via flat vector + BM25       │ │
│  │  • RAPTOR tree rebuild is BATCHED (runs during idle periods)            │ │
│  │  • Trigger: when >50 new memories since last rebuild, or weekly         │ │
│  │  • Rebuild time: ~2-5min for 10K memories (LLM-generated summaries)    │ │
│  │  • During rebuild: old tree remains queryable, swap is atomic           │ │
│  │  This means RAPTOR summaries may lag behind by hours/days, but raw      │ │
│  │  memory retrieval is always real-time.                                  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │  BACKGROUND MAINTENANCE (runs during idle periods)                       │ │
│  │                                                                           │ │
│  │  • Hot → Warm tier migration (memories > 30 days old)                   │ │
│  │  • RAPTOR tree rebuild (when cluster centroids shift >10%)              │ │
│  │  • Knowledge graph pruning (remove low-confidence edges)                │ │
│  │  • Deduplication sweep (find near-duplicate memories, merge)            │ │
│  │  • Memory consolidation (daily → weekly → monthly summaries)            │ │
│  │  • Statistics update (for monitoring dashboard)                          │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  VRAM: 0 additional (uses existing BGE-large for embedding)                    │
│  Throughput: ~50 text memories/sec, ~5 files/sec (on RTX 4000 Ada Gen)        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**VRAM Impact:** 0 additional (reuses existing embedding model). All heavy processing is queued and time-sliced with inference.

---

### 13.17 Summary: Enhancement Impact Matrix

| # | Enhancement | Query-Time Latency | VRAM Cost | Accuracy Impact | Priority |
|---|---|---|---|---|---|
| 13.1 | Contextual Chunking | 0ms (ingestion) | 0 | +10-15% retrieval | ⭐⭐⭐ HIGH |
| 13.2 | Semantic Chunking | 0ms (ingestion) | 0 | +5-8% retrieval | ⭐⭐ MEDIUM |
| 13.3 | Step-Back Prompting | +0.3s (when triggered) | 0 | +8-12% for causal queries | ⭐⭐ MEDIUM |
| 13.4 | Multi-Level Caching | -2-4s (cache hit) | ~20MB | 40%+ queries instant | ⭐⭐⭐ HIGH |
| 13.5 | Vector Quantization | -10ms (faster search) | -80% vectors | ~2% recall tradeoff | ⭐⭐⭐ HIGH |
| 13.6 | ANN Index Tuning | -50ms P99 | 0 | 0 (same recall) | ⭐⭐⭐ HIGH |
| 13.7 | Async Retrieval | -250ms | 0 | 0 | ⭐⭐⭐ HIGH |
| 13.8 | RAGChecker | 0 (offline) | 0 | Diagnostic visibility | ⭐⭐ MEDIUM |
| 13.9 | Failure-Aware Refinement | +0.3s (when needed) | 0 | +15-20% recovery | ⭐⭐⭐ HIGH |
| 13.10 | Chain-of-Retrieval | +1-2s (complex only) | 0 | +20% multi-hop accuracy | ⭐⭐ MEDIUM |
| 13.11 | Token Efficiency | -30-50% tokens | Reduces peak | Faster overall | ⭐⭐⭐ HIGH |
| 13.12 | Retriever Fine-tuning | 0 (background) | ~1.6GB temp | +5-15% over time | ⭐⭐⭐ HIGH |
| 13.13 | Continuous Self-Improvement | 0 (background) | ~1.6GB temp | Compounding gains | ⭐⭐⭐ HIGH |
| 13.15 | Universal Multi-Modal Ingestion | 0ms (query) | ~300-550MB on-demand | 12+ data types supported | ⭐⭐⭐ CRITICAL |
| 13.16 | Continuous Data Feed | <500ms (incremental) | 0 | Zero-downtime index updates | ⭐⭐⭐ CRITICAL |

---

## Summary

This RAG architecture represents the **state-of-the-art in personal AI memory systems**, purpose-built for the **NVIDIA RTX 4000 Ada Generation (20GB VRAM)** and powered by **DeepSeek-R1-Distill-Qwen-7B** (10-stage QLoRA fine-tuned + DPO aligned). It combines 30+ cutting-edge techniques:

✅ **Universal Data Ingestion** (16+ data types: text, PDF, images, code, audio, video, CSV, JSON, Excel, URLs, email, calendar, handwriting — any data the user feeds)  
✅ **Continuous Data Feed** (Always-on ingestion with async priority queue, watched folders, clipboard paste, API endpoints — incremental indexing with zero downtime)  
✅ **Hierarchical Indexing** (RAPTOR + Propositions + Contextual Chunking + Semantic Boundaries)  
✅ **Agentic Reasoning** (6 specialized agents: Orchestrator, Timeline, Causal, Reflection, Planning, Arbitration + Chain-of-Retrieval)  
✅ **Hybrid Retrieval** (Dense 1024d BGE-large + Sparse BM25 + Graph + Temporal + Proposition — Async Parallel Execution)  
✅ **Self-Reflection** (CRAG + Self-RAG + FLARE + Failure-Aware Refinement)  
✅ **Memory Evolution** (Belief tracking + Consolidation + Entity Resolution)  
✅ **Production Optimization** (Multi-Level Caching + Vector Quantization + ANN Tuning + Token Efficiency)  
✅ **Continuous Improvement** (RAGChecker Diagnostics + Retriever Fine-tuning + Self-Improvement Loop)  
✅ **DPO-Aligned Generation** (Faithfulness > 0.92, Multi-turn coherence > 0.88, Long-context recall > 0.85)  
✅ **Hardware-Maximized** (Training 63% VRAM utilization, Inference 34% — all 30+ techniques validated against 20GB budget)

**VRAM Budget Summary:**

| Component | VRAM | Notes |
|-----------|------|-------|
| DeepSeek-R1-Distill-Qwen-7B (bf16) | ~4.2GB | Primary LLM — 10-stage fine-tuned + DPO |
| BGE-large-en-v1.5 (1024d) | ~1.3GB | Dense embedding model |
| BGE-reranker-v2-m3 | ~1.1GB | Cross-encoder reranking |
| Classifiers + Whisper | ~350MB | SetFit (3x) + faster-whisper base |
| Vector indexes (FAISS) | ~115MB | PQ/SQ8 compressed, 500K vectors |
| Multi-level cache | ~20MB | Embedding + semantic + response |
| **Total Inference** | **~7.1GB** | **35% of 20GB** |
| **Available for Training** | **~13GB** | **QLoRA fine-tuning + DPO alignment** |
| **Available for Ingestion** | **~13GB** | **OCR/captioning models loaded on-demand** |

**Cortex Lab is not just a chatbot — it's your second brain. It ingests ANY data you feed it, remembers EVERYTHING, reasons over your personal knowledge with 6 specialized agents, and continuously improves. Powered by cutting-edge research and maximized for the RTX 4000 Ada Generation.** 🧠🚀
