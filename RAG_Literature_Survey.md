# Comprehensive Literature Survey: Retrieval-Augmented Generation (RAG) Systems

> **Last Updated:** February 2025
> **Total Resources:** 100+ curated papers, tools, datasets, and implementations
> **Scope:** Advanced RAG topics only — excludes general deep-learning surveys

---

## Table of Contents

1. [Must-Read Landmark Papers](#1-must-read-landmark-papers)
2. [Indexing &amp; Multi-Representation Indexing](#2-indexing--multi-representation-indexing)
3. [Retrieval Techniques](#3-retrieval-techniques)
4. [Query Translation, Fusion &amp; Decomposition](#4-query-translation-fusion--decomposition)
5. [Routing &amp; Query Routing](#5-routing--query-routing)
6. [Text → Metadata Filtering](#6-text--metadata-filtering)
7. [Active, Adaptive &amp; Corrective RAG](#7-active-adaptive--corrective-rag)
8. [Evaluation Frameworks &amp; Benchmarks](#8-evaluation-frameworks--benchmarks)
9. [Datasets for RAG Evaluation](#9-datasets-for-rag-evaluation)
10. [Scalability &amp; Vector Databases](#10-scalability--vector-databases)
11. [Production Deployments](#11-production-deployments)
12. [Conference Talks &amp; Videos](#12-conference-talks--videos)
13. [Framework Implementations &amp; Code Repositories](#13-framework-implementations--code-repositories)
14. [Annotated Reading Path](#14-annotated-reading-path)

---

## 1. Must-Read Landmark Papers

### ⭐⭐⭐ Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Patrick Lewis et al. (Facebook AI Research)
**Summary:** The foundational RAG paper introducing the paradigm of combining parametric (model weights) and non-parametric (retrieved documents) memory for knowledge-intensive tasks. Establishes RAG-sequence and RAG-token approaches.
**PDF:** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | [NeurIPS 2020 PDF](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
**Venue:** NeurIPS 2020
**Code:** [huggingface/rag](https://github.com/huggingface/transformers/tree/main/src/transformers/models/rag)
**Tags:** `foundation`, `retrieval-generation`, `Meta`

---

### ⭐⭐⭐ REALM: Retrieval-Augmented Language Model Pre-Training

**Authors:** Kelvin Guu et al. (Google Research)
**Summary:** Pioneering work integrating retrieval into language model pre-training with a latent knowledge retriever. First to show retrieval-augmented pre-training significantly improves open-domain QA.
**PDF:** [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) | [ICML 2020 PDF](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)
**Venue:** ICML 2020
**Code:** [google-research/language/realm](https://github.com/google-research/language/blob/master/language/realm/README.md)
**Tags:** `pre-training`, `retrieval`, `Google`

---

### ⭐⭐⭐ Dense Passage Retrieval for Open-Domain Question Answering

**Authors:** Vladimir Karpukhin et al. (Facebook AI Research)
**Summary:** Introduces DPR (Dense Passage Retriever), demonstrating that dense embeddings alone can outperform BM25 for passage retrieval. Foundation for modern dense retrieval in RAG systems.
**PDF:** [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) | [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.550.pdf)
**Venue:** EMNLP 2020
**Code:** [facebookresearch/DPR](https://github.com/facebookresearch/DPR)
**Tags:** `dense-retrieval`, `DPR`, `Meta`

---

### ⭐⭐⭐ Atlas: Few-shot Learning with Retrieval Augmented Language Models

**Authors:** Gautier Izacard et al. (DeepMind/Google)
**Summary:** Demonstrates that retrieval-augmented language models can match or exceed 540B parameter models with 50x fewer parameters. Introduces joint pre-training of retriever and language model.
**PDF:** [arXiv:2208.03299](https://arxiv.org/abs/2208.03299) | [JMLR 2023](http://www.jmlr.org/papers/volume24/23-0037/23-0037.pdf)
**Venue:** JMLR 2023 (originally arXiv 2022)
**Code:** N/A (DeepMind internal)
**Tags:** `few-shot`, `pre-training`, `DeepMind`

---

### ⭐⭐ Retrieval-Augmented Generation for Large Language Models: A Survey

**Authors:** Yunfan Gao et al.
**Summary:** Comprehensive survey covering Naive RAG, Advanced RAG, and Modular RAG paradigms. Essential reading for understanding the evolution and taxonomy of RAG systems.
**PDF:** [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
**Venue:** arXiv 2023 (Highly Cited)
**Code:** N/A
**Tags:** `survey`, `taxonomy`, `advanced-rag`

---

### ⭐⭐ A Comprehensive Survey of Retrieval-Augmented Generation (RAG)

**Authors:** Huimin Xu et al.
**Summary:** Extensive survey with 235+ citations covering RAG evolution, current landscape, and future directions. Provides detailed analysis of retrieval-generation integration.
**PDF:** [arXiv:2410.12837](https://arxiv.org/abs/2410.12837)
**Venue:** arXiv 2024
**Code:** N/A
**Tags:** `survey`, `evolution`, `future-directions`

---

## 2. Indexing & Multi-Representation Indexing

### ⭐⭐⭐ RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**Authors:** Parth Sarthi et al. (Stanford)
**Summary:** Introduces tree-structured indexing that recursively clusters and summarizes documents, enabling retrieval at multiple abstraction levels. Significant improvement on QA tasks requiring reasoning across document hierarchies.
**PDF:** [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) | [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf)
**Venue:** ICLR 2024 (376 citations)
**Code:** [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor)
**Tags:** `tree-indexing`, `hierarchical`, `summarization`

---

### ⭐⭐ Proposition-Based Retrieval

**Authors:** (Various implementations following Chen et al.)
**Summary:** Retrieval by atomic propositions rather than passages or sentences. Demonstrates improved performance on downstream QA tasks by retrieving at the granularity of individual facts.
**PDF:** [arXiv:2312.06648](https://arxiv.org/pdf/2312.06648)
**Venue:** EMNLP 2024 Findings
**Code:** Available in RAG technique repositories
**Tags:** `proposition`, `granularity`, `fact-retrieval`

---

### ⭐⭐ TreeRAG: Unleashing the Power of Hierarchical Storage

**Authors:** (ACL 2025)
**Summary:** Focuses on connectivity between chunks and preservation of hierarchical contextual information. Builds on RAPTOR concepts for improved document understanding.
**PDF:** [ACL 2025 PDF](https://aclanthology.org/2025.findings-acl.20.pdf)
**Venue:** ACL 2025 Findings (6 citations)
**Code:** Emerging implementations
**Tags:** `hierarchical`, `tree-structure`, `context-preservation`

---

## 3. Retrieval Techniques

### ⭐⭐⭐ ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction

**Authors:** Omar Khattab et al. (Stanford)
**Summary:** Introduces late interaction mechanism where query and document tokens interact via MaxSim operations. Achieves both high effectiveness (outperforming BERT rerankers) and efficiency (indexable representations).
**PDF:** [arXiv:2004.12832](https://arxiv.org/abs/2004.12832) | [SIGIR 2020 PDF](https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf)
**Venue:** SIGIR 2020 (2197 citations)
**Code:** [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
**Tags:** `late-interaction`, `neural-retrieval`, `efficient`

---

### ⭐⭐⭐ ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction

**Authors:** Keshav Santhanam et al. (Stanford)
**Summary:** Improves ColBERT with aggressive residual compression and denoised supervision. Reduces storage by 6-10x while maintaining effectiveness. Foundation for modern multi-vector retrieval.
**PDF:** [arXiv:2112.01488](https://arxiv.org/abs/2112.01488) | [NAACL 2022](https://aclanthology.org/2022.naacl-main.272)
**Venue:** NAACL 2022 (749 citations)
**Code:** [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
**Tags:** `compression`, `multi-vector`, `efficient`

---

### ⭐⭐⭐ HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels

**Authors:** Luyu Gao et al. (CMU/Google)
**Summary:** Hypothetical Document Embeddings generates a hypothetical answer to the query, then retrieves documents similar to this hypothetical document. Enables zero-shot retrieval without training data.
**PDF:** [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | [ACL 2023](https://aclanthology.org/2023.acl-long.99)
**Venue:** ACL 2023 (673 citations)
**Code:** [NirDiamant/RAG_Techniques/HyDE](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb)
**Tags:** `zero-shot`, `hypothetical-document`, `query-expansion`

---

### ⭐⭐ SPLADE: Sparse Lexical and Expansion Model for Information Retrieval

**Authors:** Thibault Formal et al. (Naver Labs)
**Summary:** Sparse neural retrieval combining sparse representations with learned term expansion. Achieves state-of-the-art effectiveness while maintaining exact matching capabilities.
**PDF:** [arXiv:2109.10086](https://arxiv.org/abs/2109.10086) | [TOIS 2024](https://dl.acm.org/doi/10.1145/3634912)
**Venue:** SIGIR 2021 / TOIS 2024
**Code:** [naver/splade](https://github.com/naver/splade)
**Tags:** `sparse-retrieval`, `term-expansion`, `neural-IR`

---

### ⭐⭐ BGE & MTEB Embedding Benchmarks

**Authors:** BAAI / Hugging Face
**Summary:** BGE (BAAI General Embedding) models achieve state-of-the-art on MTEB benchmark. Supports dense, sparse, and multi-vector retrieval. Essential for selecting embedding models in RAG.
**PDF:** [MTEB Paper](https://arxiv.org/html/2406.01607v1)
**Venue:** arXiv 2024
**Code:** [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
**Tags:** `embeddings`, `benchmark`, `BGE`

---

## 4. Query Translation, Fusion & Decomposition

### ⭐⭐⭐ RAG-Fusion: A New Take on Retrieval-Augmented Generation

**Authors:** Zackary Rackauckas
**Summary:** Combines multi-query generation with Reciprocal Rank Fusion (RRF) to aggregate and rerank results. Demonstrates improved retrieval coverage and relevance.
**PDF:** [arXiv:2402.03367](https://arxiv.org/abs/2402.03367)
**Venue:** arXiv 2024 (146 citations)
**Code:** Available in LangChain/LlamaIndex
**Tags:** `multi-query`, `RRF`, `fusion`

---

### ⭐⭐ Multi-Query Generation & Query Decomposition

**Summary:** Techniques for breaking complex queries into multiple simpler queries or sub-questions. Improves retrieval coverage for complex, multi-part questions.
**Implementation:** [LangChain Query Transformations](https://blog.langchain.com/query-transformations)
**Tags:** `query-decomposition`, `multi-query`, `complex-queries`

---

### ⭐⭐ Step-Back Prompting

**Summary:** Generates a higher-level abstraction (step-back question) before retrieving, helping with complex reasoning scenarios where direct retrieval may miss relevant context.
**Tags:** `abstraction`, `reasoning`, `prompting`

---

## 5. Routing & Query Routing

### ⭐⭐ RAGRouter: Learning to Route Queries to Multiple Retrieval-Augmented Language Models

**Authors:** Various
**Summary:** Proposes RAG-aware routing design leveraging document embeddings and RAG capability embeddings with contrastive learning. Routes queries to optimal retrieval strategies or models.
**PDF:** [arXiv:2505.23052](https://arxiv.org/abs/2505.23052)
**Venue:** arXiv 2025
**Code:** Emerging implementations
**Tags:** `routing`, `model-selection`, `optimization`

---

### ⭐⭐ Route Before Retrieve: Activating Latent Routing Abilities

**Authors:** (OpenReview)
**Summary:** Introduces Pre-Route framework for choosing between RAG and Long Context approaches based on query characteristics.
**PDF:** [OpenReview](https://openreview.net/forum?id=N1E7rFZJGH)
**Venue:** OpenReview 2024
**Tags:** `routing`, `long-context`, `adaptive`

---

### ⭐⭐ RouteRAG: Adaptive Routing in RAG Systems

**Summary:** Dynamically routes queries in RAG frameworks by selecting optimal retrieval pathways for efficient and accurate question answering.
**Tags:** `adaptive`, `routing`, `pathway-selection`

---

## 6. Text → Metadata Filtering

### ⭐⭐ Two-Step RAG for Metadata Filtering

**Summary:** Addresses poor retrieval accuracy when vague prompts or metadata mismatches occur. Uses structured metadata to narrow search results before semantic retrieval.
**PDF:** [ResearchGate](https://www.researchgate.net/publication/397332230_Two-Step_RAG_for_Metadata_Filtering_and_Statistical_LLM_Evaluation)
**Tags:** `metadata`, `filtering`, `structured-retrieval`

---

### ⭐⭐ Graph-Based Metadata Filtering

**Summary:** Optimizes vector retrieval by leveraging structured data to narrow search results. Essential for enterprise RAG with rich document metadata.
**Tags:** `graph`, `metadata`, `enterprise`

---

### ⭐⭐ AMAQA: A Metadata-based QA Dataset for RAG Systems

**Summary:** Integrates metadata with structured QA dataset for holistic evaluation considering text-metadata interplay.
**PDF:** [arXiv:2505.13557](https://arxiv.org/html/2505.13557v2)
**Venue:** arXiv 2025
**Tags:** `dataset`, `metadata`, `evaluation`

---

## 7. Active, Adaptive & Corrective RAG

### ⭐⭐⭐ Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Authors:** Akari Asai et al. (University of Washington/Allen AI)
**Summary:** Introduces self-reflective retrieval-augmented generation where the model learns to retrieve, generate, and critique its own outputs. Achieves significant improvements in factuality and quality.
**PDF:** [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)
**Venue:** ICLR 2024
**Code:** [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)
**Tags:** `self-reflection`, `adaptive`, `critique`

---

### ⭐⭐⭐ CRAG: Corrective Retrieval Augmented Generation

**Authors:** (Various)
**Summary:** Improves RAG robustness by evaluating retrieved document quality and deciding whether to use, discard, or correct retrieved content before generation.
**PDF:** [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) | [OpenReview PDF](https://openreview.net/pdf?id=JnWJbrnaUE)
**Venue:** arXiv 2024
**Code:** [HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)
**Tags:** `corrective`, `quality-evaluation`, `robustness`

---

### ⭐⭐⭐ FLARE: Forward-Looking Active Retrieval Augmented Generation

**Authors:** Zhengbao Jiang et al.
**Summary:** Actively decides when and what to retrieve during generation using prediction of upcoming sentences. Retrieves when low-confidence tokens are detected in provisional generation.
**PDF:** [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) | [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.495)
**Venue:** EMNLP 2023 (1070 citations)
**Code:** [jzbjyb/FLARE](https://github.com/jzbjyb/FLARE)
**Tags:** `active-retrieval`, `confidence-based`, `iterative`

---

### ⭐⭐ Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models

**Authors:** (Various)
**Summary:** Learns to adaptively select between different retrieval strategies based on query complexity. Routes simple queries to no-retrieval, moderate to single-step RAG, and complex to multi-step RAG.
**PDF:** [NAACL 2024](https://aclanthology.org/2024.naacl-long.389.pdf)
**Venue:** NAACL 2024
**Tags:** `adaptive`, `query-complexity`, `strategy-selection`

---

### ⭐⭐ Agentic RAG: A Survey on Agentic RAG

**Authors:** (Various)
**Summary:** Comprehensive survey on integrating agent capabilities (planning, tool use, reflection) with RAG systems. Explores multi-agent RAG architectures.
**PDF:** [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
**Venue:** arXiv 2025
**Tags:** `agentic`, `multi-agent`, `tool-use`

---

## 8. Evaluation Frameworks & Benchmarks

### ⭐⭐⭐ RAGAS: Evaluation Framework for RAG

**Authors:** Esau et al.
**Summary:** Reference framework for evaluating RAG systems using faithfulness, answer relevancy, context recall, and context precision metrics. Industry standard for RAG evaluation.
**PDF:** [RAGAS Paper](https://arxiv.org/abs/2309.15217)
**Venue:** arXiv 2023
**Code:** [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
**Tags:** `evaluation`, `metrics`, `framework`

---

### ⭐⭐⭐ RAGChecker: A Fine-grained Framework for Diagnosing RAG

**Authors:** Dong et al.
**Summary:** Fine-grained diagnostic framework for RAG systems with detailed metrics for both retrieval and generation components. Evaluates 8 state-of-the-art RAG systems.
**PDF:** [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf)
**Venue:** NeurIPS 2024 D&B Track
**Tags:** `diagnostic`, `fine-grained`, `benchmark`

---

### ⭐⭐⭐ RAGBench: Explainable Benchmark for Retrieval-Augmented Generation

**Authors:** (Various)
**Summary:** First comprehensive, large-scale RAG benchmark with 100k examples covering five industry-specific domains and various RAG task types.
**PDF:** [arXiv:2407.11005](https://arxiv.org/html/2407.11005v1)
**Venue:** arXiv 2024
**Tags:** `benchmark`, `industry-domains`, `large-scale`

---

### ⭐⭐ ARES: An Automated Evaluation Framework for RAG

**Summary:** Automated evaluation framework comparing RAG systems. Contextualizes RAGAS, RAGChecker, and TruLens within unified evaluation.
**PDF:** [ResearchGate](https://www.researchgate.net/publication/382633733_ARES_An_Automated_Evaluation_Framework_for_Retrieval-Augmented_Generation_Systems)
**Tags:** `automated`, `evaluation`, `comparison`

---

### ⭐⭐ TruLens

**Summary:** Open-source evaluation framework by Snowflake for evaluating LLM applications including RAG. Provides groundedness, relevance, and comprehensiveness metrics.
**Code:** [trulens/trulens](https://github.com/trulens/trulens)
**Tags:** `evaluation`, `open-source`, `Snowflake`

---

### ⭐⭐ Evaluation of Retrieval-Augmented Generation: A Survey

**Authors:** (Various)
**Summary:** Comprehensive survey examining quantifiable metrics for Retrieval and Generation components including relevance, accuracy, and faithfulness.
**PDF:** [arXiv:2405.07437](https://arxiv.org/abs/2405.07437)
**Venue:** arXiv 2024 (343 citations)
**Tags:** `survey`, `metrics`, `evaluation`

---

## 9. Datasets for RAG Evaluation

### ⭐⭐⭐ MS MARCO

**Summary:** Large-scale question answering dataset from Microsoft with 8.8M passages and 1M+ queries. Standard benchmark for retrieval evaluation.
**Link:** [HuggingFace](https://huggingface.co/datasets/microsoft/ms_marco)
**Tags:** `benchmark`, `Microsoft`, `large-scale`

---

### ⭐⭐⭐ HotpotQA

**Summary:** Multi-hop question answering dataset requiring reasoning across multiple documents. Includes supporting fact supervision for explainability.
**Link:** [hotpotqa.github.io](https://hotpotqa.github.io)
**Tags:** `multi-hop`, `reasoning`, `explainable`

---

### ⭐⭐⭐ Natural Questions

**Summary:** Open-domain QA dataset with questions from real Google search queries. Includes long and short answer annotations.
**Tags:** `open-domain`, `real-queries`, `Google`

---

### ⭐⭐ MultiHop-RAG

**Summary:** Benchmark dataset specifically designed for evaluating multi-hop retrieval and reasoning capabilities in RAG systems.
**PDF:** [OpenReview](https://openreview.net/forum?id=t4eB3zYWBK)
**Venue:** OpenReview (241 citations)
**Tags:** `multi-hop`, `benchmark`, `reasoning`

---

### ⭐⭐ Natural Questions & WebQuestions

**Summary:** Standard open-domain QA benchmarks used for evaluating retrieval performance in RAG systems.
**Tags:** `benchmark`, `open-domain`, `standard`

---

## 10. Scalability & Vector Databases

### ⭐⭐⭐ Pinecone

**Summary:** Fully-managed vector database designed for production AI applications requiring semantic search at scale. Battle-tested for production workloads.
**Link:** [pinecone.io](https://www.pinecone.io)
**Tags:** `managed`, `production`, `scalable`

---

### ⭐⭐⭐ Milvus

**Summary:** Open-source vector database designed for high scalability, supporting distributed deployments for billions of vectors. Ideal for enterprises with massive datasets.
**Link:** [milvus.io](https://milvus.io)
**Code:** [milvus-io/milvus](https://github.com/milvus-io/milvus)
**Tags:** `open-source`, `distributed`, `scalable`

---

### ⭐⭐⭐ Weaviate

**Summary:** Open-source vector database with built-in ML modules. Supports hybrid search (vector + keyword) and GraphQL API.
**Link:** [weaviate.io](https://weaviate.io)
**Code:** [weaviate/weaviate](https://github.com/weaviate/weaviate)
**Tags:** `open-source`, `hybrid-search`, `GraphQL`

---

### ⭐⭐ Qdrant

**Summary:** High-performance, open-source vector database written in Rust. Offers filtering capabilities and distributed deployment.
**Link:** [qdrant.tech](https://qdrant.tech)
**Code:** [qdrant/qdrant](https://github.com/qdrant/qdrant)
**Tags:** `open-source`, `high-performance`, `Rust`

---

### ⭐⭐ Chroma

**Summary:** Open-source embedding database designed for LLM applications. Lightweight and easy to integrate.
**Link:** [trychroma.com](https://www.trychroma.com)
**Tags:** `open-source`, `lightweight`, `developer-friendly`

---

## 11. Production Deployments

### ⭐⭐⭐ Microsoft GraphRAG

**Summary:** Microsoft Research's approach using LLM-generated knowledge graphs to enhance RAG. Outperforms baseline RAG for complex information analysis.
**Link:** [Microsoft Research](https://www.microsoft.com/en-us/research/project/graphrag)
**PDF:** [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (1431 citations)
**Code:** [microsoft/graphrag](https://github.com/microsoft/graphrag)
**Tags:** `knowledge-graph`, `Microsoft`, `production`

---

### ⭐⭐⭐ OpenAI ChatGPT Retrieval Plugin

**Summary:** Official retrieval plugin enabling ChatGPT to access personal/work documents via natural language queries.
**Code:** [openai/chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin)
**Tags:** `OpenAI`, `plugin`, `production`

---

### ⭐⭐ Anthropic Contextual Retrieval

**Summary:** Anthropic's method for dramatically improving retrieval step in RAG using contextual chunking and contextual BM25.
**Link:** [Anthropic Blog](https://www.anthropic.com/news/contextual-retrieval)
**Tags:** `Anthropic`, `contextual`, `production`

---

### ⭐⭐ Google Vertex AI Vector Search

**Summary:** Google Cloud's managed vector search with hybrid search capabilities (semantic + keyword).
**Link:** [Google Cloud Docs](https://docs.cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search)
**Tags:** `Google`, `managed`, `hybrid-search`

---

### ⭐⭐ Enterprise LLM Market Report 2025

**Summary:** Industry analysis showing RAG implementations becoming common in enterprise, with Anthropic gaining 40% enterprise LLM spend.
**Link:** [Menlo Ventures Report](https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise)
**Tags:** `enterprise`, `market-analysis`, `trends`

---

## 12. Conference Talks & Videos

### ⭐⭐ NeurIPS 2024: Chain-of-Retrieval Augmented Generation

**Summary:** Introduces approach for training o1-like RAG models that retrieve and reason step-by-step.
**Link:** [neurips.cc](https://neurips.cc/virtual/2025/poster/116740)
**Tags:** `NeurIPS`, `chain-of-retrieval`, `reasoning`

---

### ⭐⭐ EMNLP 2023: Active Retrieval Augmented Generation (FLARE)

**Summary:** Lab seminar explaining the FLARE paper from EMNLP 2023.
**Link:** [YouTube](https://www.youtube.com/watch?v=fx97GTV02ZE)
**Tags:** `EMNLP`, `FLARE`, `tutorial`

---

### ⭐⭐ RAG Tutorial 2025: Complete Introduction

**Summary:** Comprehensive RAG fundamentals tutorial for building AI applications with LLMs and vector databases.
**Link:** [YouTube](https://www.youtube.com/watch?v=63B-3rqRFbQ)
**Tags:** `tutorial`, `introduction`, `hands-on`

---

### ⭐⭐ Query Decomposition + Fusion RAG Explained

**Summary:** Video explanation of handling multi-part queries through decomposition and fusion techniques.
**Link:** [YouTube](https://www.youtube.com/watch?v=mnfzje4dl_0)
**Tags:** `decomposition`, `fusion`, `video`

---

### ⭐⭐ IBM: What is Retrieval-Augmented Generation?

**Summary:** Senior Research Scientist Marina Danilevsky explains the LLM/RAG framework.
**Link:** [YouTube](https://www.youtube.com/watch?v=T-D1OfcDW1M)
**Tags:** `IBM`, `introduction`, `explanation`

---

## 13. Framework Implementations & Code Repositories

### ⭐⭐⭐ LangChain

**Summary:** Leading framework for building LLM applications with extensive RAG support including document loaders, retrievers, and query transformations.
**Code:** [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
**Docs:** [python.langchain.com](https://python.langchain.com)
**Tags:** `framework`, `open-source`, `production`

---

### ⭐⭐⭐ LlamaIndex

**Summary:** Data framework specifically designed for LLM applications. Excellent RAG pipeline support with advanced indexing and query engine capabilities.
**Code:** [run-llama/llama_index](https://github.com/run-llama/llama_index)
**Docs:** [docs.llamaindex.ai](https://docs.llamaindex.ai)
**Tags:** `framework`, `data-framework`, `RAG-focused`

---

### ⭐⭐⭐ RAG Techniques Repository

**Summary:** Comprehensive repository showcasing various advanced techniques for RAG systems with code implementations.
**Code:** [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
**Tags:** `techniques`, `code-examples`, `educational`

---

### ⭐⭐⭐ RAG from Scratch

**Summary:** Educational repository for building RAG systems from scratch without frameworks.
**Code:** [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)
**Tags:** `educational`, `from-scratch`, `learning`

---

### ⭐⭐ LangGraph

**Summary:** Framework for building stateful, multi-actor applications with LLMs. Excellent for implementing CRAG, Self-RAG, and agentic RAG patterns.
**Code:** [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
**Docs:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
**Tags:** `stateful`, `agents`, `advanced`

---

### ⭐⭐ Haystack

**Summary:** NLP framework with strong RAG support. Includes HyDE integration and production-ready pipelines.
**Code:** [deepset-ai/haystack](https://github.com/deepset-ai/haystack)
**Tags:** `framework`, `production`, `pipelines`

---

### ⭐⭐ Awesome-GraphRAG

**Summary:** Curated list of resources on graph-based retrieval-augmented generation.
**Code:** [DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
**Tags:** `curated-list`, `graph-rag`, `resources`

---

## 14. Annotated Reading Path

### 🟢 Introduction Level (Beginner)

**Start here if you're new to RAG:**

1. **[RAG for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)** - The foundational paper. Read to understand the core concept of combining parametric and non-parametric memory.
2. **[Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)** - Essential for understanding dense retrieval, a core component of modern RAG.
3. **[RAG Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997)** - Comprehensive overview of RAG paradigms (Naive → Advanced → Modular).
4. **[YouTube: IBM RAG Introduction](https://www.youtube.com/watch?v=T-D1OfcDW1M)** - Video explanation for visual learners.
5. **[LlamaIndex Documentation](https://docs.llamaindex.ai)** - Hands-on tutorials for building your first RAG application.

---

### 🟡 Intermediate Level

**Once you understand the basics, explore these:**

1. **[ColBERT (Khattab et al., 2020)](https://arxiv.org/abs/2004.12832)** - Late interaction mechanism for improved retrieval effectiveness.
2. **[HyDE (Gao et al., 2023)](https://arxiv.org/abs/2212.10496)** - Zero-shot retrieval via hypothetical documents.
3. **[RAPTOR (Sarthi et al., 2024)](https://arxiv.org/abs/2401.18059)** - Tree-structured indexing for hierarchical retrieval.
4. **[RAG-Fusion (Rackauckas, 2024)](https://arxiv.org/abs/2402.03367)** - Multi-query with reciprocal rank fusion.
5. **[RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)** - Learn to evaluate your RAG systems.

---

### 🔴 Advanced Level

**For practitioners building production systems:**

1. **[Self-RAG (Asai et al., 2024)](https://arxiv.org/abs/2310.11511)** - Self-reflective retrieval and generation.
2. **[CRAG (2024)](https://arxiv.org/abs/2401.15884)** - Corrective retrieval for robustness.
3. **[FLARE (Jiang et al., 2023)](https://arxiv.org/abs/2305.06983)** - Active retrieval during generation.
4. **[Adaptive-RAG (NAACL 2024)](https://aclanthology.org/2024.naacl-long.389.pdf)** - Query complexity-based strategy selection.
5. **[GraphRAG (Microsoft, 2024)](https://arxiv.org/abs/2404.16130)** - Knowledge graph-enhanced retrieval.
6. **[Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)** - Production-grade retrieval improvements.

---

### ⚫ System Implementation Level

**For deploying RAG in production:**

1. **[RAGChecker (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf)** - Fine-grained diagnostic evaluation.
2. **[RAGBench (2024)](https://arxiv.org/html/2407.11005v1)** - Large-scale benchmark for industry applications.
3. **[Agentic RAG Survey (2025)](https://arxiv.org/abs/2501.09136)** - Multi-agent RAG architectures.
4. **[LangGraph Documentation](https://langchain-ai.github.io/langgraph)** - Implementing complex RAG patterns.
5. **Vector Database Comparisons:**

   - [Pinecone](https://pinecone.io) - Managed, production-ready
   - [Milvus](https://milvus.io) - Open-source, scalable
   - [Weaviate](https://weaviate.io) - Hybrid search capabilities

---

## Quick Reference: Key Papers by Topic

| Topic               | Primary Paper            | Year | Citations |
| ------------------- | ------------------------ | ---- | --------- |
| Foundation          | RAG (Lewis et al.)       | 2020 | 5000+     |
| Pre-training        | REALM (Guu et al.)       | 2020 | 2000+     |
| Dense Retrieval     | DPR (Karpukhin et al.)   | 2020 | 4000+     |
| Late Interaction    | ColBERT (Khattab et al.) | 2020 | 2197      |
| Zero-shot Retrieval | HyDE (Gao et al.)        | 2023 | 673       |
| Tree Indexing       | RAPTOR (Sarthi et al.)   | 2024 | 376       |
| Self-Reflection     | Self-RAG (Asai et al.)   | 2024 | 500+      |
| Corrective RAG      | CRAG                     | 2024 | Growing   |
| Active Retrieval    | FLARE (Jiang et al.)     | 2023 | 1070      |
| Graph-Enhanced      | GraphRAG (Microsoft)     | 2024 | 1431      |
| Query Fusion        | RAG-Fusion               | 2024 | 146       |
| Evaluation          | RAGBench                 | 2024 | Growing   |

---

*This survey is continuously updated. For the CSV dataset of all resources, see the accompanying download file.*
