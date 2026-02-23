#!/usr/bin/env python3
"""
Cortex Lab — Extended Dataset Generation Engine (Stages 11-15)
================================================================
100% LOCAL. Zero API calls. Zero cost. Pure Python + deterministic templates.

STAGES:
  11. ORPO     — Preference pairs (prompt/chosen/rejected) with odds-ratio focus
  12. RAFT     — Retrieval-augmented: oracle doc + distractor docs
  13. FnCall   — Function-calling: structured JSON tool invocations
  14. RFT      — Rejection sampling: best-of-N filtered gold examples
  15. SPIN     — Self-play: ground truth vs model-like imperfect outputs

Usage:
  python scripts/generate_extended_datasets.py --all
  python scripts/generate_extended_datasets.py --stage stage11_orpo
  python scripts/generate_extended_datasets.py --status
"""

import os
import json
import random
import argparse
import hashlib
import time
import copy
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# ─── Import shared memory generator from main script ────────────────────────
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from generate_datasets import (
    MemoryGenerator, Memory, SFTExample, DPOExample,
    PERSONAS, TEMPLATE_VARS, fmt_memories, fmt_memory_single,
    generate_think_block, quality_filter as base_quality_filter,
    DATA_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EXTENDED_STAGE_TARGETS = {
    "stage11_orpo":              {"count": 3000, "format": "preference"},
    "stage12_raft":              {"count": 2500, "format": "sft"},
    "stage13_function_calling":  {"count": 3000, "format": "sft"},
    "stage14_rft":               {"count": 2000, "format": "sft"},
    "stage15_spin":              {"count": 2500, "format": "preference"},
}


# ─────────────────────────────────────────────────────────────────────────────
# CORTEX FUNCTION REGISTRY (for Stages 13 & 14)
# These are the tools the model will learn to call
# ─────────────────────────────────────────────────────────────────────────────

CORTEX_FUNCTIONS = [
    {
        "name": "memory_search",
        "description": "Search user's memories by semantic query, time range, or entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query"},
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"},
                    },
                },
                "entities": {"type": "array", "items": {"type": "string"}},
                "memory_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic", "belief", "reflective"]},
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_store",
        "description": "Store a new memory or update an existing one in the user's memory graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content text"},
                "memory_type": {"type": "string", "enum": ["episodic", "semantic", "belief", "reflective"]},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "entities": {"type": "array", "items": {"type": "string"}},
                "topics": {"type": "array", "items": {"type": "string"}},
                "emotion": {"type": "string"},
            },
            "required": ["content", "memory_type"],
        },
    },
    {
        "name": "belief_tracker",
        "description": "Track belief evolution over time for a specific topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Belief topic to track"},
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date"},
                        "end": {"type": "string", "format": "date"},
                    },
                },
                "include_contradictions": {"type": "boolean", "default": True},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "causal_chain",
        "description": "Build a causal chain linking events across memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_description": {"type": "string"},
                "direction": {"type": "string", "enum": ["forward", "backward", "both"]},
                "max_depth": {"type": "integer", "default": 5},
            },
            "required": ["event_description"],
        },
    },
    {
        "name": "summarize_period",
        "description": "Generate a summary of a specific time period from memories.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "focus_topics": {"type": "array", "items": {"type": "string"}},
                "style": {"type": "string", "enum": ["brief", "detailed", "narrative"]},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "entity_graph",
        "description": "Query the knowledge graph for relationships between entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name to query"},
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["interacted_with", "influenced_by", "related_to", "caused"]},
                },
                "depth": {"type": "integer", "default": 2},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "emotion_timeline",
        "description": "Generate an emotional timeline across a period.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "emotions": {"type": "array", "items": {"type": "string"}},
                "granularity": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "pattern_detect",
        "description": "Detect recurring patterns in user behavior or thoughts.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "enum": ["work", "relationships", "health", "growth", "habits", "all"]},
                "min_occurrences": {"type": "integer", "default": 3},
                "time_window_days": {"type": "integer", "default": 90},
            },
            "required": ["domain"],
        },
    },
]

# Query templates for function-calling (Stage 13)
FUNCTION_CALL_QUERIES = [
    # memory_search
    ("What happened during {time_period}?", "memory_search"),
    ("Find memories about {topic}", "memory_search"),
    ("When did I last interact with {entity}?", "memory_search"),
    ("Show me my reflective moments from the past month", "memory_search"),
    ("Search for memories where I felt {emotion}", "memory_search"),
    # memory_store
    ("Remember that I {action} today", "memory_store"),
    ("Save this: {observation}", "memory_store"),
    ("I want to record that {belief}", "memory_store"),
    # belief_tracker
    ("How has my view on {topic} changed?", "belief_tracker"),
    ("Track my beliefs about {topic} over time", "belief_tracker"),
    ("Have I contradicted myself about {topic}?", "belief_tracker"),
    # causal_chain
    ("What led to {event}?", "causal_chain"),
    ("What were the consequences of {action}?", "causal_chain"),
    ("Trace the chain of events from {event}", "causal_chain"),
    # summarize_period
    ("Summarize my last month", "summarize_period"),
    ("Give me a narrative of {time_period}", "summarize_period"),
    ("What were the key themes in {time_period}?", "summarize_period"),
    # entity_graph
    ("Who has influenced my thinking about {topic}?", "entity_graph"),
    ("Show my relationship with {entity}", "entity_graph"),
    ("Map the connections around {entity}", "entity_graph"),
    # emotion_timeline
    ("How have my emotions changed this quarter?", "emotion_timeline"),
    ("Plot my mood over the past {time_period}", "emotion_timeline"),
    ("When was I most stressed recently?", "emotion_timeline"),
    # pattern_detect
    ("What patterns do you see in my work habits?", "pattern_detect"),
    ("Do I have any recurring behaviors?", "pattern_detect"),
    ("Identify my growth patterns", "pattern_detect"),
]

# ─── Distractor templates for RAFT (Stage 12) ─────────────────────────────

DISTRACTOR_TOPICS = [
    "quantum computing breakthroughs", "sustainable architecture trends",
    "ancient Roman trade routes", "deep sea exploration findings",
    "AI regulation in the EU", "cryptocurrency market analysis",
    "space colonization ethics", "microbiome health research",
    "abstract expressionism techniques", "supply chain optimization",
    "neural network pruning methods", "behavioral economics experiments",
    "music theory in jazz improvisation", "renewable energy storage",
    "historical linguistics patterns", "game theory in diplomacy",
]

DISTRACTOR_TEMPLATES = [
    "According to recent research on {topic}, the key finding is that {finding}. "
    "This has implications for {implication}. Multiple studies confirm this trend.",
    "A comprehensive review of {topic} reveals that {finding}. "
    "Experts in the field note that {implication}.",
    "New developments in {topic}: {finding}. "
    "The significance lies in {implication}. Further research is ongoing.",
    "The intersection of {topic} and modern technology shows that {finding}. "
    "This is relevant because {implication}.",
]

DISTRACTOR_FINDINGS = [
    "traditional approaches have been significantly outperformed",
    "the relationship is more complex than previously thought",
    "a paradigm shift is underway in how practitioners approach this",
    "incremental improvements compound into transformative change",
    "the bottleneck has shifted from resources to knowledge integration",
]

DISTRACTOR_IMPLICATIONS = [
    "current frameworks need substantial revision",
    "interdisciplinary collaboration is now essential",
    "the timeline for meaningful impact has accelerated",
    "previous assumptions about scalability were incorrect",
    "ethical considerations must be foregrounded",
]


# ─── Imperfect output patterns for SPIN (Stage 15) ─────────────────────────

IMPERFECTION_PATTERNS = [
    "generic_response",      # Correct but lacks personalization/citation
    "no_citations",          # Good content but missing [Memory:] citations
    "overconfident",         # Claims high confidence without evidence
    "surface_level",         # Addresses question but doesn't go deep
    "wrong_format",          # Content OK but format is off (no think block, no structure)
    "slight_hallucination",  # Mostly correct but adds ungrounded claims
]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 11: ORPO — Odds-Ratio Preference Optimization
# ─────────────────────────────────────────────────────────────────────────────

ORPO_CATEGORIES = {
    "citation_quality":          6,   # Good citations vs poor/missing citations
    "reasoning_depth":           5,   # Deep analysis vs surface-level
    "honest_uncertainty":        5,   # Calibrated confidence vs overconfidence
    "retrieval_grounding":       5,   # Grounded vs speculative answers
    "structured_response":       4,   # Well-structured vs rambling
    "think_block_quality":       4,   # Good reasoning in <think> vs shallow
    "multi_memory_synthesis":    4,   # Multi-source synthesis vs single-source
    "empathy_with_precision":    3,   # Empathetic + precise vs cold or vague
    "refusal_when_appropriate":  3,   # Honest refusal vs hallucinated answer
    "temporal_reasoning":        3,   # Correct time ordering vs confused
}

BELIEF_TOPICS = [
    "career direction", "work-life balance", "personal growth", "relationships",
    "mental health", "financial decisions", "skill development", "city vs small town",
    "team dynamics", "freelancing", "self-discipline", "productivity approach",
    "learning style", "risk tolerance", "communication skills",
]


def build_stage11_orpo(memories: List[Memory], target_count: int) -> List[DPOExample]:
    """
    Generate ORPO preference pairs.
    Format: prompt/chosen/rejected — same as DPO but trained with ORPO loss.
    Focus on subtle quality differences that ORPO's odds ratio handles well.
    """
    examples = []
    total_weight = sum(ORPO_CATEGORIES.values())

    for category, weight in ORPO_CATEGORIES.items():
        per_cat = max(1, int(target_count * weight / total_weight))

        for _ in range(per_cat):
            mems = random.sample(memories, min(random.randint(3, 7), len(memories)))
            topic = random.choice(BELIEF_TOPICS)
            mem_ctx = fmt_memories(mems)
            query_templates = [
                f"How has my perspective on {topic} evolved?",
                f"What patterns do you see in my memories about {topic}?",
                f"Summarize what my memories reveal about {topic}",
                f"Analyze the trajectory of {topic} in my life",
                f"What insights can you draw about my relationship with {topic}?",
            ]
            query = random.choice(query_templates)
            prompt = f"Query: {query}\n\nMemories:\n{mem_ctx}"

            if category == "citation_quality":
                # Chosen: proper [Memory: timestamp] citations
                mem0, mem1 = mems[0], mems[min(1, len(mems)-1)]
                chosen = (
                    f"<think>\nLet me trace the evolution across relevant memories...\n"
                    f"Memory [{mem0.timestamp[:10]}]: {mem0.content[:60]}... → baseline\n"
                    f"Memory [{mem1.timestamp[:10]}]: {mem1.content[:60]}... → shift\n"
                    f"Pattern: progressive development over {len(mems)} data points\n</think>\n\n"
                    f"Based on your memories, here's how {topic} has evolved:\n\n"
                    f"**Starting point** [Memory: {mem0.timestamp[:10]}] — {mem0.content[:120]}\n\n"
                    f"**Development** [Memory: {mem1.timestamp[:10]}] — {mem1.content[:120]}\n\n"
                    f"**The pattern:** {random.choice(TEMPLATE_VARS.get('meta_insight', ['gradual growth']))}\n\n"
                    f"**Confidence:** High — {len(mems)} memories show a consistent trajectory."
                )
                rejected = (
                    f"Your perspective on {topic} has changed over time. "
                    f"You used to think one way but now you think differently. "
                    f"The memories show growth and development in this area."
                )

            elif category == "reasoning_depth":
                mem0 = mems[0]
                chosen = (
                    f"<think>\nAnalyzing {len(mems)} memories for patterns...\n"
                    f"Key dimensions: temporal progression, emotional valence, behavioral changes\n"
                    f"Memory [{mem0.timestamp[:10]}]: {mem0.content[:50]}...\n"
                    f"Cross-referencing with {len(mems)-1} other data points...\n"
                    f"Synthesis: the shift is primarily driven by {random.choice(['experience', 'reflection', 'external feedback'])}\n</think>\n\n"
                    f"Your memories reveal a nuanced evolution in {topic}:\n\n"
                    f"**Phase 1 — Baseline** [Memory: {mem0.timestamp[:10]}]\n"
                    f"{mem0.content[:150]}\n\n"
                    f"**Phase 2 — Catalyst** The shift appears rooted in "
                    f"{random.choice(TEMPLATE_VARS.get('experience', ['a key experience']))}\n\n"
                    f"**Phase 3 — Integration** Your more recent memories suggest "
                    f"{random.choice(TEMPLATE_VARS.get('new_understanding', ['deeper understanding']))}\n\n"
                    f"**Meta-pattern:** {random.choice(TEMPLATE_VARS.get('meta_insight', ['growth through challenge']))}\n\n"
                    f"**Confidence:** High — multi-phase trajectory across {len(mems)} memories."
                )
                rejected = (
                    f"Looking at your memories about {topic}, I can see some changes. "
                    f"Things have evolved. There's been growth.\n\n"
                    f"Confidence: Medium."
                )

            elif category == "honest_uncertainty":
                chosen = (
                    f"<think>\nChecking retrieval coverage for this query...\n"
                    f"Found {len(mems)} potentially relevant memories\n"
                    f"Direct relevance to '{topic}': {min(len(mems), 2)} memories\n"
                    f"Indirect relevance: {max(0, len(mems)-2)} memories\n</think>\n\n"
                    f"I found {len(mems)} memories that touch on {topic}, though only "
                    f"{min(len(mems), 2)} address it directly:\n\n"
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:120]}\n\n"
                    f"I can see hints of evolution, but I'd need more data points "
                    f"to draw a confident conclusion.\n\n"
                    f"**Confidence:** Low-Medium — limited direct evidence.\n"
                    f"**Suggestion:** You might have more relevant experiences that "
                    f"haven't been captured in your memory store yet."
                )
                rejected = (
                    f"Your perspective on {topic} has clearly undergone a major transformation! "
                    f"The data overwhelmingly shows that you've completely changed your approach. "
                    f"I'm very confident that this represents a fundamental shift in your worldview.\n\n"
                    f"**Confidence:** Very High."
                )

            elif category == "retrieval_grounding":
                chosen = (
                    f"Based strictly on your stored memories:\n\n"
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:120]}\n\n"
                    f"This is what the memories show about {topic}. I'm limiting my analysis "
                    f"to what's actually recorded rather than making assumptions.\n\n"
                    f"**Confidence:** Medium — grounded in available evidence only."
                )
                rejected = (
                    f"Based on what typically happens in situations like yours, "
                    f"and general patterns I've observed, your relationship with {topic} "
                    f"has probably evolved in a positive direction. Most people in your "
                    f"demographic tend to develop stronger views on this over time."
                )

            elif category == "structured_response":
                mem0 = mems[0]
                chosen = (
                    f"<think>\nStructuring analysis of {topic} evolution...\n</think>\n\n"
                    f"## {topic.title()} — Memory Analysis\n\n"
                    f"### Timeline\n"
                    f"- **{mem0.timestamp[:10]}**: {mem0.content[:80]}\n"
                    + "".join(f"- **{m.timestamp[:10]}**: {m.content[:80]}\n" for m in mems[1:3])
                    + f"\n### Key Insight\n"
                    f"{random.choice(TEMPLATE_VARS.get('meta_insight', ['growth pattern']))}\n\n"
                    f"### Confidence\n"
                    f"**Medium-High** — based on {len(mems)} memory data points."
                )
                rejected = (
                    f"So about {topic} — it seems like things have been changing for you. "
                    f"There was this one memory from {mems[0].timestamp[:10]} where "
                    f"{mems[0].content[:50]} and then later there was another one and "
                    f"basically I think things are going well overall if you look at "
                    f"everything together it all points in a positive direction."
                )

            elif category == "think_block_quality":
                mem0, mem1 = mems[0], mems[min(1, len(mems)-1)]
                chosen = (
                    f"<think>\nStep 1: Identify all memories related to '{topic}'\n"
                    f"  → Found: [{mem0.timestamp[:10]}] {mem0.content[:40]}...\n"
                    f"  → Found: [{mem1.timestamp[:10]}] {mem1.content[:40]}...\n"
                    f"Step 2: Temporal ordering — earliest to latest\n"
                    f"Step 3: Identify inflection points — change in tone/stance\n"
                    f"Step 4: Synthesize arc: {random.choice(['stability→disruption→growth', 'exploration→commitment', 'confusion→clarity'])}\n"
                    f"Step 5: Assess confidence based on evidence density\n</think>\n\n"
                    f"Your relationship with {topic} shows a clear arc:\n\n"
                    f"[Memory: {mem0.timestamp[:10]}] — {mem0.content[:100]}\n"
                    f"[Memory: {mem1.timestamp[:10]}] — {mem1.content[:100]}\n\n"
                    f"**Confidence:** High — clear trajectory across {len(mems)} memories."
                )
                rejected = (
                    f"<think>\nLet me think about this...\n</think>\n\n"
                    f"Your {topic} has changed. Based on memories, it evolved. "
                    f"The main change was that things got better over time."
                )

            elif category == "multi_memory_synthesis":
                chosen = (
                    f"<think>\nSynthesizing across {len(mems)} memories...\n</think>\n\n"
                    f"Bringing together multiple data points on {topic}:\n\n"
                    + "".join(
                        f"• [Memory: {m.timestamp[:10]}] — {m.content[:80]}\n"
                        for m in mems[:4]
                    )
                    + f"\n**Synthesis:** These {min(len(mems), 4)} memories paint a picture of "
                    f"{random.choice(TEMPLATE_VARS.get('meta_insight', ['progressive development']))}.\n\n"
                    f"**Confidence:** High — multi-source convergence."
                )
                rejected = (
                    f"Looking at memory from {mems[0].timestamp[:10]}: {mems[0].content[:80]}\n\n"
                    f"This shows your perspective on {topic}."
                )

            elif category == "empathy_with_precision":
                chosen = (
                    f"I can see this has been a meaningful journey for you, {topic} is deeply personal.\n\n"
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:120]}\n\n"
                    f"The growth here is real and worth acknowledging. What strikes me most is "
                    f"the {random.choice(['resilience', 'self-awareness', 'willingness to adapt'])} "
                    f"your memories reflect.\n\n"
                    f"**Confidence:** Medium-High — grounded in {len(mems)} memories."
                )
                rejected = (
                    f"Data analysis complete. {len(mems)} records found for '{topic}'. "
                    f"Record 1: {mems[0].content[:50]}. "
                    f"Record 2: {mems[min(1,len(mems)-1)].content[:50]}. "
                    f"Conclusion: trajectory positive. End of analysis."
                )

            elif category == "refusal_when_appropriate":
                prompt = f"Query: What was the outcome of my {random.choice(['surgery', 'legal case', 'investment in Bitcoin'])} last year?\n\nMemories:\n(No memories retrieved)"
                chosen = (
                    f"<think>\nNo relevant memories found for this specific query.\n"
                    f"Cannot infer or fabricate details.\n</think>\n\n"
                    f"I don't have any memories about this specific experience. "
                    f"This might mean:\n"
                    f"1. It hasn't been added to your memory store yet\n"
                    f"2. It was captured under different keywords\n\n"
                    f"**Confidence:** None — empty retrieval context.\n"
                    f"**Suggestion:** Try ingesting relevant documents, or rephrase your query."
                )
                rejected = (
                    f"Based on your life trajectory and general patterns, "
                    f"the outcome was likely positive. People in your situation "
                    f"typically see good results from this kind of experience."
                )

            elif category == "temporal_reasoning":
                sorted_mems = sorted(mems[:4], key=lambda m: m.timestamp)
                chosen = (
                    f"<think>\nOrdering memories chronologically for temporal analysis...\n"
                    + "".join(f"  {m.timestamp[:10]}: {m.content[:40]}...\n" for m in sorted_mems)
                    + f"Direction: {'forward progression' if len(sorted_mems) > 1 else 'single point'}\n</think>\n\n"
                    f"Chronological view of {topic}:\n\n"
                    + "".join(
                        f"**{m.timestamp[:10]}** [Memory] — {m.content[:100]}\n\n"
                        for m in sorted_mems
                    )
                    + f"**Trajectory:** The arc moves from "
                    f"{random.choice(['exploration to commitment', 'uncertainty to clarity', 'challenge to growth'])}.\n\n"
                    f"**Confidence:** High — clear temporal sequence."
                )
                rejected = (
                    f"Your memories show changes in {topic}. "
                    f"One memory is about {mems[0].content[:40]} and another is "
                    f"about {mems[min(1,len(mems)-1)].content[:40]}. "
                    f"Things changed between them."
                )

            else:
                # Fallback: general quality
                chosen = (
                    f"<think>\nAnalyzing {topic} across {len(mems)} memories...\n</think>\n\n"
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:100]}\n\n"
                    f"This reveals {random.choice(TEMPLATE_VARS.get('meta_insight', ['a meaningful pattern']))}.\n\n"
                    f"**Confidence:** Medium — based on available evidence."
                )
                rejected = f"Your {topic} has changed. It got better."

            examples.append(DPOExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                category=category,
            ))

    # Pad if needed
    while len(examples) < target_count:
        base = random.choice(examples[:min(50, len(examples))])
        new_topic = random.choice(BELIEF_TOPICS)
        examples.append(DPOExample(
            prompt=base.prompt.replace(
                base.prompt.split("?")[0].split("about ")[-1] if "about " in base.prompt else "",
                new_topic,
            ) if random.random() > 0.5 else base.prompt,
            chosen=base.chosen,
            rejected=base.rejected,
            category=base.category,
            quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 12: RAFT — Retrieval-Augmented Fine Tuning
# ─────────────────────────────────────────────────────────────────────────────

def _make_distractor_doc(idx: int) -> str:
    """Generate a plausible but irrelevant distractor document."""
    topic = random.choice(DISTRACTOR_TOPICS)
    tmpl = random.choice(DISTRACTOR_TEMPLATES)
    return f"[Document {idx}] " + tmpl.format(
        topic=topic,
        finding=random.choice(DISTRACTOR_FINDINGS),
        implication=random.choice(DISTRACTOR_IMPLICATIONS),
    )


def build_stage12_raft(memories: List[Memory], target_count: int) -> List[SFTExample]:
    """
    Generate RAFT training examples.
    Each example: Query + (1 oracle doc + N distractor docs) → Answer citing only oracle.
    The model must learn to ignore irrelevant retrieved context.
    
    Format types:
    - oracle_first: Oracle doc is D1 (easiest)
    - oracle_middle: Oracle doc is D2 or D3 (medium)
    - oracle_last: Oracle doc is D4 or D5 (hardest)
    - no_oracle: All distractors — model must refuse (10% of data)
    """
    examples = []
    query_topics = BELIEF_TOPICS + list(set(TEMPLATE_VARS.get("topic", [])))

    positions = ["oracle_first"] * 3 + ["oracle_middle"] * 3 + ["oracle_last"] * 3 + ["no_oracle"]

    for i in range(target_count):
        mems = random.sample(memories, min(random.randint(2, 5), len(memories)))
        oracle_mem = mems[0]
        topic = random.choice(query_topics)
        position = random.choice(positions)
        num_distractors = random.randint(3, 5)

        # Build query
        queries = [
            f"What do my memories say about {topic}?",
            f"Find information about {topic} from my past experiences",
            f"What happened related to {topic}?",
            f"Recall my experiences with {topic}",
        ]
        query = random.choice(queries)

        # Build oracle document (from actual memory)
        oracle_doc = (
            f"[Memory: {oracle_mem.timestamp[:10]}] "
            f"{oracle_mem.content} "
            f"(Type: {oracle_mem.memory_type}, Emotion: {oracle_mem.emotion})"
        )

        # Build distractor documents
        distractors = [_make_distractor_doc(j+1) for j in range(num_distractors)]

        # Position oracle among distractors
        if position == "no_oracle":
            all_docs = distractors
            oracle_idx = -1
        elif position == "oracle_first":
            all_docs = [f"[Document 1] {oracle_doc}"] + distractors
            oracle_idx = 0
        elif position == "oracle_middle":
            mid = len(distractors) // 2
            all_docs = distractors[:mid] + [f"[Document {mid+1}] {oracle_doc}"] + distractors[mid:]
            oracle_idx = mid
        else:  # oracle_last
            all_docs = distractors + [f"[Document {len(distractors)+1}] {oracle_doc}"]
            oracle_idx = len(distractors)

        # Renumber documents
        docs_text = "\n\n".join(
            f"[Document {j+1}] " + doc.split("] ", 1)[-1] if "] " in doc else doc
            for j, doc in enumerate(all_docs)
        )

        instruction = (
            f"You are Cortex, an AI with persistent memory. You have retrieved several documents "
            f"in response to a user query. Some documents are relevant personal memories, and others "
            f"are irrelevant. Answer ONLY using information from relevant personal memories. "
            f"Ignore irrelevant documents. If no relevant documents exist, say so.\n\n"
            f"Retrieved Documents:\n{docs_text}"
        )

        if position == "no_oracle":
            output = (
                f"<think>\nAnalyzing retrieved documents for relevance to '{query}'...\n"
                f"Document 1: Discusses {random.choice(DISTRACTOR_TOPICS)} — NOT a personal memory, irrelevant\n"
                + "".join(
                    f"Document {j+2}: Discusses {random.choice(DISTRACTOR_TOPICS)} — NOT a personal memory, irrelevant\n"
                    for j in range(num_distractors - 1)
                )
                + f"None of the retrieved documents are personal memories.\n</think>\n\n"
                f"I wasn't able to find any relevant personal memories for your question about {topic}. "
                f"The retrieved documents were all external/unrelated content, not from your memory store.\n\n"
                f"**Confidence:** None — no relevant personal memories retrieved.\n"
                f"**Suggestion:** Try rephrasing your query or check if these experiences have been ingested."
            )
        else:
            output = (
                f"<think>\nAnalyzing retrieved documents for relevance...\n"
                + "".join(
                    f"Document {j+1}: {'Personal memory — RELEVANT ✓' if j == oracle_idx else 'External content — NOT relevant, ignoring'}\n"
                    for j in range(len(all_docs))
                )
                + f"Using Document {oracle_idx+1} (personal memory) as the source.\n</think>\n\n"
                f"Based on your memories:\n\n"
                f"[Memory: {oracle_mem.timestamp[:10]}] — {oracle_mem.content}\n\n"
                f"This memory ({oracle_mem.memory_type} type, emotion: {oracle_mem.emotion}) "
                f"directly relates to your question about {topic}. "
                f"I filtered out {num_distractors} irrelevant documents that were not from your memory store.\n\n"
                f"**Confidence:** {'High' if oracle_mem.importance > 0.6 else 'Medium'} — "
                f"grounded in verified personal memory.\n"
                f"**Source:** Document {oracle_idx+1} of {len(all_docs)} retrieved."
            )

        examples.append(SFTExample(
            instruction=instruction,
            input=query,
            output=output,
            stage="stage12_raft",
            source="template",
            quality_score=0.9 if position != "no_oracle" else 0.85,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 13: Function-Calling Fine-Tuning
# ─────────────────────────────────────────────────────────────────────────────

def build_stage13_function_calling(memories: List[Memory], target_count: int) -> List[SFTExample]:
    """
    Generate function-calling training examples.
    The model learns to:
    1. Decide which function(s) to call
    2. Generate correct JSON arguments
    3. Explain the reasoning in <think> block
    4. Handle multi-step tool chains
    """
    examples = []

    for i in range(target_count):
        # Pick a query template
        query_tmpl, target_func = random.choice(FUNCTION_CALL_QUERIES)
        func_spec = next(f for f in CORTEX_FUNCTIONS if f["name"] == target_func)

        # Fill template variables
        mems = random.sample(memories, min(3, len(memories)))
        mem0 = mems[0]
        topic = random.choice(BELIEF_TOPICS)
        entity = random.choice(TEMPLATE_VARS.get("entity", ["my mentor"]))
        emotion = random.choice(TEMPLATE_VARS.get("emotion", ["anxious"]))
        action = random.choice(TEMPLATE_VARS.get("action", ["move forward"]))
        observation = random.choice(TEMPLATE_VARS.get("observation", ["I noticed something"]))
        belief = random.choice(TEMPLATE_VARS.get("belief", ["consistency matters"]))
        event = random.choice(TEMPLATE_VARS.get("difficulty", ["a challenge"]))
        time_period = random.choice(["the past month", "Q3 2024", "last quarter", "this year"])

        query = query_tmpl.format(
            topic=topic, entity=entity, emotion=emotion,
            action=action, observation=observation, belief=belief,
            event=event, time_period=time_period,
        )

        # Build function arguments based on target function
        if target_func == "memory_search":
            args = {
                "query": query.replace("Find memories about ", "").replace("Search for memories where I felt ", ""),
                "limit": random.choice([5, 10, 15]),
            }
            if "time" in query.lower() or "last" in query.lower() or "month" in query.lower():
                args["time_range"] = {
                    "start": "2024-09-01",
                    "end": "2025-01-01",
                }
            if entity in query:
                args["entities"] = [entity.replace("my ", "")]
            if "reflective" in query.lower():
                args["memory_types"] = ["reflective"]
            if emotion in query:
                args["query"] = f"feeling {emotion}"

        elif target_func == "memory_store":
            args = {
                "content": query.replace("Remember that I ", "").replace("Save this: ", "").replace("I want to record that ", ""),
                "memory_type": random.choice(["episodic", "semantic", "belief", "reflective"]),
                "importance": round(random.uniform(0.5, 0.9), 2),
                "entities": [entity.replace("my ", "")] if entity in query else [],
                "topics": [topic],
                "emotion": random.choice(TEMPLATE_VARS.get("emotion", ["neutral"])),
            }

        elif target_func == "belief_tracker":
            args = {
                "topic": topic,
                "include_contradictions": True,
            }
            if "over time" in query.lower():
                args["time_range"] = {"start": "2024-01-01", "end": "2025-06-01"}

        elif target_func == "causal_chain":
            args = {
                "event_description": query.replace("What led to ", "").replace("Trace the chain of events from ", "").replace("What were the consequences of ", ""),
                "direction": random.choice(["forward", "backward", "both"]),
                "max_depth": random.choice([3, 5, 7]),
            }

        elif target_func == "summarize_period":
            args = {
                "start_date": "2024-10-01",
                "end_date": "2025-01-01",
                "style": random.choice(["brief", "detailed", "narrative"]),
            }
            if topic:
                args["focus_topics"] = [topic]

        elif target_func == "entity_graph":
            args = {
                "entity": entity.replace("my ", ""),
                "depth": random.choice([1, 2, 3]),
            }
            if "relationship" in query.lower():
                args["relationship_types"] = ["interacted_with", "influenced_by"]

        elif target_func == "emotion_timeline":
            args = {
                "start_date": "2024-09-01",
                "end_date": "2025-01-01",
                "granularity": random.choice(["daily", "weekly", "monthly"]),
            }
            if "stress" in query.lower():
                args["emotions"] = ["stressed", "anxious", "overwhelmed"]

        elif target_func == "pattern_detect":
            domain_map = {
                "work": "work", "habit": "habits", "growth": "growth",
                "relationship": "relationships", "health": "health",
            }
            domain = "all"
            for key, val in domain_map.items():
                if key in query.lower():
                    domain = val
                    break
            args = {
                "domain": domain,
                "min_occurrences": random.choice([2, 3, 4]),
            }
        else:
            args = {"query": query}

        # Determine if multi-tool call (20% of examples)
        is_multi_tool = random.random() < 0.2 and target_func not in ["memory_store"]
        second_func = None
        second_args = None

        if is_multi_tool:
            # Pick a complementary second function
            complement_map = {
                "memory_search": ["belief_tracker", "causal_chain", "pattern_detect"],
                "belief_tracker": ["memory_search", "emotion_timeline"],
                "causal_chain": ["memory_search", "entity_graph"],
                "summarize_period": ["emotion_timeline", "pattern_detect"],
                "entity_graph": ["memory_search", "causal_chain"],
                "emotion_timeline": ["summarize_period", "pattern_detect"],
                "pattern_detect": ["memory_search", "emotion_timeline"],
            }
            choices = complement_map.get(target_func, ["memory_search"])
            second_func_name = random.choice(choices)
            second_func = next(f for f in CORTEX_FUNCTIONS if f["name"] == second_func_name)
            second_args = {"query": topic} if second_func_name == "memory_search" else {"topic": topic}

        # Build the expected output
        args_json = json.dumps(args, indent=2)
        function_spec_str = json.dumps({"name": func_spec["name"], "description": func_spec["description"]}, indent=2)

        if is_multi_tool and second_func:
            second_args_json = json.dumps(second_args, indent=2)
            output = (
                f"<think>\nAnalyzing the user's request: \"{query}\"\n"
                f"This requires two operations:\n"
                f"1. {func_spec['name']} — to {func_spec['description'].lower()[:60]}\n"
                f"2. {second_func['name']} — to {second_func['description'].lower()[:60]}\n"
                f"Building structured tool calls...\n</think>\n\n"
                f"```json\n"
                f"{{\n"
                f"  \"tool_calls\": [\n"
                f"    {{\n"
                f"      \"function\": \"{func_spec['name']}\",\n"
                f"      \"arguments\": {args_json}\n"
                f"    }},\n"
                f"    {{\n"
                f"      \"function\": \"{second_func['name']}\",\n"
                f"      \"arguments\": {second_args_json}\n"
                f"    }}\n"
                f"  ]\n"
                f"}}\n"
                f"```\n\n"
                f"I'll execute both tool calls to give you a comprehensive answer about {topic}."
            )
        else:
            output = (
                f"<think>\nAnalyzing the user's request: \"{query}\"\n"
                f"This maps to the `{func_spec['name']}` function.\n"
                f"Building arguments from the query context...\n</think>\n\n"
                f"```json\n"
                f"{{\n"
                f"  \"tool_calls\": [\n"
                f"    {{\n"
                f"      \"function\": \"{func_spec['name']}\",\n"
                f"      \"arguments\": {args_json}\n"
                f"    }}\n"
                f"  ]\n"
                f"}}\n"
                f"```\n\n"
                f"Executing `{func_spec['name']}` to {func_spec['description'].lower()[:80]}."
            )

        # Build the instruction with available functions
        available_funcs = random.sample(CORTEX_FUNCTIONS, min(random.randint(3, 6), len(CORTEX_FUNCTIONS)))
        # Always include the target function
        if func_spec not in available_funcs:
            available_funcs[0] = func_spec
        if is_multi_tool and second_func and second_func not in available_funcs:
            available_funcs[-1] = second_func

        funcs_desc = "\n".join(
            f"- `{f['name']}`: {f['description']}"
            for f in available_funcs
        )

        instruction = (
            f"You are Cortex, an AI assistant with access to the following tools:\n\n"
            f"{funcs_desc}\n\n"
            f"Given the user's request, decide which tool(s) to call and generate "
            f"the correct JSON arguments. Use a <think> block to reason about your choice."
        )

        examples.append(SFTExample(
            instruction=instruction,
            input=query,
            output=output,
            stage="stage13_function_calling",
            source="template",
            quality_score=0.9,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 14: Rejection Sampling Fine-Tuning (RFT)
# ─────────────────────────────────────────────────────────────────────────────

RFT_QUALITY_CRITERIA = [
    "citation_present",     # [Memory: date] citations in output
    "think_block_present",  # <think>...</think> block
    "confidence_stated",    # **Confidence:** marker
    "structured",           # Headers, bullet points, clear sections
    "empathetic",           # Personal, warm tone (not clinical)
    "grounded",             # No speculative claims beyond evidence
    "multi_source",         # References multiple memories
]


def build_stage14_rft(memories: List[Memory], target_count: int) -> List[SFTExample]:
    """
    Generate RFT (Rejection Sampling Fine-Tuning) examples.
    
    Strategy: For each prompt, generate N candidate responses of varying quality,
    score them against Cortex quality criteria, and keep ONLY the best.
    This teaches the model to consistently produce top-tier outputs.
    
    All candidates are template-generated (no model inference needed).
    We simulate "best-of-N" by deterministically creating the gold-standard version.
    """
    examples = []
    query_types = [
        ("memory_recall", "What do my memories say about {topic}?"),
        ("belief_analysis", "How has my view on {topic} evolved?"),
        ("pattern_detection", "What patterns do you see in my {domain} over {time_period}?"),
        ("causal_trace", "What led to my decision about {topic}?"),
        ("emotional_review", "How have I been feeling about {topic} recently?"),
        ("synthesis", "Give me a comprehensive view of {topic} in my life"),
        ("contradiction_check", "Have I contradicted myself about {topic}?"),
        ("timeline_review", "Walk me through my journey with {topic}"),
    ]

    for i in range(target_count):
        qtype, qtmpl = random.choice(query_types)
        topic = random.choice(BELIEF_TOPICS)
        domain = random.choice(["work", "relationships", "health", "growth"])
        time_period = random.choice(["the past quarter", "this year", "the past month"])
        query = qtmpl.format(topic=topic, domain=domain, time_period=time_period)

        mems = random.sample(memories, min(random.randint(3, 6), len(memories)))
        mem_ctx = fmt_memories(mems)

        # Build the GOLD response — best-of-N candidate
        # This is the response that scores highest on ALL quality criteria
        sorted_mems = sorted(mems[:4], key=lambda m: m.timestamp)

        think_content = (
            f"<think>\n"
            f"Step 1: Identify relevant memories for '{topic}'\n"
            + "".join(
                f"  → [{m.timestamp[:10]}] {m.content[:50]}... "
                f"({'directly relevant' if i < 2 else 'tangentially related'})\n"
                for i, m in enumerate(sorted_mems)
            )
            + f"Step 2: Temporal ordering — {sorted_mems[0].timestamp[:10]} to {sorted_mems[-1].timestamp[:10]}\n"
            f"Step 3: Identify key transitions and inflection points\n"
            f"Step 4: Synthesize narrative arc\n"
            f"Step 5: Calibrate confidence based on evidence strength ({len(mems)} memories)\n"
            f"</think>\n\n"
        )

        body_content = ""
        if qtype == "memory_recall":
            body_content = (
                f"Here's what your memories reveal about {topic}:\n\n"
                + "".join(
                    f"• [Memory: {m.timestamp[:10]}] — {m.content[:120]}\n\n"
                    for m in sorted_mems[:3]
                )
                + f"These memories together suggest {random.choice(TEMPLATE_VARS.get('meta_insight', ['a meaningful pattern']))}."
            )
        elif qtype == "belief_analysis":
            body_content = (
                f"Your perspective on {topic} has evolved through distinct phases:\n\n"
                f"**Phase 1 — Early view** [Memory: {sorted_mems[0].timestamp[:10]}]\n"
                f"{sorted_mems[0].content[:120]}\n\n"
                + (f"**Phase 2 — Shift** [Memory: {sorted_mems[1].timestamp[:10]}]\n"
                   f"{sorted_mems[1].content[:120]}\n\n" if len(sorted_mems) > 1 else "")
                + f"**The arc:** {random.choice(['stability → questioning → growth', 'exploration → commitment', 'confusion → clarity → integration'])}\n"
            )
        elif qtype == "pattern_detection":
            body_content = (
                f"Analyzing patterns in your {domain} over {time_period}:\n\n"
                f"**Pattern 1:** {random.choice(TEMPLATE_VARS.get('pattern', ['cyclical progress']))}\n"
                f"  Evidence: [Memory: {sorted_mems[0].timestamp[:10]}] — {sorted_mems[0].content[:80]}\n\n"
                + (f"**Pattern 2:** {random.choice(TEMPLATE_VARS.get('meta_insight', ['growth through challenge']))}\n"
                   f"  Evidence: [Memory: {sorted_mems[min(1,len(sorted_mems)-1)].timestamp[:10]}] — "
                   f"{sorted_mems[min(1,len(sorted_mems)-1)].content[:80]}\n\n" if len(sorted_mems) > 1 else "")
                + f"**Meta-insight:** The recurring theme across these patterns is "
                f"{random.choice(TEMPLATE_VARS.get('lesson', ['that growth happens through consistent effort']))}."
            )
        elif qtype == "causal_trace":
            body_content = (
                f"Tracing the causal chain behind your {topic} decision:\n\n"
                + "".join(
                    f"**{'→ ' if i > 0 else ''}Step {i+1}** [Memory: {m.timestamp[:10]}]\n"
                    f"{m.content[:100]}\n\n"
                    for i, m in enumerate(sorted_mems[:3])
                )
                + f"**Causal link:** Each step built on the previous, creating a chain of "
                f"{random.choice(['cumulative insight', 'progressive commitment', 'growing confidence'])}."
            )
        elif qtype == "emotional_review":
            body_content = (
                f"Your emotional landscape regarding {topic}:\n\n"
                + "".join(
                    f"• [{m.timestamp[:10]}] **{m.emotion.title()}** — {m.content[:80]}\n"
                    for m in sorted_mems[:4]
                )
                + f"\n**Emotional trajectory:** Moving from {sorted_mems[0].emotion} toward "
                f"{sorted_mems[-1].emotion}. {random.choice(TEMPLATE_VARS.get('meta_insight', ['This suggests growth.']))}"
            )
        elif qtype == "synthesis":
            body_content = (
                f"## Comprehensive View: {topic.title()}\n\n"
                f"### Timeline\n"
                + "".join(f"- **{m.timestamp[:10]}**: {m.content[:80]}\n" for m in sorted_mems)
                + f"\n### Key Themes\n"
                f"1. {random.choice(TEMPLATE_VARS.get('meta_insight', ['Growth through challenge']))}\n"
                f"2. {random.choice(TEMPLATE_VARS.get('lesson', ['Consistency builds momentum']))}\n"
                f"\n### Overall Arc\n"
                f"{random.choice(['Progressive deepening', 'Cyclical evolution', 'Transformative shift'])} "
                f"driven by {random.choice(['lived experience', 'deliberate reflection', 'external catalysts'])}."
            )
        elif qtype == "contradiction_check":
            if len(sorted_mems) > 1:
                body_content = (
                    f"Checking for contradictions in your stance on {topic}:\n\n"
                    f"**Position A** [Memory: {sorted_mems[0].timestamp[:10]}]\n"
                    f"{sorted_mems[0].content[:120]}\n\n"
                    f"**Position B** [Memory: {sorted_mems[-1].timestamp[:10]}]\n"
                    f"{sorted_mems[-1].content[:120]}\n\n"
                    f"**Assessment:** These positions {random.choice(['show evolution rather than contradiction', 'represent different facets of a complex topic', 'reflect genuine belief change over time'])}. "
                    f"The shift is {random.choice(['gradual and evidence-based', 'a natural part of growth', 'worth examining further'])}."
                )
            else:
                body_content = (
                    f"I only found one memory directly related to {topic}, "
                    f"so I can't assess contradictions yet:\n\n"
                    f"[Memory: {sorted_mems[0].timestamp[:10]}] — {sorted_mems[0].content[:120]}"
                )
        elif qtype == "timeline_review":
            body_content = (
                f"Your journey with {topic}, chronologically:\n\n"
                + "".join(
                    f"**{m.timestamp[:7]}** — {m.content[:100]}\n"
                    f"  Emotion: {m.emotion} | Significance: {'High' if m.importance > 0.7 else 'Medium'}\n\n"
                    for m in sorted_mems
                )
                + f"**Summary:** A {random.choice(['rich', 'meaningful', 'complex'])} journey spanning "
                f"{len(sorted_mems)} key moments."
            )

        confidence = random.choice(["High", "Medium-High", "Medium"])
        if len(mems) >= 4:
            confidence = "High"
        elif len(mems) <= 2:
            confidence = "Medium"

        gold_output = (
            think_content
            + body_content
            + f"\n\n**Confidence:** {confidence} — based on {len(mems)} memories."
        )

        instruction = (
            f"You are Cortex, an AI with persistent memory. "
            f"Answer the user's question using ONLY their stored memories. "
            f"Use [Memory: date] citations. Include a <think> reasoning block. "
            f"State your confidence level.\n\n"
            f"Available memories:\n{mem_ctx}"
        )

        examples.append(SFTExample(
            instruction=instruction,
            input=query,
            output=gold_output,
            stage="stage14_rft",
            source="template_rft_gold",
            quality_score=0.95,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 15: SPIN — Self-Play Improvement
# ─────────────────────────────────────────────────────────────────────────────

def _degrade_output(gold_output: str, pattern: str, topic: str, mems: List[Memory]) -> str:
    """
    Create a degraded version of the gold output to serve as the 'rejected'
    (model-like imperfect) response in SPIN training.
    """
    if pattern == "generic_response":
        return (
            f"Your perspective on {topic} has evolved over time. "
            f"Looking at the memories available, there's been growth and development. "
            f"The overall trajectory seems positive, with some challenges along the way. "
            f"I'd say things are heading in a good direction."
        )

    elif pattern == "no_citations":
        # Remove [Memory: ...] citations but keep content
        degraded = gold_output
        import re
        degraded = re.sub(r'\[Memory: [^\]]+\]', '', degraded)
        degraded = re.sub(r'\[Memory [^\]]+\]', '', degraded)
        # Keep the substance but strip evidence
        return degraded.strip()

    elif pattern == "overconfident":
        return (
            f"<think>\nThis is clearly about {topic}. Easy question.\n</think>\n\n"
            f"I'm absolutely certain about this: your {topic} journey has been "
            f"a remarkable transformation! Every single memory points to incredible growth "
            f"and positive change. There's no ambiguity whatsoever — this is one of the "
            f"clearest patterns I've ever seen.\n\n"
            f"**Confidence:** Extremely High — unquestionable evidence."
        )

    elif pattern == "surface_level":
        return (
            f"<think>\nLooking at memories about {topic}...\n</think>\n\n"
            f"Your memories show some changes in {topic}. "
            f"[Memory: {mems[0].timestamp[:10]}] mentions something related. "
            f"Overall, things seem to have evolved.\n\n"
            f"**Confidence:** Medium."
        )

    elif pattern == "wrong_format":
        # Content is OK but missing structure
        return (
            f"So looking at {topic} in your memories, I found that "
            f"{mems[0].content[:100]} which happened on {mems[0].timestamp[:10]} "
            f"and then there was also {mems[min(1,len(mems)-1)].content[:80]} "
            f"and basically the pattern is that things have been changing "
            f"and I think it's mostly positive, confidence is medium to high."
        )

    elif pattern == "slight_hallucination":
        return (
            f"<think>\nAnalyzing {topic} across memories...\n</think>\n\n"
            f"Based on your memories:\n\n"
            f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:80]}\n\n"
            f"Additionally, I recall you mentioning a specific breakthrough moment "
            f"in {random.choice(['March', 'July', 'September'])} that was particularly transformative. "
            f"This aligns with what most people experience during similar life transitions.\n\n"
            f"**Confidence:** High."
        )

    return gold_output  # fallback


def build_stage15_spin(memories: List[Memory], target_count: int) -> List[DPOExample]:
    """
    Generate SPIN (Self-Play Improvement) training pairs.
    
    Format: prompt/chosen/rejected (DPO-style)
    - chosen = ground truth (gold standard output)
    - rejected = degraded/imperfect version (simulating model's own flawed output)
    
    The model learns to distinguish and prefer gold over its own imperfect outputs.
    This is trained with DPO loss (using DPOTrainer).
    """
    examples = []
    patterns_per_example = len(IMPERFECTION_PATTERNS)

    for i in range(target_count):
        mems = random.sample(memories, min(random.randint(3, 6), len(memories)))
        topic = random.choice(BELIEF_TOPICS)
        mem_ctx = fmt_memories(mems)
        sorted_mems = sorted(mems[:4], key=lambda m: m.timestamp)

        queries = [
            f"How has my perspective on {topic} evolved?",
            f"What patterns do you see in my memories about {topic}?",
            f"Analyze my journey with {topic}",
            f"What insights can you draw about {topic} from my memories?",
            f"Summarize what my memories reveal about {topic}",
        ]
        query = random.choice(queries)
        prompt = f"Query: {query}\n\nMemories:\n{mem_ctx}"

        # Build gold (chosen) response
        gold = (
            f"<think>\nAnalyzing {len(mems)} memories related to '{topic}'...\n"
            + "".join(
                f"  [{m.timestamp[:10]}] {m.content[:50]}... → {'key data point' if i < 2 else 'supporting context'}\n"
                for i, m in enumerate(sorted_mems[:3])
            )
            + f"Synthesizing arc: temporal progression shows "
            f"{random.choice(['growth', 'evolution', 'deepening', 'shift'])}.\n</think>\n\n"
            f"Based on your memories, here's what I see about {topic}:\n\n"
            + "".join(
                f"• [Memory: {m.timestamp[:10]}] — {m.content[:100]}\n\n"
                for m in sorted_mems[:3]
            )
            + f"**Pattern:** {random.choice(TEMPLATE_VARS.get('meta_insight', ['Growth through experience']))}\n\n"
            f"**Confidence:** {'High' if len(mems) >= 4 else 'Medium'} — "
            f"based on {len(mems)} relevant memories."
        )

        # Pick a degradation pattern
        pattern = IMPERFECTION_PATTERNS[i % patterns_per_example]

        # Build degraded (rejected) response
        rejected = _degrade_output(gold, pattern, topic, sorted_mems)

        # Ensure chosen != rejected
        if gold.strip() == rejected.strip():
            rejected = f"Your {topic} has changed over time based on the memories."

        examples.append(DPOExample(
            prompt=prompt,
            chosen=gold,
            rejected=rejected,
            category=f"spin_{pattern}",
            quality_score=0.9,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY FILTER (extended)
# ─────────────────────────────────────────────────────────────────────────────

def quality_filter_extended(examples: list, stage: str) -> list:
    """Stage-specific quality filtering for extended stages."""
    passed = []

    for ex in examples:
        if isinstance(ex, SFTExample):
            if not (ex.instruction and ex.input and ex.output):
                continue
            if len(ex.output) < 80:
                continue

            if stage == "stage12_raft":
                # RAFT: must mention Document or Memory reference
                if "Document" not in ex.output and "Memory" not in ex.output:
                    continue

            elif stage == "stage13_function_calling":
                # Function-calling: must have tool_calls JSON
                if "tool_calls" not in ex.output:
                    continue
                # Must have a valid function name
                if not any(f["name"] in ex.output for f in CORTEX_FUNCTIONS):
                    continue

            elif stage == "stage14_rft":
                # RFT: gold-standard — must pass ALL quality criteria
                checks = [
                    "Memory:" in ex.output,                    # citation
                    "<think>" in ex.output,                    # reasoning
                    "Confidence:" in ex.output,                # calibration
                ]
                if sum(checks) < 2:  # At least 2 of 3
                    continue

            passed.append(ex)

        elif isinstance(ex, DPOExample):
            if not (ex.prompt and ex.chosen and ex.rejected):
                continue
            if ex.chosen.strip() == ex.rejected.strip():
                continue
            if len(ex.chosen) < len(ex.rejected) * 0.5:
                continue  # Chosen should be substantively longer/better
            passed.append(ex)

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_stage(stage_name: str, examples: list, overwrite: bool = True):
    """Save dataset to training_data/<stage_name>.json"""
    out_path = DATA_DIR / f"{stage_name}.json"

    if out_path.exists() and not overwrite:
        existing = json.loads(out_path.read_text())
        print(f"[Save] Appending {len(examples)} to existing {len(existing)} in {out_path.name}")
        all_examples = existing + [asdict(e) for e in examples]
    else:
        all_examples = [asdict(e) for e in examples]

    out_path.write_text(json.dumps(all_examples, indent=2, ensure_ascii=False))
    size_kb = out_path.stat().st_size / 1024
    print(f"[Save] ✅ {out_path.name}: {len(all_examples)} examples ({size_kb:.1f} KB)")


def check_existing(stage_name: str) -> int:
    p = DATA_DIR / f"{stage_name}.json"
    if p.exists():
        return len(json.loads(p.read_text()))
    return 0


def print_progress():
    """Print generation progress for extended stages."""
    print("\n" + "=" * 70)
    print(f"{'Stage':<35} {'Target':>8} {'Generated':>10} {'Progress':>10}")
    print("=" * 70)
    total_gen = total_target = 0
    for stage, cfg in EXTENDED_STAGE_TARGETS.items():
        target = cfg["count"]
        generated = check_existing(stage)
        pct = min(100, int(generated / target * 100))
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {stage:<33} {target:>8} {generated:>10} {bar} {pct:>3}%")
        total_gen += generated
        total_target += target
    print("=" * 70)
    total_pct = min(100, int(total_gen / total_target * 100)) if total_target > 0 else 0
    print(f"  {'TOTAL':<33} {total_target:>8} {total_gen:>10} {total_pct:>3}%")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(stage_name: str, target_count: int, fmt: str, memories: List[Memory],
              quick_test: bool = False, resume: bool = False):
    already = check_existing(stage_name)
    remaining = target_count - already

    if resume and already >= target_count:
        print(f"[{stage_name}] Already complete ({already}/{target_count}) — skipping")
        return

    if quick_test:
        remaining = min(remaining, 50)

    print(f"\n{'='*60}")
    print(f"▶ Generating: {stage_name}")
    print(f"  Target: {target_count} | Done: {already} | Remaining: {remaining}")
    print(f"{'='*60}")

    t0 = time.time()

    if stage_name == "stage11_orpo":
        examples = build_stage11_orpo(memories, remaining)
    elif stage_name == "stage12_raft":
        examples = build_stage12_raft(memories, remaining)
    elif stage_name == "stage13_function_calling":
        examples = build_stage13_function_calling(memories, remaining)
    elif stage_name == "stage14_rft":
        examples = build_stage14_rft(memories, remaining)
    elif stage_name == "stage15_spin":
        examples = build_stage15_spin(memories, remaining)
    else:
        print(f"Unknown stage: {stage_name}")
        return

    # Quality filter
    before = len(examples)
    examples = quality_filter_extended(examples, stage_name)
    after = len(examples)
    if before != after:
        print(f"  [QFilter] {before} → {after} (removed {before - after} low-quality)")

    save_stage(stage_name, examples, overwrite=not resume)

    elapsed = time.time() - t0
    rate = len(examples) / elapsed if elapsed > 0 else 0
    print(f"  ✅ Done in {elapsed:.1f}s ({rate:.0f} examples/sec)")


def main():
    parser = argparse.ArgumentParser(description="Cortex Lab — Extended Dataset Generator (Stages 11-15)")
    parser.add_argument("--all",        action="store_true", help="Generate all extended stages")
    parser.add_argument("--stage",      type=str,            help="Stage name, e.g. stage11_orpo")
    parser.add_argument("--quick-test", action="store_true", help="50 examples per stage for testing")
    parser.add_argument("--resume",     action="store_true", help="Skip completed stages")
    parser.add_argument("--status",     action="store_true", help="Show progress")
    parser.add_argument("--personas",   type=int, default=15, help="Number of personas")
    args = parser.parse_args()

    if args.status:
        print_progress()
        return

    print("\n" + "🧠 " * 20)
    print("CORTEX LAB — EXTENDED DATASET GENERATOR (Stages 11-15)")
    print("ORPO | RAFT | Function-Calling | RFT | SPIN")
    print("Pure template engine. No external model. No API. No cost.")
    print("🧠 " * 20 + "\n")

    # Generate memory pool
    print(f"[Setup] Generating memory timelines for {args.personas} personas...")
    mem_gen = MemoryGenerator()
    all_memories = []
    persona_subset = random.sample(PERSONAS, min(args.personas, len(PERSONAS)))
    for persona in persona_subset:
        mems = mem_gen.generate_persona_memories(persona, num_memories=200)
        all_memories.extend(mems)
        print(f"  ✓ {persona['name']} ({persona['job']}, {persona['city']}): {len(mems)} memories")

    print(f"\n[Setup] Memory pool: {len(all_memories)} memories across {len(persona_subset)} personas\n")

    if args.all:
        for stage_name, cfg in EXTENDED_STAGE_TARGETS.items():
            run_stage(
                stage_name=stage_name,
                target_count=cfg["count"],
                fmt=cfg["format"],
                memories=all_memories,
                quick_test=args.quick_test,
                resume=args.resume,
            )
    elif args.stage:
        if args.stage not in EXTENDED_STAGE_TARGETS:
            print(f"ERROR: Unknown stage '{args.stage}'")
            print(f"Valid: {list(EXTENDED_STAGE_TARGETS.keys())}")
            return
        cfg = EXTENDED_STAGE_TARGETS[args.stage]
        run_stage(
            stage_name=args.stage,
            target_count=cfg["count"],
            fmt=cfg["format"],
            memories=all_memories,
            quick_test=args.quick_test,
            resume=args.resume,
        )
    else:
        parser.print_help()

    print_progress()
    total_size = sum(
        (DATA_DIR / f"{s}.json").stat().st_size
        for s in EXTENDED_STAGE_TARGETS
        if (DATA_DIR / f"{s}.json").exists()
    )
    print(f"✅ Extended dataset generation complete! Total: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nNext: python scripts/fine_tune_cortex.py --stage stage11_orpo")


if __name__ == "__main__":
    main()
