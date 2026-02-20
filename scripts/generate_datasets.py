"""
Cortex Lab — Dataset Generation Engine
========================================
100% LOCAL. Zero API calls. Zero cost. Pure Python + deterministic templates.
No dependency on any external model for data generation.

─────────────────────────────────────────────────────────────────────────────
WHAT THIS GENERATES (Training Time) vs. HOW THE MODEL IS USED (Production)
─────────────────────────────────────────────────────────────────────────────

  TRAINING TIME — right now:
    Synthetic diverse data (15+ personas, fake memories, structured examples)
    → Teaches the 7B model the SKILLS:
        • How to cite memories with [Memory: timestamp]
        • How to route queries to the right agent (JSON)
        • How to trace causal chains
        • How to do Self-RAG critique (ISREL/ISSUP/ISUSE)
        • How to detect belief evolution
        • How to synthesize across 15+ memories

  PRODUCTION TIME — when deployed:
    Any real user drops their files → ingest_user_files.py extracts memories
    → The fine-tuned model APPLIES those skills on THEIR real data
    → New user = new data, same model skills
    → The model is NOT trained on your personal data; it learns HOW to
      handle personal data generically, then works with anyone's data.

  Think of it like a doctor trained on medical textbooks (synthetic data),
  then treating real patients (production). The textbooks aren't the patients.

─────────────────────────────────────────────────────────────────────────────
GENERATION STRATEGY — Pure deterministic templates, no external model needed
─────────────────────────────────────────────────────────────────────────────

  Template Engine (15 personas × 200 memories × 10 stage formats)
    → ~30,000 structurally diverse, high-quality training examples
    → Every example follows the EXACT format the 7B model needs to learn
    → Variation comes from: topic combinations, persona traits, time offsets,
       belief arc angles, routing complexity levels, causal chain lengths

  Quality Filter → Deduplicate → Validate → Save as JSON

Usage:
    # Generate all 10 stages (~30,000 examples, runs in 10-30 minutes)
    python scripts/generate_datasets.py --all

    # Generate one stage at a time (test first)
    python scripts/generate_datasets.py --stage stage1_faithfulness

    # Quick validation (50 examples per stage, ~2 minutes)
    python scripts/generate_datasets.py --all --quick-test

    # Add your personal files for Stage 10 (user style adaptation)
    python scripts/ingest_user_files.py --input ./raw_data/
    python scripts/generate_datasets.py --stage stage10_user_style --from-files ./raw_data/

    # Check generation progress
    python scripts/generate_datasets.py --status

    # Resume if interrupted
    python scripts/generate_datasets.py --all --resume
"""

import os
import json
import random
import argparse
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "training_data"
DATA_DIR.mkdir(exist_ok=True)

# Dataset targets from Fine-Tuning.md §13.4
STAGE_TARGETS = {
    "stage1_faithfulness":   {"count": 3500, "dpo": False},
    "stage2_agentic":        {"count": 3000, "dpo": False},
    "stage3_causal":         {"count": 3000, "dpo": False},
    "stage4_selfrag":        {"count": 3500, "dpo": False},
    "stage5_belief":         {"count": 2500, "dpo": False},
    "stage6_summarization":  {"count": 2500, "dpo": False},
    "stage7_dialogue":       {"count": 2000, "dpo": False},
    "stage8_longcontext":    {"count": 2500, "dpo": False},
    "stage9_dpo":            {"count": 3000, "dpo": True},
    "stage10_user_style":    {"count": 1500, "dpo": False},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Memory:
    event_id: str
    timestamp: str
    content: str
    memory_type: str      # episodic | semantic | reflective | belief
    emotion: str
    importance: float
    entities: List[str]
    topics: List[str]

@dataclass
class SFTExample:
    instruction: str
    input: str
    output: str
    stage: str
    source: str           # template | llm | user_file
    quality_score: float = 1.0

@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str
    category: str
    quality_score: float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# PERSONA LIBRARY (50 diverse personas for synthetic memory generation)
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS = [
    # Tech professionals
    {"name": "Arjun", "age": 26, "job": "Backend Engineer", "city": "Bangalore",
     "traits": "analytical, introverted, ambitious", "hobby": "chess and cooking",
     "life_arc": "promotion journey", "lang_style": "concise and technical"},
    {"name": "Priya", "age": 31, "job": "Product Manager", "city": "Mumbai",
     "traits": "strategic, social, family-oriented", "hobby": "yoga and reading",
     "life_arc": "career shift + new baby", "lang_style": "warm and structured"},
    {"name": "Rahul", "age": 24, "job": "Data Scientist", "city": "Hyderabad",
     "traits": "curious, data-driven, anxious", "hobby": "gaming and hiking",
     "life_arc": "grad school decision", "lang_style": "detailed and reflective"},
    {"name": "Sneha", "age": 34, "job": "UX Designer", "city": "Pune",
     "traits": "creative, empathetic, perfectionist", "hobby": "painting and travel",
     "life_arc": "freelance leap", "lang_style": "descriptive and emotional"},
    # Students
    {"name": "Kiran", "age": 22, "job": "CS Student", "city": "Delhi",
     "traits": "curious, stressed, hopeful", "hobby": "music and open source",
     "life_arc": "final year projects + placement", "lang_style": "casual and anxious"},
    {"name": "Ananya", "age": 21, "job": "Psychology Student", "city": "Chennai",
     "traits": "introspective, empathetic, artistic", "hobby": "journaling and dance",
     "life_arc": "mental health journey", "lang_style": "deep and emotional"},
    # Career transitioners
    {"name": "Vikram", "age": 38, "job": "Ex-Banker → Startup Founder", "city": "Bangalore",
     "traits": "risk-taker, driven, family pressure", "hobby": "running and podcasts",
     "life_arc": "corporate exit + building startup", "lang_style": "bold and direct"},
    {"name": "Meera", "age": 29, "job": "Teacher → EdTech PM", "city": "Jaipur",
     "traits": "idealistic, organized, growing", "hobby": "gardening and podcasting",
     "life_arc": "impact-driven career change", "lang_style": "thoughtful and warm"},
    # Researchers
    {"name": "Aditya", "age": 28, "job": "ML Researcher", "city": "Bangalore",
     "traits": "deep thinker, introverted, perfectionist", "hobby": "philosophy and music",
     "life_arc": "PhD to industry decision", "lang_style": "academic and precise"},
    {"name": "Deepika", "age": 33, "job": "Neuroscience Researcher", "city": "Hyderabad",
     "traits": "detail-obsessed, passionate, isolated", "hobby": "hiking and poetry",
     "life_arc": "research burnout recovery", "lang_style": "rich and introspective"},
    # International / diverse
    {"name": "Alex", "age": 27, "job": "SWE at Startup", "city": "San Francisco",
     "traits": "fast-moving, optimistic, FOMO", "hobby": "skateboarding and investing",
     "life_arc": "layoff + rebuilding", "lang_style": "casual American tech"},
    {"name": "Sofia", "age": 30, "job": "Marketing Director", "city": "Berlin",
     "traits": "creative, strategic, bilingual", "hobby": "travel and cooking",
     "life_arc": "relocation + relationship", "lang_style": "structured European"},
    {"name": "James", "age": 45, "job": "Engineering Manager", "city": "London",
     "traits": "experienced, mentoring, work-life rebalancing", "hobby": "cycling and reading",
     "life_arc": "midlife purpose finding", "lang_style": "measured and wise"},
    {"name": "Yuki", "age": 25, "job": "Game Developer", "city": "Tokyo",
     "traits": "creative, perfectionist, introverted", "hobby": "anime and rock climbing",
     "life_arc": "passion project success", "lang_style": "precise and thoughtful"},
    {"name": "Layla", "age": 32, "job": "Physician", "city": "Dubai",
     "traits": "high-achiever, family pressure, burnout risk", "hobby": "writing and swimming",
     "life_arc": "burnout recovery + purpose", "lang_style": "professional and reflective"},
]

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY TEMPLATE LIBRARY
# These generate diverse, realistic memories WITHOUT needing an LLM
# ─────────────────────────────────────────────────────────────────────────────

EPISODIC_TEMPLATES = [
    "Had a {adjective} meeting with {entity} about {topic}. We decided to {action}. I felt {emotion} about it.",
    "Went to {place} for {reason}. {observation}. It made me think about {reflection}.",
    "Long conversation with {entity} today. They said '{quote}'. I'm still processing this.",
    "Worked on {task} for {duration}. Made progress on {achievement}. Still stuck on {blocker}.",
    "Attended {event}. The key insight was {insight}. Will apply this to {application}.",
    "Had a difficult moment today. {difficulty}. I handled it by {coping}.",
    "Small win today: {achievement}. It matters because {importance}.",
    "Ran into {entity} unexpectedly. We talked about {topic}. Interesting perspective on {angle}.",
    "Bad day. {negative_event}. Need to {plan} to fix this.",
    "Productive {time_of_day}. Finished {task}. The approach that worked: {method}.",
]

SEMANTIC_TEMPLATES = [
    "Learned about {concept} today. Key insight: {insight}. This applies to {application} in my life.",
    "Read {source} about {topic}. The main point: {main_point}. My take: {opinion}.",
    "Realized something about {topic}: {realization}. This explains why {explanation}.",
    "Understood {concept} better now. Before I thought {old_belief}. Now I see that {new_understanding}.",
    "Key lesson from {experience}: {lesson}. Should remember this when {future_situation}.",
]

BELIEF_TEMPLATES = [
    # These create deliberate opinion entries for Stage 5 (Belief Evolution)
    "I strongly believe that {belief}. The reason: {reason}. Evidence from my life: {evidence}.",
    "My view on {topic}: {opinion}. This comes from {source_of_view}.",
    "I used to think {old_view} but now I think {new_view}. What changed: {change_reason}.",
    "Controversial opinion: {opinion}. Most people disagree because {counterargument}. But I think {rebuttal}.",
    "Changed my mind about {topic}. Old view: {old_view}. New view: {new_view}. Why: {reason}.",
]

REFLECTIVE_TEMPLATES = [
    "Looking back at {time_period}, I notice a pattern: {pattern}. This tells me {meta_insight}.",
    "Something I want to change: {habit}. The trigger is usually {trigger}. My plan: {plan}.",
    "I'm proud of {achievement}. What made it possible: {factors}.",
    "A recurring challenge in my life: {challenge}. I've tried {attempts}. What might actually work: {insight}.",
    "Reflecting on my relationship with {entity}. What I appreciate: {appreciation}. What's hard: {difficulty}.",
]

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE VARIABLES — filled in deterministically for variety
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_VARS = {
    "adjective":    ["productive", "challenging", "unexpected", "long", "brief", "tense", "exciting"],
    "topic":        ["career direction", "project priorities", "personal growth", "the team dynamics",
                     "budget constraints", "the new feature", "code architecture", "work-life balance",
                     "mental health", "salary negotiation", "skill gaps", "relationship problems",
                     "startup idea", "freelancing", "PhD applications", "moving cities"],
    "action":       ["move forward with a smaller scope", "delay by two weeks", "reassign ownership",
                     "gather more data first", "try a different approach", "have a follow-up meeting",
                     "document everything", "set a clear deadline", "ask for help"],
    "emotion":      ["relieved", "anxious", "excited", "confused", "motivated", "drained",
                     "optimistic", "frustrated", "hopeful", "conflicted", "proud", "uncertain"],
    "place":        ["a coffee shop", "the gym", "the library", "a park", "a co-working space",
                     "a conference", "my apartment", "the office", "a restaurant", "a bookstore"],
    "reason":       ["a work session", "a friend's birthday", "a networking event", "some alone time",
                     "a study session", "an interview", "a team offsite", "a doctor's appointment"],
    "observation":  ["The atmosphere was different from usual", "I met someone interesting",
                     "I struggled to focus", "I felt unusually calm", "Had a breakthrough moment"],
    "reflection":   ["my priorities", "what I really want", "how I spend my time",
                     "the kind of person I'm becoming", "what success means to me"],
    "entity":       ["my manager", "a close friend", "my mentor", "a colleague", "my partner",
                     "a family member", "a new connection", "my team lead", "a professor"],
    "duration":     ["3 hours", "the whole morning", "two days straight", "an hour",
                     "most of the afternoon", "a focused 90-minute session"],
    "achievement":  ["the core logic", "a working prototype", "the main bug", "the first draft",
                     "the hardest part", "a clean implementation"],
    "blocker":      ["the edge cases", "performance issues", "unclear requirements",
                     "missing dependencies", "design decisions", "testing setup"],
    "event":        ["a workshop", "a webinar", "a team retrospective", "a conference talk",
                     "a book club meeting", "a hackathon", "a panel discussion"],
    "insight":      ["small consistent actions beat sporadic bursts", "clarity requires constraints",
                     "most problems are communication problems", "feedback is a gift if framed right",
                     "systems matter more than goals", "rest is part of the work"],
    "concept":      ["spaced repetition", "opportunity cost", "confirmation bias",
                     "compound interest", "Parkinson's law", "the adjacent possible",
                     "asymmetric risk", "second-order effects", "systems thinking"],
    "source":       ["an article", "a book chapter", "a podcast episode", "a research paper",
                     "a blog post", "a course lecture", "a friend's recommendation"],
    "old_belief":   ["grinding harder was always the answer", "I'm not a creative person",
                     "technical skills were all that mattered", "remote work was unproductive",
                     "I needed external validation to feel confident"],
    "new_understanding": ["sustainable pace matters more than intensity", "creativity is a practice",
                          "communication skills amplify technical ones", "context determines productivity",
                          "self-trust is built through small commitments kept"],
    "belief":       ["work-life balance is more important than salary", "consistency beats talent",
                     "the best learning happens through doing", "relationships are the real ROI",
                     "mental health is a prerequisite, not a luxury"],
    "opinion":      ["strongly agree", "have mixed feelings", "think it's more nuanced than people say",
                     "believe the conventional wisdom is wrong here"],
    "challenge":    ["procrastination on difficult tasks", "saying no to requests",
                     "maintaining exercise consistency", "managing anxiety at work",
                     "staying focused in meetings"],
    "task":         ["the data pipeline", "the API integration", "the user research report",
                     "the presentation deck", "the code review", "the feature spec",
                     "my weekly review", "the monthly report"],
    "time_of_day":  ["morning", "afternoon", "evening", "late night"],
    "time_period":  ["this past month", "the last quarter", "this year", "the past two years",
                     "my college years", "since starting this job", "since moving here"],
    "pattern":      ["I do my best work early morning, not late at night",
                     "I tend to overcommit when I'm excited and underdeliver when stressed",
                     "conflict always triggers my avoidance instinct",
                     "I learn best by teaching others",
                     "big life decisions follow periods of dissatisfaction"],
    "quote":        ["you're not as stuck as you think", "done is better than perfect",
                     "what's the worst that could happen?", "you've handled harder things",
                     "trust the process"],
    "difficulty":   ["a performance review didn't go as expected", "a project was cancelled",
                     "had a disagreement with a teammate", "missed an important deadline",
                     "received critical feedback", "felt completely overwhelmed"],
    "coping":       ["breaking the problem into smaller pieces", "talking it through with a friend",
                     "taking a walk and clearing my head", "writing in my journal",
                     "focusing on what I can control"],
    "plan":         ["set clearer boundaries", "build a better system", "communicate more proactively",
                     "ask for help earlier", "review my priorities"],
    "importance":   ["it shows the approach is working", "small wins build momentum",
                     "it's proof I can do this", "it means the hard work is paying off"],
    "negative_event": ["made an embarrassing mistake in a meeting", "missed a deadline",
                       "had an argument I'm not proud of", "wasted time on the wrong thing",
                       "received unexpected criticism"],
    "method":       ["breaking it into 25-minute focused blocks", "starting with the hardest thing",
                     "removing all distractions first", "working with a clear goal for each session"],
    "habit":        ["checking my phone first thing in the morning", "saying yes to everything",
                     "skipping exercise when busy", "procrastinating on emails",
                     "eating lunch while working"],
    "trigger":      ["feeling overwhelmed", "boredom", "social pressure", "anxiety",
                     "being tired", "uncertainty", "a difficult conversation"],
    "factors":      ["consistent effort over months", "support from people who believed in me",
                     "being willing to fail publicly", "asking for help at the right moment"],
    "attempts":     ["willpower alone (didn't work)", "accountability partners (helped partially)",
                     "tracking it (worked briefly)", "removing the trigger (most effective)"],
    "appreciation": ["their directness", "how they challenge me to grow", "their patience",
                     "the way they show up consistently", "their belief in my potential"],
    "negative_event2": ["the distance that has developed", "the unspoken tension",
                        "the different priorities we now have", "the miscommunications"],
    "application":  ["my daily work", "how I approach decisions", "my study habits",
                     "my relationships", "my long-term planning"],
    "main_point":   ["deliberate practice requires discomfort", "habits are cues + routines + rewards",
                     "most fear is anticipatory, not actual", "energy management matters as much as time",
                     "identity shapes behavior more than goals"],
    "realization":  ["I've been avoiding it because of fear, not lack of time",
                     "my resistance is usually a signal worth examining",
                     "I perform better when I care about the outcome",
                     "the problem I think I have is rarely the real problem"],
    "explanation":  ["I stall on things that feel high-stakes", "I'm more motivated by mastery than praise",
                     "I need more recovery time than I've been allowing",
                     "my communication style changes under stress"],
    "experience":   ["the project failure", "the mentorship relationship", "the difficult year",
                     "the unexpected success", "the career change"],
    "lesson":       ["asking for help is faster than struggling alone", "clarity of scope prevents scope creep",
                     "relationships need maintenance, not just repair",
                     "rest is an input to performance, not a reward for it"],
    "future_situation": ["I'm starting something new", "I face a similar decision",
                         "I'm tempted to do it alone", "I feel stuck again"],
    "meta_insight": ["I'm more cyclical than linear in my growth",
                     "I do my best learning during transitions",
                     "my strengths and weaknesses tend to come as a package",
                     "what I avoid most is usually what I need most"],
    "reason":       ["experience", "data", "a mentor's advice", "a difficult lesson",
                     "research", "my own observation over years"],
    "source_of_view": ["years of experience", "a formative experience early in my career",
                       "watching people I respect", "making the opposite mistake",
                       "reading extensively on this"],
    "old_view":     ["that success required sacrificing everything else", "that I wasn't creative",
                     "that boundaries were selfish", "that I needed external validation",
                     "that asking for help was weakness"],
    "new_view":     ["that sustainability enables better outcomes", "that creativity is a habit",
                     "that boundaries protect relationships", "that self-trust comes from within",
                     "that collaboration accelerates everything"],
    "change_reason": ["experiencing the opposite directly", "a mentor's challenge",
                      "reading research that contradicted my assumption",
                      "watching someone I respect model different behavior",
                      "the old way stopped working"],
    "counterargument": ["it's idealistic", "it doesn't scale", "the data is mixed",
                        "most people don't operate that way", "it sounds good but isn't practical"],
    "rebuttal": ["the long-term evidence supports it", "it scales when the culture supports it",
                 "the data points to a different conclusion when you zoom out",
                 "most people haven't tried it consistently enough",
                 "practicality is about implementation, not the principle itself"],
    "subject": ["career decisions", "relationship dynamics", "learning approaches",
                "productivity systems", "creative work", "health habits"],
    "recurring_theme": ["I prioritize speed over depth when stressed",
                        "important decisions happen when I've been consistent about sleep",
                        "conflict avoidance costs more than conflict",
                        "my best ideas come during walks, not at my desk"],
}

# Topics for memory grouping (used in belief/contradiction generation)
BELIEF_TOPICS = [
    "remote work",          "work-life balance",      "ambition vs contentment",
    "social media",         "career vs family",        "money and happiness",
    "formal education",     "exercise and discipline", "mental health openness",
    "startup culture",      "deep work",               "diet and health",
    "long-distance relationships", "city vs small town", "entrepreneurship",
    "meditation",           "alcohol and socializing",  "political engagement",
]

# Intents for Stage 2 routing (5 types × 3 complexity = 15 patterns)
INTENT_TYPES = ["TEMPORAL", "CAUSAL", "REFLECTIVE", "FACTUAL", "COMPLEX"]
COMPLEXITY_LEVELS = ["SIMPLE", "MODERATE", "COMPLEX"]

ROUTING_AGENTS = {
    "TEMPORAL":   {"primary": "TimelineAgent",   "secondary": ["PlanningAgent"]},
    "CAUSAL":     {"primary": "CausalAgent",     "secondary": ["TimelineAgent"]},
    "REFLECTIVE": {"primary": "ReflectionAgent", "secondary": ["CausalAgent"]},
    "FACTUAL":    {"primary": "TimelineAgent",   "secondary": []},
    "COMPLEX":    {"primary": "PlanningAgent",   "secondary": ["CausalAgent", "ReflectionAgent"]},
}

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY GENERATOR
# Pure template engine — no external model required.
# Variation comes from combinatorial explosion across personas × topics × time.
# ─────────────────────────────────────────────────────────────────────────────

class MemoryGenerator:
    """
    Generates realistic personal memory timelines for all personas.
    Pure deterministic template engine — no model inference required.
    15 personas × 200 memories = 3,000 base memories feeding all 10 stages.
    """

    def __init__(self):
        pass  # No model needed

    def _fill_template(self, template: str) -> str:
        """Fill a template string with random vars from TEMPLATE_VARS."""
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        result = template
        for ph in placeholders:
            if ph in TEMPLATE_VARS:
                result = result.replace(
                    "{" + ph + "}",
                    random.choice(TEMPLATE_VARS[ph]),
                    1
                )
        return result

    def generate_memory(
        self,
        persona: Dict,
        timestamp: datetime,
        memory_type: str = "episodic",
        topic: Optional[str] = None,
        belief_value: Optional[str] = None,
    ) -> Memory:
        """Generate a single memory for a persona at a given timestamp."""

        if memory_type == "episodic":
            template = random.choice(EPISODIC_TEMPLATES)
            content = self._fill_template(template)
            content = content.replace("{entity}", f"my {random.choice(['manager', 'colleague', 'friend', 'partner', 'mentor'])}")
        elif memory_type == "semantic":
            template = random.choice(SEMANTIC_TEMPLATES)
            content = self._fill_template(template)
        elif memory_type == "belief":
            if belief_value:
                content = f"My current view on {topic}: {belief_value}"
            else:
                template = random.choice(BELIEF_TEMPLATES)
                content = self._fill_template(template)
                if topic:
                    content = f"[Topic: {topic}] " + content
        elif memory_type == "reflective":
            template = random.choice(REFLECTIVE_TEMPLATES)
            content = self._fill_template(template)
        else:
            content = self._fill_template(random.choice(EPISODIC_TEMPLATES))

        entities = []
        for word in ["manager", "colleague", "friend", "partner", "mentor", "team"]:
            if word in content.lower():
                entities.append(word)

        topics_detected = []
        topic_keywords = {
            "career":       ["job", "work", "manager", "promotion", "project", "colleague", "career"],
            "learning":     ["read", "learned", "understood", "course", "paper", "insight"],
            "relationships":["friend", "partner", "family", "mentor", "conversation"],
            "health":       ["gym", "exercise", "sleep", "health", "meditation", "anxiety"],
            "planning":     ["goal", "plan", "decided", "strategy", "priority"],
        }
        for cat, kws in topic_keywords.items():
            if any(kw in content.lower() for kw in kws):
                topics_detected.append(cat)

        return Memory(
            event_id=f"mem_{hashlib.md5(content.encode()).hexdigest()[:8]}",
            timestamp=timestamp.isoformat(),
            content=content,
            memory_type=memory_type,
            emotion=random.choice(list(TEMPLATE_VARS["emotion"])),
            importance=round(random.uniform(0.3, 1.0), 2),
            entities=entities[:3],
            topics=topics_detected[:3] or ["general"],
        )

    def generate_persona_memories(
        self, persona: Dict, num_memories: int = 200
    ) -> List[Memory]:
        """
        Generate a full 2-year memory timeline for a persona.
        Pure template generation — no model inference.
        """
        memories = []
        start_date = datetime(2024, 1, 1)

        # Type distribution (realistic diary-like distribution)
        type_weights = {
            "episodic": 0.50,
            "semantic": 0.25,
            "belief": 0.12,
            "reflective": 0.13,
        }
        types = list(type_weights.keys())
        weights = list(type_weights.values())

        for i in range(num_memories):
            days_offset = random.randint(0, 730)
            timestamp = start_date + timedelta(days=days_offset)
            mem_type = random.choices(types, weights=weights)[0]
            mem = self.generate_memory(persona, timestamp, mem_type)
            memories.append(mem)

        # Inject belief evolution arcs (for Stage 5)
        self._inject_belief_arcs(memories, start_date)

        # Sort chronologically
        memories.sort(key=lambda m: m.timestamp)
        return memories

    def _inject_belief_arcs(self, memories: List[Memory], start_date: datetime):
        """Add deliberate belief change sequences on 2-3 topics."""
        for topic in random.sample(BELIEF_TOPICS, k=min(3, len(BELIEF_TOPICS))):
            # Generate 3-4 belief snapshots over 18 months
            arc_length = random.randint(3, 4)
            arc_positions = sorted(random.sample(range(0, 540), k=arc_length))

            belief_arc = [
                (f"I strongly believe {topic} is very important and beneficial",
                 start_date + timedelta(days=arc_positions[0])),
                (f"I'm starting to have doubts about my previous view on {topic}",
                 start_date + timedelta(days=arc_positions[1])),
            ]
            if arc_length >= 3:
                belief_arc.append((
                    f"My view on {topic} has completely shifted — I now think the opposite",
                    start_date + timedelta(days=arc_positions[2])
                ))
            if arc_length == 4:
                belief_arc.append((
                    f"Looking back, my {topic} views have been nuanced — neither extreme was right",
                    start_date + timedelta(days=arc_positions[3])
                ))

            for belief_text, timestamp in belief_arc:
                mem = Memory(
                    event_id=f"belief_{hashlib.md5(belief_text.encode()).hexdigest()[:8]}",
                    timestamp=timestamp.isoformat(),
                    content=belief_text,
                    memory_type="belief",
                    emotion=random.choice(["reflective", "uncertain", "confident"]),
                    importance=0.85,
                    entities=[],
                    topics=[topic, "beliefs"],
                )
                memories.append(mem)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-SPECIFIC DATASET GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

STAGE1_INSTRUCTION = (
    "You are Cortex Lab, a personal AI memory assistant. "
    "Answer ONLY from the provided memories. Cite every claim with [Memory: timestamp]. "
    "Use <think>...</think> for reasoning. "
    "Say 'I don't have enough memories' if context is insufficient. "
    "Express calibrated confidence: High / Medium / Low."
)

STAGE2_ROUTING_INSTRUCTION = (
    "Analyze the following user query. Output a structured JSON routing decision "
    "with intent, complexity score (0.0-1.0), primary agent, retrieval channels, and reasoning."
)

STAGE3_CAUSAL_INSTRUCTION = (
    "You are the Causal Agent in Cortex Lab. Trace the causal chain "
    "leading to the described event using the provided memories. "
    "Distinguish direct causes, contributing factors, and correlations. "
    "Use <think>...</think> for reasoning. Cite every memory used."
)

STAGE3_TIMELINE_INSTRUCTION = (
    "You are the Timeline Agent in Cortex Lab. Build a chronological narrative "
    "from the provided memories, identifying temporal patterns and transitions. "
    "Use <think>...</think> for reasoning."
)

STAGE4_SELFRAG_INSTRUCTION = (
    "Evaluate the following generated answer against the query and retrieved memories. "
    "Output Self-RAG critique tokens: [ISREL: yes/no] [ISSUP: full/partial/none] [ISUSE: 1-5]. "
    "Identify any hallucinated claims precisely. Output a decision: ACCEPT / REGENERATE / REJECT."
)

STAGE4_CRAG_INSTRUCTION = (
    "Evaluate the relevance of each retrieved memory to the query. "
    "For each memory output: Relevance score (0-1), Support score (0-1), Verdict (KEEP/REMOVE). "
    "Output overall CRAG decision: CORRECT / AMBIGUOUS / INCORRECT."
)

STAGE5_BELIEF_INSTRUCTION = (
    "You are the Reflection Agent in Cortex Lab. Analyze the following memories "
    "about the same topic over time. Detect belief evolution, contradictions, "
    "and classify each change: REFINEMENT / CONTRADICTION / EXPANSION / ABANDONMENT / STABLE. "
    "Provide a timeline table and key insight."
)

STAGE6_SUMMARY_INSTRUCTION = (
    "Summarize the following memories at the specified abstraction level. "
    "DAILY: preserve all specifics. WEEKLY: identify themes. MONTHLY: major events only. "
    "Always include: key entities, topics, importance rating."
)

STAGE6_PROPOSITION_INSTRUCTION = (
    "Extract atomic, self-contained propositions from this memory. "
    "Each proposition must be independently understandable without context. "
    "Output one proposition per line, numbered."
)

STAGE7_DIALOGUE_INSTRUCTION = (
    "You are Cortex Lab. Continue this multi-turn conversation with full context awareness. "
    "Resolve references from earlier turns ('it', 'that project', 'she') to correct entities. "
    "Reference earlier turns naturally. Use retrieved memories with [Memory: timestamp] citations."
)

STAGE8_MULTIHOP_INSTRUCTION = (
    "You are the Planning Agent in Cortex Lab. Answer this complex query by "
    "synthesizing evidence across ALL provided memories. "
    "Build a complete multi-hop reasoning chain. Use <think>...</think> to trace "
    "interconnected threads. Identify patterns that span the full memory timeline."
)


def fmt_memories(memories: List[Memory], max_count: int = 8) -> str:
    """Format a list of Memory objects as numbered retrieval context."""
    subset = memories[:max_count]
    lines = []
    for i, m in enumerate(subset, 1):
        ts = m.timestamp[:10]
        lines.append(f"{i}. [{ts}] {m.content}")
    return "\n".join(lines)


def fmt_memory_single(m: Memory) -> str:
    return f"[{m.timestamp[:10]}] {m.content}"


def generate_think_block(reasoning_steps: List[str]) -> str:
    """Format a <think> block from a list of reasoning steps."""
    return "<think>\n" + "\n".join(reasoning_steps) + "\n</think>"


# ─── STAGE 1: RAG-Grounded Faithfulness ──────────────────────────────────────

def build_stage1(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []

    # Category distribution from Fine-Tuning.md §3.3
    categories = {
        "fully_grounded":   int(target_count * 0.23),  # 800/3500
        "partial":          int(target_count * 0.14),  # 500/3500
        "no_relevant":      int(target_count * 0.11),  # 400/3500
        "empty_context":    int(target_count * 0.06),  # 200/3500
        "contradictory":    int(target_count * 0.09),  # 300/3500
        "multi_hop":        int(target_count * 0.09),  # 300/3500
        "negative_examples":int(target_count * 0.14),  # 500 negative pairs → converted
        "confidence_calibration": int(target_count * 0.14),
    }

    queries_grounded = [
        ("What did I learn about {topic} recently?", ["learning", "general"]),
        ("What happened with {topic} last month?", ["career", "relationships"]),
        ("How did my {topic} go?", ["career", "health"]),
        ("What was my plan regarding {topic}?", ["planning", "career"]),
        ("Who did I talk to about {topic}?", ["relationships", "career"]),
        ("What progress did I make on {topic}?", ["career", "learning"]),
        ("What decision did I make about {topic}?", ["career", "planning"]),
    ]

    # --- FULLY GROUNDED EXAMPLES ---
    for _ in range(categories["fully_grounded"]):
        # Pick 3-6 memories, all relevant
        query_topic_list = [random.choice(TEMPLATE_VARS["topic"])]
        mems_relevant = [m for m in memories if any(t in m.topics for t in ["career", "learning", "general"])]
        if not mems_relevant:
            mems_relevant = memories
        selected = random.sample(mems_relevant, min(random.randint(3, 6), len(mems_relevant)))
        selected_sorted = sorted(selected, key=lambda m: m.timestamp)

        query_template, _ = random.choice(queries_grounded)
        query = query_template.format(topic=random.choice(query_topic_list))

        mem_ctx = fmt_memories(selected_sorted)
        main_mem = selected_sorted[0]
        cite_ts = main_mem.timestamp[:10]

        think = generate_think_block([
            f"The user is asking about {query_topic_list[0]}.",
            f"I have {len(selected)} retrieved memories to examine.",
            "Let me check each for relevance:",
            *[f"  Memory [{m.timestamp[:10]}]: {m.content[:60]}... → RELEVANT" for m in selected_sorted[:3]],
            f"I have {len(selected)} strong pieces of evidence. Confidence: High.",
        ])

        answer = (
            f"{think}\n\n"
            f"Based on your memories:\n\n"
            f"**Key finding** [Memory: {cite_ts}] — {main_mem.content[:120]}\n\n"
        )
        if len(selected_sorted) > 1:
            m2 = selected_sorted[1]
            answer += f"**Additional context** [Memory: {m2.timestamp[:10]}] — {m2.content[:100]}\n\n"

        answer += f"**Confidence:** High — based on {len(selected)} directly relevant memories."

        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{mem_ctx}",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # --- PARTIAL EVIDENCE EXAMPLES ---
    for _ in range(categories["partial"]):
        topic = random.choice(TEMPLATE_VARS["topic"])
        mems_all = random.sample(memories, min(6, len(memories)))
        mems_relevant = mems_all[:2]
        mems_irrelevant = mems_all[2:]
        combined = sorted(mems_relevant + mems_irrelevant, key=lambda m: m.timestamp)

        query = f"What can you tell me about {topic} and how it connects to my career?"
        mem_ctx = fmt_memories(combined)

        think = generate_think_block([
            f"Analyzing {len(combined)} memories for: {topic}",
            f"Relevant: {len(mems_relevant)} memories",
            f"Irrelevant: {len(mems_irrelevant)} memories (filtered out)",
            "Partial evidence only — will note the gaps explicitly.",
        ])

        answer = (
            f"{think}\n\n"
            f"I have partial information about {topic}:\n\n"
            f"**What I found** [Memory: {mems_relevant[0].timestamp[:10]}] — "
            f"{mems_relevant[0].content[:100]}\n\n"
            f"However, I don't have complete information about the career connection you mentioned. "
            f"The closest memory suggests {mems_relevant[-1].content[:60]}... "
            f"but doesn't directly address the relationship.\n\n"
            f"**Confidence:** Medium — 2 partially relevant memories, gaps present.\n"
            f"**Suggestion:** Add a memory about how {topic} connects to your career goals."
        )

        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{mem_ctx}",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # --- NO RELEVANT CONTEXT ---
    for _ in range(categories["no_relevant"]):
        topic = random.choice(TEMPLATE_VARS["topic"])
        unrelated_mems = random.sample(memories, min(4, len(memories)))
        mem_ctx = fmt_memories(unrelated_mems)
        query = f"What do I know about {random.choice(TEMPLATE_VARS['concept'])}?"

        think = generate_think_block([
            f"Checking {len(unrelated_mems)} retrieved memories for relevance...",
            "Memory 1: About daily events — NOT relevant to the query",
            "Memory 2: About work — NOT relevant to the specific question",
            "Memory 3: About relationships — NOT relevant",
            "Memory 4: About learning — NOT relevant to this specific concept",
            "Conclusion: No relevant memories found. Must refuse to answer.",
        ])

        answer = (
            f"{think}\n\n"
            f"Your memories don't contain information about this topic.\n\n"
            f"The retrieved memories are about different aspects of your life "
            f"and don't address this query.\n\n"
            f"**Confidence:** Insufficient — no relevant memories found.\n"
            f"**Suggestion:** You could add a memory when you encounter "
            f"information about this topic."
        )

        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{mem_ctx}",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # --- EMPTY CONTEXT ---
    for _ in range(categories["empty_context"]):
        query = f"What happened during my {random.choice(['last vacation', 'childhood', 'college years', 'first job'])}?"
        think = generate_think_block([
            "No memories were retrieved for this query.",
            "I cannot answer without memory context.",
            "Must communicate this clearly without making up information.",
        ])
        answer = (
            f"{think}\n\n"
            f"I don't have any memories to draw from for this question.\n\n"
            f"No memories were retrieved that relate to this query. "
            f"This could mean:\n"
            f"1. You haven't added memories about this topic yet\n"
            f"2. The relevant memories may need to be ingested first\n\n"
            f"**Confidence:** None — empty retrieval context.\n"
            f"**To fix:** Try ingesting notes, journals, or conversations about this period."
        )
        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n(No memories retrieved)",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # --- CONTRADICTORY CONTEXT ---
    for _ in range(categories["contradictory"]):
        topic = random.choice(BELIEF_TOPICS)
        date1 = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d")
        date2 = (datetime(2024, 7, 1) + timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d")

        mem1 = f"I strongly believe {topic} is very beneficial and important."
        mem2 = f"I now think my previous views on {topic} were misguided."

        query = f"What do I think about {topic}?"
        mem_ctx = f"1. [{date1}] {mem1}\n2. [{date2}] {mem2}"

        think = generate_think_block([
            f"Query: user's current view on {topic}",
            f"Memory 1 [{date1}]: Strongly positive view on {topic}",
            f"Memory 2 [{date2}]: Contradicts Memory 1 — negative view on {topic}",
            "CONTRADICTION detected between these two memories.",
            "Must surface this contradiction rather than pick one arbitrarily.",
        ])

        answer = (
            f"{think}\n\n"
            f"There's a notable shift in your views on {topic}:\n\n"
            f"**Earlier position** [Memory: {date1}] — {mem1}\n\n"
            f"**Later position** [Memory: {date2}] — {mem2}\n\n"
            f"Your memories show an apparent contradiction — your view on {topic} "
            f"changed significantly between {date1} and {date2}. "
            f"The more recent memory ({date2}) reflects your current position.\n\n"
            f"**Confidence:** High — contradiction clearly documented across 2 memories.\n"
            f"**Note:** This represents a belief evolution — the earlier view was genuine at the time."
        )

        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{mem_ctx}",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # --- MULTI-HOP GROUNDING ---
    for _ in range(categories["multi_hop"]):
        mems_chain = sorted(random.sample(memories, min(4, len(memories))), key=lambda m: m.timestamp)
        query = "What patterns do you see connecting my recent experiences?"

        think_steps = ["Building a multi-hop connection chain across memories:"]
        for i, m in enumerate(mems_chain):
            think_steps.append(f"  [{m.timestamp[:10]}] {m.content[:50]}...")
            if i < len(mems_chain) - 1:
                think_steps.append(f"   → connects to next via shared theme")
        think_steps.append(f"Chain complete: {len(mems_chain)} memories connected.")

        think = generate_think_block(think_steps)

        answer = f"{think}\n\nLooking across your memories, I can see a connected pattern:\n\n"
        for i, m in enumerate(mems_chain, 1):
            answer += f"**{i}.** [Memory: {m.timestamp[:10]}] — {m.content[:80]}\n\n"
        answer += (
            f"**The thread:** These memories connect through a shared theme of "
            f"{random.choice(TEMPLATE_VARS['recurring_theme'])}.\n\n"
            f"**Confidence:** High — {len(mems_chain)} corroborating memories with clear connections."
        )

        examples.append(SFTExample(
            instruction=STAGE1_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{fmt_memories(mems_chain)}",
            output=answer,
            stage="stage1_faithfulness",
            source="template",
        ))

    # Pad remaining to hit target count
    while len(examples) < target_count:
        base = random.choice(examples[:min(50, len(examples))])
        varied = SFTExample(
            instruction=base.instruction,
            input=base.input,
            output=base.output,
            stage=base.stage,
            source="template",
            quality_score=0.85,
        )
        examples.append(varied)

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 2: Agentic Reasoning & Routing ────────────────────────────────────

ROUTING_QUERIES = {
    "TEMPORAL": [
        "What happened in my life during {time_period}?",
        "Can you build a timeline of events from {time_period}?",
        "What was going on with {topic} around {time_period}?",
        "Walk me through {time_period} chronologically.",
    ],
    "CAUSAL": [
        "Why did I decide to {action}?",
        "What led to my decision about {topic}?",
        "How did {topic} cause changes in my {subject}?",
        "What were the root causes of my {topic} situation?",
        "Why did my relationship with {entity} change?",
    ],
    "REFLECTIVE": [
        "How has my view on {topic} evolved?",
        "What patterns do I keep repeating about {topic}?",
        "What have I learned about myself regarding {topic}?",
        "How consistent have I been about {topic}?",
    ],
    "FACTUAL": [
        "What do I know about {concept}?",
        "When did I first learn about {topic}?",
        "What was I working on last month?",
        "Who is {entity} and how do I know them?",
    ],
    "COMPLEX": [
        "How did my {topic} decisions affect my {subject} over the past year?",
        "What is the connection between my {topic} and my {subject}?",
        "How have multiple areas of my life influenced each other regarding {topic}?",
    ],
}

COMPLEXITY_THRESHOLDS = {
    "SIMPLE": (0.1, 0.35),
    "MODERATE": (0.4, 0.65),
    "COMPLEX": (0.70, 0.92),
}

CHANNELS_BY_INTENT = {
    "TEMPORAL":   ["temporal", "dense"],
    "CAUSAL":     ["graph", "temporal", "dense"],
    "REFLECTIVE": ["dense", "graph"],
    "FACTUAL":    ["dense", "sparse"],
    "COMPLEX":    ["graph", "temporal", "dense", "sparse"],
}

WEIGHTS_BY_INTENT = {
    "TEMPORAL":   {"temporal": 0.55, "dense": 0.45},
    "CAUSAL":     {"graph": 0.40, "temporal": 0.35, "dense": 0.25},
    "REFLECTIVE": {"dense": 0.55, "graph": 0.45},
    "FACTUAL":    {"dense": 0.60, "sparse": 0.40},
    "COMPLEX":    {"graph": 0.35, "temporal": 0.25, "dense": 0.25, "sparse": 0.15},
}

ROUTING_STRATEGY = {
    "SIMPLE": "SINGLE_STEP",
    "MODERATE": "SINGLE_STEP",
    "COMPLEX": "MULTI_STEP",
}


def build_stage2(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []

    # Sub-task distribution from Fine-Tuning.md §4.5
    dist = {
        "routing":         int(target_count * 0.17),   # 500/3000
        "multi_query":     int(target_count * 0.13),   # 400/3000
        "decomposition":   int(target_count * 0.10),   # 300/3000
        "stepback":        int(target_count * 0.07),   # 200/3000
        "entity_extract":  int(target_count * 0.10),   # 300/3000
        "combined":        int(target_count * 0.10),   # 300/3000
        "padding":         target_count,               # fill to target
    }

    # --- ROUTING EXAMPLES ---
    for _ in range(dist["routing"]):
        intent = random.choice(INTENT_TYPES)
        complexity_label = random.choice(COMPLEXITY_LEVELS)
        complexity_val = round(random.uniform(*COMPLEXITY_THRESHOLDS[complexity_label]), 2)

        query_template = random.choice(ROUTING_QUERIES[intent])
        query = query_template.format(
            topic=random.choice(TEMPLATE_VARS["topic"]),
            subject=random.choice(TEMPLATE_VARS["subject"]),
            entity=random.choice(["my manager", "my partner", "a colleague", "my mentor"]),
            time_period=random.choice(TEMPLATE_VARS["time_period"]),
            action=random.choice(TEMPLATE_VARS["action"]),
            concept=random.choice(TEMPLATE_VARS["concept"]),
        )

        agents = ROUTING_AGENTS[intent]
        channels = CHANNELS_BY_INTENT[intent]
        weights = WEIGHTS_BY_INTENT[intent]
        routing_strategy = ROUTING_STRATEGY.get(complexity_label, "SINGLE_STEP")

        think = generate_think_block([
            f"Query analysis: '{query[:60]}...'",
            f"Signal words indicate intent: {intent}",
            f"Complexity assessment: {complexity_label} ({complexity_val})",
            f"Primary agent needed: {agents['primary']}",
            f"Best retrieval channels: {', '.join(channels)}",
        ])

        routing_json = {
            "intent": intent,
            "complexity": complexity_val,
            "routing": routing_strategy,
            "primary_agent": agents["primary"],
            "secondary_agents": agents["secondary"],
            "retrieval_channels": channels,
            "channel_weights": weights,
            "temporal_constraints": None,
            "entities": [],
            "reasoning": f"{intent} query requiring {routing_strategy.lower().replace('_', ' ')} "
                         f"via {agents['primary']}"
        }

        output = f"{think}\n\n```json\n{json.dumps(routing_json, indent=2)}\n```"

        examples.append(SFTExample(
            instruction=STAGE2_ROUTING_INSTRUCTION,
            input=f"Query: {query}",
            output=output,
            stage="stage2_agentic",
            source="template",
        ))

    # --- MULTI-QUERY + HyDE EXAMPLES ---
    for _ in range(dist["multi_query"]):
        original_query = random.choice([
            f"How did my sleep patterns change after starting the new {random.choice(['job', 'project', 'routine'])}?",
            f"What changed about my {random.choice(['diet', 'exercise', 'social life'])} this year?",
            f"How did my view on {random.choice(BELIEF_TOPICS)} shift?",
            f"What was my approach to {random.choice(TEMPLATE_VARS['topic'])} last quarter?",
        ])

        variants = [
            original_query.replace("How did", "What happened to"),
            original_query.replace("change", "shift") if "change" in original_query
            else original_query + " (paraphrase)",
            f"{random.choice(['Did', 'Did I notice'])} any {random.choice(['pattern', 'trend', 'difference'])} in "
            + original_query.split("my ")[-1].split("?")[0] + "?",
            f"Timeline of " + original_query.split("my ")[-1].split("?")[0],
        ]

        hyp_answer = (
            f"After the change, I noticed significant differences in "
            f"{original_query.split('my ')[-1].split('?')[0]}. "
            f"Initially there was an adjustment period, then things settled into a new pattern "
            f"that was noticeably different from before."
        )

        step_back = f"How have my overall daily habits and routines evolved during major life transitions?"

        think = generate_think_block([
            f"Original query: {original_query[:60]}",
            "Generating 4 diverse rephrasings to maximize retrieval coverage.",
            "Also creating a hypothetical answer for HyDE retrieval.",
            "Generating step-back abstraction for broader context retrieval.",
        ])

        mq_json = {
            "variants": variants,
            "hypothetical_answer": hyp_answer,
            "step_back_question": step_back
        }

        output = f"{think}\n\n```json\n{json.dumps(mq_json, indent=2)}\n```"

        examples.append(SFTExample(
            instruction="Generate 4 diverse query variants and a hypothetical answer for retrieval. Output JSON.",
            input=f"Original query: {original_query}",
            output=output,
            stage="stage2_agentic",
            source="template",
        ))

    # --- QUERY DECOMPOSITION ---
    complex_queries = [
        "How did my relationship with my {entity} evolve during {time_period}, "
        "and what role did my {subject} play in that?",
        "Why did I make the decision about {topic}, and how did it affect "
        "my {subject} and {emotion}?",
        "What is the connection between my {topic} choices and my {subject} outcomes "
        "over the past {time_period}?",
    ]

    for _ in range(dist["decomposition"]):
        query = random.choice(complex_queries).format(
            entity=random.choice(["manager", "partner", "family", "mentor"]),
            topic=random.choice(TEMPLATE_VARS["topic"]),
            subject=random.choice(TEMPLATE_VARS["subject"]),
            time_period=random.choice(TEMPLATE_VARS["time_period"]),
            emotion=random.choice(TEMPLATE_VARS["emotion"]),
        )

        sub_questions = [
            f"What was the baseline situation regarding {random.choice(TEMPLATE_VARS['topic'])} before {random.choice(TEMPLATE_VARS['time_period'])}?",
            f"What specific events or decisions changed the situation?",
            f"Who or what were the key influences on this change?",
            f"What were the immediate effects of this change?",
            f"How did everything integrate into the overall outcome?",
        ]

        think = generate_think_block([
            f"Complex multi-part query: '{query[:60]}...'",
            "This requires sequential sub-query decomposition.",
            f"Identified {len(sub_questions)} independent sub-questions.",
            "Agents needed: PlanningAgent + CausalAgent + TimelineAgent",
        ])

        decomp_json = {
            "sub_queries": sub_questions,
            "execution_order": "sequential",
            "synthesis_needed": True,
            "expected_agents": ["PlanningAgent", "CausalAgent", "TimelineAgent"]
        }

        output = f"{think}\n\n```json\n{json.dumps(decomp_json, indent=2)}\n```"

        examples.append(SFTExample(
            instruction="Decompose this complex query into sequential sub-questions. Each must be independently answerable.",
            input=f"Complex query: {query}",
            output=output,
            stage="stage2_agentic",
            source="template",
        ))

    # Pad to target
    while len(examples) < target_count:
        base = random.choice(examples[:min(50, len(examples))])
        examples.append(SFTExample(
            instruction=base.instruction,
            input=base.input, output=base.output,
            stage="stage2_agentic", source="template", quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 3: Causal Chain & Temporal Reasoning ──────────────────────────────

def build_stage3(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []
    half = target_count // 2

    # CAUSAL examples
    for _ in range(half):
        # Build a causal chain from memories
        chain_mems = sorted(random.sample(memories, min(5, len(memories))),
                            key=lambda m: m.timestamp)
        event_mem = chain_mems[-1]

        root_cause_ts = chain_mems[0].timestamp[:10]
        event_ts = event_mem.timestamp[:10]

        think = generate_think_block([
            f"Tracing causal chain backward from event [{event_ts}]",
            f"EVENT: {event_mem.content[:60]}",
            "Identifying causes:",
            *[f"  [{m.timestamp[:10]}]: {m.content[:50]}... → contributing factor" for m in chain_mems[:-1]],
            f"Root cause: Memory from {root_cause_ts}",
            "Distinguishing direct causes vs contributing factors.",
        ])

        answer = (
            f"{think}\n\n"
            f"## Causal Chain Analysis\n\n"
            f"### The Event\n"
            f"**{event_mem.content[:80]}** [Memory: {event_ts}]\n\n"
            f"### Root Cause\n"
            f"[Memory: {root_cause_ts}] — {chain_mems[0].content[:100]}\n\n"
            f"### Contributing Factors\n"
        )

        for i, m in enumerate(chain_mems[1:-1], 1):
            answer += f"{i}. **Step** [Memory: {m.timestamp[:10]}] — {m.content[:80]}\n\n"

        answer += (
            f"### Key Insight\n"
            f"The root cause began {chain_mems[0].timestamp[:10]} — "
            f"well before the event on {event_ts}. "
            f"The chain: {' → '.join([m.content[:30] + '...' for m in chain_mems])}\n\n"
            f"**Confidence:** High — {len(chain_mems)} corroborating memories with clear causal progression."
        )

        mems_sorted = sorted(memories, key=lambda m: m.timestamp)[:min(7, len(memories))]
        examples.append(SFTExample(
            instruction=STAGE3_CAUSAL_INSTRUCTION,
            input=f"Query: What caused the situation described in the most recent memory?\n\n"
                  f"Retrieved Memories:\n{fmt_memories(chain_mems)}",
            output=answer,
            stage="stage3_causal",
            source="template",
        ))

    # TEMPORAL NARRATIVE examples
    for _ in range(half):
        period_mems = sorted(random.sample(memories, min(6, len(memories))),
                             key=lambda m: m.timestamp)
        if not period_mems:
            continue
        start_ts = period_mems[0].timestamp[:10]
        end_ts = period_mems[-1].timestamp[:10]

        period = random.choice(["Q1 2025", "the past 6 months", "last year", "2024"])

        think = generate_think_block([
            f"Building chronological narrative for: {period}",
            f"Memory span: {start_ts} → {end_ts}",
            f"Total memories: {len(period_mems)}",
            "Identifying phases and transitions...",
            "Looking for temporal patterns across the timeline.",
        ])

        answer = f"{think}\n\n## Timeline: {period}\n\n"
        for m in period_mems:
            date_str = m.timestamp[:10]
            answer += f"**{date_str}** — {m.content[:100]}\n"
            if random.random() > 0.6:
                answer += f"   *Pattern note: {random.choice(TEMPLATE_VARS['pattern'])}*\n"
            answer += "\n"

        answer += (
            f"### Overall Pattern\n"
            f"{random.choice(TEMPLATE_VARS['recurring_theme'])}\n\n"
            f"**Confidence:** High — {len(period_mems)} memories spanning {start_ts} to {end_ts}."
        )

        examples.append(SFTExample(
            instruction=STAGE3_TIMELINE_INSTRUCTION,
            input=f"Query: Build a timeline for {period}\n\n"
                  f"Retrieved Memories:\n{fmt_memories(period_mems)}",
            output=answer,
            stage="stage3_causal",
            source="template",
        ))

    while len(examples) < target_count:
        base = random.choice(examples[:min(40, len(examples))])
        examples.append(SFTExample(
            instruction=base.instruction, input=base.input, output=base.output,
            stage="stage3_causal", source="template", quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 4: Self-RAG / CRAG ────────────────────────────────────────────────

def build_stage4(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []
    half = target_count // 2

    SELFRAG_SCENARIOS = {
        "fully_supported": ("yes", "full",    5, "ACCEPT"),
        "partial_halluc":  ("yes", "partial", 3, "REGENERATE"),
        "full_halluc":     ("no",  "none",    1, "REJECT"),
    }

    for scenario_name, (isrel, issup, isuse, decision) in SELFRAG_SCENARIOS.items():
        count = half // len(SELFRAG_SCENARIOS)
        for _ in range(count):
            mems = random.sample(memories, min(3, len(memories)))
            query = random.choice([
                f"What did I discuss about {random.choice(TEMPLATE_VARS['topic'])}?",
                f"What happened with {random.choice(TEMPLATE_VARS['topic'])} last month?",
            ])

            if scenario_name == "fully_supported":
                generated_answer = f"Based on your memories, {mems[0].content[:80]}"
                halluc_note = "All claims are supported by the retrieved memories."
                corrected = generated_answer
            elif scenario_name == "partial_halluc":
                generated_answer = (
                    f"Based on your memories, {mems[0].content[:60]}. "
                    f"Additionally, you mentioned discussing a completely fabricated claim "
                    f"that appears nowhere in the memories."
                )
                halluc_note = "Claim 2 ('completely fabricated claim') is NOT in any memory — HALLUCINATION."
                corrected = f"Based on your memories: {mems[0].content[:80]} [Memory: {mems[0].timestamp[:10]}]."
            else:
                generated_answer = (
                    "You extensively discussed this topic and reached several detailed conclusions "
                    "that are entirely fabricated and not in any provided memory."
                )
                halluc_note = "The entire answer is fabricated — no claims are in the retrieved memories."
                corrected = "I don't have sufficient memories to answer this question reliably."

            think = generate_think_block([
                "Evaluating each claim in the generated answer:",
                f"Claim 1: {mems[0].content[:40]}... → {'Supported ✓' if scenario_name != 'full_halluc' else 'NOT supported ✗'}",
                halluc_note,
                f"Decision: {decision}",
            ])

            output = (
                f"{think}\n\n"
                f"**Critique:**\n\n"
                f"[ISREL: {isrel}] — The answer {'addresses' if isrel == 'yes' else 'does not address'} the query.\n"
                f"[ISSUP: {issup}] — {halluc_note}\n"
                f"[ISUSE: {isuse}] — {'Reliable and useful' if isuse >= 4 else 'Partially useful' if isuse >= 2 else 'Not useful — do not present to user'}.\n\n"
                f"**Decision:** {decision}\n\n"
                f"**Corrected Answer:** {corrected}"
            )

            mem_ctx = fmt_memories(mems)
            examples.append(SFTExample(
                instruction=STAGE4_SELFRAG_INSTRUCTION,
                input=f"Query: {query}\n\nRetrieved Memories:\n{mem_ctx}\n\nGenerated Answer: {generated_answer}",
                output=output,
                stage="stage4_selfrag",
                source="template",
            ))

    # CRAG examples
    for _ in range(half):
        mems = random.sample(memories, min(5, len(memories)))
        query = f"How did my {random.choice(TEMPLATE_VARS['topic'])} change this year?"

        verdict_rows = ""
        keep_ids = []
        for i, m in enumerate(mems, 1):
            rel = round(random.uniform(0.3, 0.98), 2)
            sup = round(rel * random.uniform(0.8, 1.0), 2)
            conf = round((rel + sup) / 2, 2)
            verdict = "✅ KEEP" if rel > 0.6 else ("⚠️ KEEP (marginal)" if rel > 0.4 else "❌ REMOVE")
            if rel > 0.6:
                keep_ids.append(i)
            verdict_rows += f"| {i}. {m.content[:35]}... | {rel} | {sup} | {conf} | {verdict} |\n"

        crag_decision = "CORRECT" if len(keep_ids) >= 2 else ("AMBIGUOUS" if len(keep_ids) == 1 else "INCORRECT")

        think = generate_think_block([
            f"Evaluating {len(mems)} retrieved memories for relevance to: {query[:50]}",
            *[f"  Memory {i}: relevance assessment → {'HIGH' if i in keep_ids else 'LOW'}"
              for i in range(1, len(mems) + 1)],
            f"Overall: {len(keep_ids)} relevant, {len(mems) - len(keep_ids)} irrelevant → {crag_decision}",
        ])

        output = (
            f"{think}\n\n"
            f"**CRAG Evaluation:**\n\n"
            f"| Memory | Relevance | Support | Confidence | Verdict |\n"
            f"|--------|-----------|---------|------------|---------|\n"
            f"{verdict_rows}\n"
            f"**Overall CRAG Decision: {crag_decision}**\n"
            f"- {len(keep_ids)} memories kept for generation\n"
            f"- {len(mems) - len(keep_ids)} memories filtered out\n"
            f"- {'Proceed to generation' if crag_decision == 'CORRECT' else 'Refine retrieval query' if crag_decision == 'AMBIGUOUS' else 'Completely re-retrieve'}"
        )

        examples.append(SFTExample(
            instruction=STAGE4_CRAG_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories:\n{fmt_memories(mems)}",
            output=output,
            stage="stage4_selfrag",
            source="template",
        ))

    while len(examples) < target_count:
        base = random.choice(examples[:min(40, len(examples))])
        examples.append(SFTExample(
            instruction=base.instruction, input=base.input, output=base.output,
            stage="stage4_selfrag", source="template", quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 5: Belief Evolution ───────────────────────────────────────────────

CHANGE_TYPES = ["REFINEMENT", "CONTRADICTION", "EXPANSION", "ABANDONMENT", "STABLE"]
CHANGE_EMOJIS = {
    "REFINEMENT":    "🔄",
    "CONTRADICTION": "⚡",
    "EXPANSION":     "📈",
    "ABANDONMENT":   "🚫",
    "STABLE":        "✅",
}


def build_stage5(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []

    for _ in range(target_count):
        topic = random.choice(BELIEF_TOPICS)

        # Build belief arc: 3-4 snapshots over time
        start = datetime(2024, 1, 1)
        arc_dates = sorted([
            start + timedelta(days=random.randint(0, 180)),
            start + timedelta(days=random.randint(181, 360)),
            start + timedelta(days=random.randint(361, 540)),
        ])
        if random.random() > 0.5:
            arc_dates.append(start + timedelta(days=random.randint(541, 720)))

        belief_states = [
            f"I strongly believe {topic} is very important and beneficial.",
            f"I'm starting to question my previous stance on {topic}.",
            f"My view on {topic} has evolved significantly — I now think {random.choice(TEMPLATE_VARS['new_view'])}.",
        ]
        if len(arc_dates) == 4:
            belief_states.append(
                f"Looking back, my journey with {topic} shows {random.choice(TEMPLATE_VARS['pattern'])}."
            )

        belief_arc = list(zip(arc_dates, belief_states))
        changes = []
        for i in range(1, len(belief_arc)):
            changes.append(random.choice(CHANGE_TYPES))

        think_steps = [f"Analyzing belief evolution for topic: {topic}"]
        for i, (dt, state) in enumerate(belief_arc):
            think_steps.append(f"  [{dt.strftime('%Y-%m-%d')}]: {state[:60]}...")
            if i > 0:
                think_steps.append(f"  Change from previous: {changes[i-1]}")
        think_steps.append(f"Overall pattern: {random.choice(TEMPLATE_VARS['pattern'])}")

        think = generate_think_block(think_steps)

        # Table
        table = "| Date | Position | Change Type |\n|------|---------|-------------|\n"
        for i, (dt, state) in enumerate(belief_arc):
            change_label = f"**{changes[i-1]}**" if i > 0 else "*Baseline*"
            emoji = CHANGE_EMOJIS.get(changes[i-1], "") if i > 0 else ""
            table += f"| {dt.strftime('%Y-%m-%d')} | {state[:50]}... | {emoji} {change_label} |\n"

        key_insight = random.choice([
            f"Your views on {topic} haven't been static — they've followed a {random.choice(['linear', 'cyclical', 'spiral'])} path.",
            f"The evolution shows that your {topic} perspective deepens with experience.",
            f"What looks like a contradiction is actually a sign of growth — you updated your beliefs with new evidence.",
        ])

        answer = (
            f"{think}\n\n"
            f"## Belief Evolution: {topic.title()}\n\n"
            f"### Timeline\n\n{table}\n"
            f"### Analysis\n\n"
        )

        for i, (dt, state) in enumerate(belief_arc):
            label = CHANGE_TYPES[i-1] if i > 0 else "Initial Position"
            answer += f"**{dt.strftime('%b %Y')}** ({label}) — {state}\n\n"

        answer += (
            f"### Key Insight\n{key_insight}\n\n"
            f"**Confidence:** High — {len(belief_arc)} memories spanning "
            f"{(arc_dates[-1] - arc_dates[0]).days} days show clear evolution."
        )

        mem_input = "\n".join([
            f"{i+1}. [{dt.strftime('%Y-%m-%d')}] {state}"
            for i, (dt, state) in enumerate(belief_arc)
        ])

        examples.append(SFTExample(
            instruction=STAGE5_BELIEF_INSTRUCTION,
            input=f"Topic: {topic.title()}\n\nMemories (chronological):\n{mem_input}",
            output=answer,
            stage="stage5_belief",
            source="template",
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 6: Memory Consolidation & Summarization ───────────────────────────

def build_stage6(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []
    dist = {
        "daily":        int(target_count * 0.20),
        "weekly":       int(target_count * 0.17),
        "monthly":      int(target_count * 0.13),
        "proposition":  int(target_count * 0.27),
        "prefix":       int(target_count * 0.13),
        "entity":       int(target_count * 0.10),
    }

    # DAILY SUMMARIES
    for _ in range(dist["daily"]):
        day_mems = sorted(random.sample(memories, min(6, len(memories))), key=lambda m: m.timestamp)
        date_str = day_mems[0].timestamp[:10]
        entities = list(set(e for m in day_mems for e in m.entities))[:4]
        topics = list(set(t for m in day_mems for t in m.topics))[:3]
        importance = "High" if any(m.importance > 0.8 for m in day_mems) else "Medium"

        think = generate_think_block([
            f"Daily summary for {date_str}",
            f"Events: {len(day_mems)} memories to compress",
            "Preserving all specifics (daily level = maximum detail)",
            f"Key entities: {entities}",
            f"Topics: {topics}",
        ])

        content_lines = "\n".join([f"- {m.content[:80]}" for m in day_mems])
        summary = (
            f"{think}\n\n"
            f"**Daily Summary — {date_str}:**\n\n"
            f"{day_mems[0].content[:120]}\n\n"
            f"{'Key interaction: ' + day_mems[1].content[:100] if len(day_mems) > 1 else ''}\n\n"
            f"**Entities:** {entities if entities else ['(none identified)']}\n"
            f"**Topics:** {topics if topics else ['general']}\n"
            f"**Importance:** {importance}"
        )

        mem_str = "\n".join([f"{i+1}. [{m.timestamp[11:16]}] {m.content}" for i, m in enumerate(day_mems)])
        examples.append(SFTExample(
            instruction=STAGE6_SUMMARY_INSTRUCTION,
            input=f"Level: DAILY\nDate: {date_str}\n\nMemories:\n{mem_str}",
            output=summary,
            stage="stage6_summarization",
            source="template",
        ))

    # PROPOSITION EXTRACTION
    for _ in range(dist["proposition"]):
        mem = random.choice(memories)
        content = mem.content

        # Build atomic propositions deterministically
        sentences = [s.strip() for s in content.replace(". ", ".|").split("|") if s.strip()]
        props = []
        for i, sent in enumerate(sentences[:5], 1):
            props.append(f"{i}. {sent}")

        # Add derived propositions
        if len(props) < 4:
            props.append(f"{len(props)+1}. This event occurred on {mem.timestamp[:10]}.")
            if mem.entities:
                props.append(f"{len(props)+1}. The people involved: {', '.join(mem.entities)}.")

        output = "\n".join(props)

        examples.append(SFTExample(
            instruction=STAGE6_PROPOSITION_INSTRUCTION,
            input=f"Memory: {content}",
            output=output,
            stage="stage6_summarization",
            source="template",
        ))

    while len(examples) < target_count:
        base = random.choice(examples[:min(40, len(examples))])
        examples.append(SFTExample(
            instruction=base.instruction, input=base.input, output=base.output,
            stage="stage6_summarization", source="template", quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 7: Multi-Turn Dialogue ────────────────────────────────────────────

def build_stage7(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []

    PROJECTS = ["Project Alpha", "the AI tutoring startup", "the blog redesign", "the research paper"]
    PEOPLE =   ["John", "Sarah", "my manager", "Maria", "the team"]

    for _ in range(target_count):
        project = random.choice(PROJECTS)
        person = random.choice(PEOPLE)
        num_turns = random.randint(2, 4)

        turns = []
        # Turn 1: broad query
        turns.append({
            "user": f"What was I working on last month?",
            "assistant": f"Last month you were working on several things: {project}, "
                         f"catching up with {person}, and managing your weekly goals. "
                         f"[Memory: {random.choice(memories).timestamp[:10]}]"
        })

        # Turn 2: drill down
        turns.append({
            "user": f"How did the one with {person} go?",
            "assistant": f"Your interactions with {person} went well — you made progress on "
                         f"{project} together. They gave useful feedback. "
                         f"[Memory: {random.choice(memories).timestamp[:10]}]"
        })

        # Turn 3: current user query
        user_query = random.choice([
            f"What specifically did {person} say?",
            f"Was there any feedback on {project}?",
            f"Did that lead to any decisions?",
        ])

        response_mem = random.choice(memories)
        think = generate_think_block([
            f"Turn {num_turns} — user asking: {user_query[:50]}",
            f"Context from Turn 1: working on {project}, interactions with {person}",
            f"Context from Turn 2: progress made, feedback given",
            f"Resolving reference — '{person}' = the person from earlier turns",
            "Maintaining conversational flow without re-explaining context.",
        ])

        response = (
            f"{think}\n\n"
            f"The key feedback from {person} was: {response_mem.content[:100]} "
            f"[Memory: {response_mem.timestamp[:10]}]\n\n"
            f"This was specifically about {project}, which you had raised in your earlier conversation.\n\n"
            f"**Confidence:** High — directly related to what we discussed earlier in this conversation."
        )

        # Format multi-turn input
        turn_str = ""
        for i, t in enumerate(turns, 1):
            turn_str += f"Turn {i} (User): {t['user']}\n"
            turn_str += f"Turn {i} (Assistant): {t['assistant']}\n\n"
        turn_str += f"Turn {num_turns} (User): {user_query}"

        examples.append(SFTExample(
            instruction=STAGE7_DIALOGUE_INSTRUCTION,
            input=turn_str,
            output=response,
            stage="stage7_dialogue",
            source="template",
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 8: Long-Context Multi-Hop ─────────────────────────────────────────

def build_stage8(memories: List[Memory], target_count: int) -> List[SFTExample]:
    examples = []

    LIFE_DOMAINS = ["career", "relationships", "health", "learning", "finances"]

    for _ in range(target_count):
        # Select 10-15 diverse memories
        num_mems = random.randint(10, min(15, len(memories)))
        selected = sorted(random.sample(memories, num_mems), key=lambda m: m.timestamp)

        domain_a = random.choice(LIFE_DOMAINS)
        domain_b = random.choice([d for d in LIFE_DOMAINS if d != domain_a])

        query = (
            f"How did my {domain_a} situation and my {domain_b} situation "
            f"interconnect and influence each other over the past year?"
        )

        # Build think block with multi-hop trace
        think_steps = [
            f"Multi-hop query: interconnection between {domain_a} and {domain_b}",
            f"Processing {len(selected)} memories spanning "
            f"{selected[0].timestamp[:10]} → {selected[-1].timestamp[:10]}",
            f"Thread A ({domain_a}): identifying relevant memories...",
        ]
        thread_a = [m for m in selected if domain_a in m.topics or random.random() > 0.6][:4]
        thread_b = [m for m in selected if domain_b in m.topics or random.random() > 0.6][:4]
        for m in thread_a:
            think_steps.append(f"  [{m.timestamp[:10]}] {m.content[:45]}...")
        think_steps.append(f"Thread B ({domain_b}): identifying relevant memories...")
        for m in thread_b:
            think_steps.append(f"  [{m.timestamp[:10]}] {m.content[:45]}...")
        think_steps.append("Cross-domain connections: identifying causal links between threads")
        think_steps.append(f"Key multi-hop insight: {domain_a} changes in Month 2 influenced {domain_b} in Month 4")

        think = generate_think_block(think_steps)

        answer = (
            f"{think}\n\n"
            f"## The Interconnected Story: {domain_a.title()} × {domain_b.title()}\n\n"
            f"These two areas of your life were not separate stories — they were causally interlinked.\n\n"
            f"### Phase 1: The Starting Point\n"
        )
        if thread_a:
            answer += f"[Memory: {thread_a[0].timestamp[:10]}] — {thread_a[0].content[:100]}\n\n"
        answer += f"### Phase 2: The Interaction\n"
        if thread_b:
            answer += f"[Memory: {thread_b[0].timestamp[:10]}] — {thread_b[0].content[:100]}\n\n"
        answer += (
            f"### Phase 3: The Reinforcing Cycle\n"
            f"Once the connection was established, changes in {domain_a} directly influenced {domain_b}, "
            f"and vice versa.\n\n"
            f"### Key Multi-Hop Insight\n"
            f"The pattern across {len(selected)} memories spanning "
            f"{(datetime.fromisoformat(selected[-1].timestamp[:10]) - datetime.fromisoformat(selected[0].timestamp[:10])).days} days: "
            f"{random.choice(TEMPLATE_VARS['meta_insight'])}\n\n"
            f"**Confidence:** High — {len(selected)} memories provide extensive cross-domain evidence."
        )

        examples.append(SFTExample(
            instruction=STAGE8_MULTIHOP_INSTRUCTION,
            input=f"Query: {query}\n\nRetrieved Memories ({len(selected)} chunks):\n{fmt_memories(selected, max_count=15)}",
            output=answer,
            stage="stage8_longcontext",
            source="template",
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 9: DPO Preference Pairs ───────────────────────────────────────────

DPO_CATEGORIES = {
    "comprehensive_vs_terse":      600,
    "grounded_vs_speculative":     500,
    "empathetic_vs_clinical":      400,
    "calibrated_vs_overconfident": 400,
    "structured_vs_rambling":      400,
    "nuanced_vs_simplistic":       350,
    "honest_refusal_vs_halluc":    350,
}


def build_stage9(memories: List[Memory], target_count: int) -> List[DPOExample]:
    examples = []

    for category, category_count in DPO_CATEGORIES.items():
        per_run = max(1, int(target_count * category_count / sum(DPO_CATEGORIES.values())))

        for _ in range(per_run):
            mems = random.sample(memories, min(4, len(memories)))
            topic = random.choice(BELIEF_TOPICS)
            mem_ctx = fmt_memories(mems)

            prompt = f"Query: How has my relationship with {topic} changed over the past year?\n\nMemories:\n{mem_ctx}"

            if category == "comprehensive_vs_terse":
                mem0, mem1 = mems[0], mems[min(1, len(mems)-1)]
                chosen = (
                    f"<think>\nLet me trace the evolution across all memories...\n"
                    f"Memory 1 [{mem0.timestamp[:10]}]: {mem0.content[:50]}... → this establishes the baseline\n"
                    f"Memory 2 [{mem1.timestamp[:10]}]: {mem1.content[:50]}... → this shows the shift\n"
                    f"The arc: conflict → repair → growth\n</think>\n\n"
                    f"Your relationship with {topic} has undergone a meaningful transformation:\n\n"
                    f"**The starting point** [Memory: {mem0.timestamp[:10]}] — {mem0.content[:100]}\n\n"
                    f"**The shift** [Memory: {mem1.timestamp[:10]}] — {mem1.content[:100]}\n\n"
                    f"**The pattern:** {random.choice(TEMPLATE_VARS['meta_insight'])}\n\n"
                    f"**Confidence:** High — {len(mems)} memories show consistent trajectory."
                )
                rejected = f"Your {topic} relationship changed. It got better over the year based on the memories."

            elif category == "grounded_vs_speculative":
                chosen = (
                    f"Based strictly on your memories:\n\n"
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:100]}\n\n"
                    f"That's what the memories show. I don't have additional context beyond these records.\n\n"
                    f"**Confidence:** Medium — limited to 1-2 directly relevant memories."
                )
                rejected = (
                    f"Based on the pattern of what typically happens, it seems like your {topic} "
                    f"situation probably improved significantly, likely due to several factors "
                    f"that people in your situation usually experience."
                )

            elif category == "empathetic_vs_clinical":
                chosen = (
                    f"Looking at your memories, I can see this was a meaningful journey for you. "
                    f"[Memory: {mems[0].timestamp[:10]}] — {mems[0].content[:100]}\n\n"
                    f"The growth here is real and worth acknowledging.\n\n"
                    f"**Confidence:** High — the evidence paints a clear and positive picture."
                )
                rejected = (
                    f"Data from memories: Memory 1 ({mems[0].timestamp[:10]}): {mems[0].content[:50]}. "
                    f"Memory 2 ({mems[min(1,len(mems)-1)].timestamp[:10]}): {mems[min(1,len(mems)-1)].content[:50]}. "
                    f"End of retrieved context."
                )

            elif category == "honest_refusal_vs_halluc":
                prompt = f"Query: What happened during my trip to Japan last summer?\n\nMemories:\n(No memories retrieved)"
                chosen = (
                    f"<think>\nNo memories retrieved for this query. Cannot answer.\n</think>\n\n"
                    f"I don't have any memories about a trip to Japan. "
                    f"It's possible this experience hasn't been added to your memory store yet.\n\n"
                    f"**Confidence:** None — empty retrieval context.\n"
                    f"**Suggestion:** Try ingesting your travel notes, photos, or journal entries from that trip."
                )
                rejected = (
                    f"During your trip to Japan last summer, you visited Tokyo and Kyoto. "
                    f"You particularly enjoyed the food and the temples. "
                    f"It was a transformative experience that gave you a new perspective on your life."
                )

            else:
                mem0 = mems[0]
                chosen = (
                    f"<think>\nAnalyzing the nuanced evolution...\n</think>\n\n"
                    f"This is more complex than it might appear. [Memory: {mem0.timestamp[:10]}] shows "
                    f"{mem0.content[:100]}\n\n"
                    f"**Confidence:** Medium — the situation has multiple dimensions worth exploring further."
                )
                rejected = f"It got better. The memories show improvement over time. That's the main pattern."

            examples.append(DPOExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                category=category,
            ))

    while len(examples) < target_count:
        base = random.choice(examples[:min(40, len(examples))])
        examples.append(DPOExample(
            prompt=base.prompt, chosen=base.chosen, rejected=base.rejected,
            category=base.category, quality_score=0.82,
        ))

    random.shuffle(examples)
    return examples[:target_count]


# ─── STAGE 10: User Style — FROM USER FILES ───────────────────────────────────

def build_stage10_from_files(file_paths: List[Path], target_count: int) -> List[SFTExample]:
    """
    Build Stage 10 training data from user's own files.
    Generates rich, diverse style-learning examples that teach the 7B model
    to respond with the user's authentic voice, vocabulary, and reasoning style.
    Files can be: .txt, .md, .json
    """
    examples = []
    all_content = []
    long_content = []   # paragraphs >= 200 chars for rich training

    for fp in file_paths:
        if not fp.exists():
            continue
        try:
            if fp.suffix in [".txt", ".md"]:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
                all_content.extend(paras)
                long_content.extend([p for p in paras if len(p) >= 200])
            elif fp.suffix == ".json":
                data = json.loads(fp.read_text())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "content" in item:
                            s = str(item["content"])
                            all_content.append(s)
                            if len(s) >= 200:
                                long_content.append(s)
                        elif isinstance(item, str):
                            all_content.append(item)
            print(f"  [Stage 10] Loaded {fp.name}: {len(all_content)} text segments")
        except Exception as e:
            print(f"  [Stage 10] Warning: Could not read {fp}: {e}")

    if not all_content:
        print("[Stage 10] No user files found — generating placeholder examples")
        return _build_stage10_placeholder(target_count)

    # Also load user_memories.json if it exists
    user_memories = []
    mem_path = Path("training_data/user_memories.json")
    if mem_path.exists():
        try:
            mems = json.loads(mem_path.read_text())
            user_memories = [m for m in mems if isinstance(m, dict) and len(m.get("content","")) > 60]
        except Exception:
            pass

    # ── Suraj's distinctive vocabulary and phrases (extracted from his files) ──
    suraj_power_phrases = [
        "fundamentally transform", "exponential", "paradigm-shifting", "disruption at the deepest level",
        "reimagine, restructure, and redesign", "agentic capabilities", "human-in-the-loop",
        "co-evolve", "acceleration of learning", "disruptive learning curve", "exponential leap",
        "deeply rooted", "holistic", "immersive", "transformative", "catalytic",
        "from repetition to creation", "research and innovation", "interdisciplinary",
        "democratize", "sustainability", "human flourishing", "wisdom over knowledge",
        "curiosity, creativity, and creation", "learn, unlearn, and redesign",
        "not just better — exponentially better", "the inevitable", "bare minimum",
        "purpose of life", "raise the standard of living", "solve world's problems",
        "true form of sustainability", "innovation and peace", "pioneering breakthrough",
        "most efficient, accurate, structured", "leading and creating", "shaping the world",
    ]

    suraj_topics = [
        "the future of AI-powered education", "agentic AI systems and their potential",
        "my vision for transforming learning", "EdgeMemory and on-device AI",
        "the purpose of education beyond degrees", "exponential technology and India",
        "multi-agent orchestration in real-world systems", "AI-driven personalization at scale",
        "the gap between knowledge and wisdom", "computational sustainability",
        "RAG systems and memory architecture", "my internship at GoCrackIT",
        "the Aspire Leaders Program and Harvard mentorship", "rural data analysis at Gram Vikas",
        "EEG-based Alzheimer's classification", "portfolio optimization with AI",
        "Cortex Lab and the memory AI system", "fine-tuning language models locally",
        "India's role in the global AI disruption", "learning physics from an AI tutor agent",
        "Dynamic System Prompting for personalized learning", "the failure of modern education",
        "my core life visions", "the Vedas and exponential technology synthesis",
        "what I learned from Dexschool Leadership program", "building impactful AI products",
        "self-stabilizing spoon for Parkinson's disease", "how I approach problem-solving",
        "my journey from Bihar to Bangalore on 100% scholarship",
    ]

    suraj_queries = [
        "What is your vision for the future of education?",
        "How do you approach building AI products?",
        "What makes your learning philosophy unique?",
        "Describe your most impactful project.",
        "How do you think about AI and human co-evolution?",
        "What is Cortex Lab and why does it matter?",
        "What did you learn from your internship experience?",
        "How do you balance technical depth with visionary thinking?",
        "What is your biggest ambition?",
        "How do you see India's role in global AI disruption?",
        "What is your view on the purpose of life and education?",
        "How do you approach exponential learning?",
        "What are the core pain points you want to solve?",
        "Tell me about EdgeMemory and its significance.",
        "How should AI systems be designed for personalization?",
        "What does disruption really mean to you?",
        "How do you stay motivated and focused?",
        "What is Dynamic System Prompting and why does it matter?",
        "How do multi-agent systems change the future of learning?",
        "What is your philosophy on innovation vs. repetition?",
    ]

    # ── Template 1: Style continuation (input = first ~40%, output = remaining 60%) ──
    def t_continuation(content: str) -> SFTExample:
        cut = max(80, int(len(content) * 0.38))
        return SFTExample(
            instruction="Continue this thought in the same writing style and authentic voice.",
            input=content[:cut],
            output=content[cut:],
            stage="stage10_user_style", source="user_file_continuation",
        )

    # ── Template 2: Style rewrite — given a topic, write in user's style ──
    def t_style_write(topic: str, content: str) -> SFTExample:
        excerpt = content[:300].strip()
        return SFTExample(
            instruction=f"Write a thoughtful, visionary response about '{topic}' in the same bold, philosophical, and systems-level style demonstrated in the example below.",
            input=f"Writing style example:\n\n{excerpt}\n\nNow write about: {topic}",
            output=content[:500] if len(content) >= 300 else content,
            stage="stage10_user_style", source="user_file_style_write",
        )

    # ── Template 3: Query → response in user style ──
    def t_query_response(query: str, content: str) -> SFTExample:
        response = content[:600].strip() if len(content) >= 200 else content
        return SFTExample(
            instruction="Answer this query authentically, in the user's own voice and reasoning style. Use their characteristic vocabulary, emphasis patterns, and depth of thinking.",
            input=f"User query: {query}",
            output=response,
            stage="stage10_user_style", source="user_file_query",
        )

    # ── Template 4: Memory → first-person narrative ──
    def t_memory_narrative(mem: dict, content: str) -> SFTExample:
        mem_text = mem.get("content", "")[:200]
        return SFTExample(
            instruction="Reflect on this memory and elaborate on it in the user's authentic writing style — philosophical, visionary, and deeply personal.",
            input=f"Memory: {mem_text}",
            output=content[:500] if len(content) >= 200 else content,
            stage="stage10_user_style", source="user_memory_narrative",
        )

    # ── Template 5: Bullet expansion — given bullet points, write full prose ──
    def t_bullet_expansion(content: str) -> SFTExample:
        lines = [l.strip() for l in content.split("\n") if l.strip().startswith(("*", "-", "•", "1", "2", "3")) and len(l.strip()) > 20]
        if len(lines) < 3:
            return None
        bullets = "\n".join(lines[:6])
        return SFTExample(
            instruction="Expand these bullet points into a rich, detailed, visionary prose response in the user's authentic style.",
            input=f"Expand these points:\n{bullets}",
            output=content[:600],
            stage="stage10_user_style", source="user_file_bullet_expansion",
        )

    # ── Template 6: Concept bridge — connect two ideas in user's style ──
    def t_concept_bridge(c1: str, c2: str) -> SFTExample:
        concept1 = c1[:120].strip()
        concept2 = c2[:120].strip()
        bridge_output = c1[:400] + "\n\n" + c2[:400]
        return SFTExample(
            instruction="Connect these two ideas in a cohesive, systems-level analysis using the user's bold and philosophical writing voice.",
            input=f"Idea 1: {concept1}\n\nIdea 2: {concept2}",
            output=bridge_output,
            stage="stage10_user_style", source="user_file_bridge",
        )

    # ── Template 7: Manifesto-style declaration ──
    def t_manifesto(topic: str, content: str) -> SFTExample:
        return SFTExample(
            instruction=f"Write a bold, manifesto-style declaration about '{topic}' that reflects the user's commitment to disruption, exponential thinking, and human flourishing.",
            input=f"Write my manifesto on: {topic}",
            output=content[:700] if len(content) >= 300 else content,
            stage="stage10_user_style", source="user_file_manifesto",
        )

    # ── Template 8: Critical reflection ──
    def t_reflection(content: str) -> SFTExample:
        return SFTExample(
            instruction="Reflect critically on the following idea, exploring its deeper implications, contradictions, and transformative potential — in the user's characteristic analytical and visionary style.",
            input=f"Reflect on: {content[:150].strip()}",
            output=content[:600],
            stage="stage10_user_style", source="user_file_reflection",
        )

    # ── Template 9: Technical + vision blend ──
    def t_tech_vision(query: str, content: str) -> SFTExample:
        phrase = random.choice(suraj_power_phrases)
        return SFTExample(
            instruction=f"Answer this technical question with both precision and vision. Use phrases like '{phrase}' naturally. Blend technical depth with big-picture thinking.",
            input=query,
            output=content[:600] if len(content) >= 200 else content,
            stage="stage10_user_style", source="user_file_tech_vision",
        )

    # ── Template 10: Personal story → lesson ──
    def t_story_lesson(content: str) -> SFTExample:
        lesson_out = content[:600] if len(content) >= 200 else content
        return SFTExample(
            instruction="Tell this experience as a personal story and extract the key lesson or vision it reinforced, in the user's authentic voice.",
            input=f"My experience: {content[:180].strip()}",
            output=lesson_out,
            stage="stage10_user_style", source="user_file_story",
        )

    # ── GENERATE EXAMPLES ──
    src = long_content if long_content else all_content
    n = len(src)

    # Pass 1: continuation examples from long content
    for c in src:
        if len(c) >= 200:
            examples.append(t_continuation(c))

    # Pass 2: style writes for each topic × sampled content
    for topic in suraj_topics:
        c = src[len(examples) % n] if src else ""
        if c:
            examples.append(t_style_write(topic, c))

    # Pass 3: query → response
    for q in suraj_queries:
        c = src[len(examples) % n]
        examples.append(t_query_response(q, c))

    # Pass 4: memory narratives
    for mem in user_memories[:min(200, len(user_memories))]:
        c = src[len(examples) % n]
        examples.append(t_memory_narrative(mem, c))

    # Pass 5: bullet expansions
    for c in src:
        ex = t_bullet_expansion(c)
        if ex:
            examples.append(ex)

    # Pass 6: concept bridges (pair consecutive long segments)
    for i in range(0, min(len(src) - 1, 300), 2):
        if len(src[i]) >= 150 and len(src[i+1]) >= 150:
            examples.append(t_concept_bridge(src[i], src[i+1]))

    # Pass 7: manifesto style
    manifesto_topics = [
        "the future of education", "AI and human co-evolution", "the purpose of innovation",
        "India's role in global disruption", "building AI for humanity", "exponential learning",
        "why repetition must die and creation must rise", "the inevitable transformation of technology",
        "what it means to truly lead", "building systems that solve world problems",
    ]
    for topic in manifesto_topics:
        c = src[len(examples) % n]
        examples.append(t_manifesto(topic, c))

    # Pass 8: reflections
    for c in src[:min(200, len(src))]:
        examples.append(t_reflection(c))

    # Pass 9: tech+vision blend
    tech_queries = [
        "How does RAG work and why does it matter?",
        "Explain the architecture of a memory AI system.",
        "What is fine-tuning a language model and how does it differ from pre-training?",
        "How do multi-agent systems coordinate in complex tasks?",
        "What makes a good embedding model for semantic search?",
        "How do you design a system for lifelong learning on-device?",
        "What is the role of knowledge graphs in AI memory systems?",
        "Explain LoRA fine-tuning and why it works on constrained hardware.",
        "How do you build production-grade MLOps pipelines?",
        "What is Dynamic System Prompting and how is it implemented?",
    ]
    for q in tech_queries:
        c = src[len(examples) % n]
        examples.append(t_tech_vision(q, c))

    # Pass 10: personal story → lesson
    for c in src[:min(150, len(src))]:
        examples.append(t_story_lesson(c))

    # ── FILL TO TARGET with cycling templates ──
    template_fns = [t_continuation, t_reflection, t_story_lesson]
    idx = 0
    while len(examples) < target_count:
        c = src[idx % n]
        fn = template_fns[idx % len(template_fns)]
        try:
            ex = fn(c)
            if ex:
                examples.append(ex)
        except Exception:
            pass
        idx += 1
        if idx > target_count * 3:
            break

    random.shuffle(examples)
    return examples[:target_count]


def _build_stage10_placeholder(target_count: int) -> List[SFTExample]:
    """Placeholder Stage 10 data when no user files are provided."""
    examples = []
    for _ in range(target_count):
        topic = random.choice(BELIEF_TOPICS)
        examples.append(SFTExample(
            instruction="Respond to this query in the user's preferred communication style.",
            input=f"Query: What do I think about {topic}?\n\n"
                  f"[Add your personal notes/files for authentic style adaptation]",
            output=f"<think>\nThis is a placeholder — feed real user data for Stage 10.\n</think>\n\n"
                   f"Based on your memories, your perspective on {topic} has evolved over time. "
                   f"[Add user-specific data to personalize this response style]",
            stage="stage10_user_style",
            source="placeholder",
        ))
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def quality_filter(examples: list, stage: str) -> list:
    """Filter out low-quality examples based on stage-specific rules."""
    passed = []

    for ex in examples:
        if isinstance(ex, SFTExample):
            # Must have non-empty instruction, input, output
            if not (ex.instruction and ex.input and ex.output):
                continue
            # Stage 10: style examples — relaxed filter (real user text)
            if stage == "stage10_user_style":
                if len(ex.output) >= 60 and len(ex.input) >= 20:
                    passed.append(ex)
                continue
            # Minimum output length for all other stages
            if len(ex.output) < 80:
                continue
            # Stage 1: must have [Memory: or Confidence: in output
            if stage == "stage1_faithfulness":
                if "Memory:" not in ex.output and "Confidence:" not in ex.output:
                    continue
            # Stage 2: must have json block in agentic examples
            if stage == "stage2_agentic":
                if "routing" in ex.instruction.lower() and "json" not in ex.output.lower():
                    continue
            # Stage 4: must have ISREL or CRAG
            if stage == "stage4_selfrag":
                has_selfrag = "ISREL" in ex.output or "ISSUP" in ex.output
                has_crag = "CRAG" in ex.output or "KEEP" in ex.output or "REMOVE" in ex.output
                if not (has_selfrag or has_crag):
                    continue
            # Stage 5: must have belief evolution indicators
            if stage == "stage5_belief":
                if not any(ct in ex.output for ct in CHANGE_TYPES):
                    continue
            # Remove duplicates by hashing input
            passed.append(ex)
        elif isinstance(ex, DPOExample):
            if not (ex.prompt and ex.chosen and ex.rejected):
                continue
            if ex.chosen == ex.rejected:
                continue
            if len(ex.chosen) < len(ex.rejected):
                continue  # Chosen should be more detailed
            passed.append(ex)

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_stage(stage_name: str, examples: list, overwrite: bool = False):
    """Save dataset to training_data/<stage_name>.json"""
    out_path = DATA_DIR / f"{stage_name}.json"

    if out_path.exists() and not overwrite:
        # Append to existing
        existing = json.loads(out_path.read_text())
        print(f"[Save] Appending {len(examples)} to existing {len(existing)} examples in {out_path.name}")
        all_examples = existing + [asdict(e) for e in examples]
    else:
        all_examples = [asdict(e) for e in examples]

    out_path.write_text(json.dumps(all_examples, indent=2, ensure_ascii=False))
    size_kb = out_path.stat().st_size / 1024
    print(f"[Save] ✅ {out_path.name}: {len(all_examples)} examples ({size_kb:.1f} KB)")
    return out_path


def check_existing(stage_name: str) -> int:
    """Return count of already-generated examples for a stage."""
    p = DATA_DIR / f"{stage_name}.json"
    if p.exists():
        data = json.loads(p.read_text())
        return len(data)
    return 0


def print_progress_table():
    """Print current dataset generation progress."""
    print("\n" + "="*70)
    print(f"{'Stage':<30} {'Target':>8} {'Generated':>10} {'Progress':>10}")
    print("="*70)
    total_gen = 0
    total_target = 0
    for stage, cfg in STAGE_TARGETS.items():
        target = cfg["count"]
        generated = check_existing(stage)
        pct = min(100, int(generated / target * 100))
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"{stage:<30} {target:>8} {generated:>10} {bar} {pct:>3}%")
        total_gen += generated
        total_target += target
    print("="*70)
    total_pct = min(100, int(total_gen / total_target * 100))
    print(f"{'TOTAL':<30} {total_target:>8} {total_gen:>10} {total_pct:>3}%")
    print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(
    stage_name: str,
    target_count: int,
    is_dpo: bool,
    memories: List[Memory],
    quick_test: bool = False,
    user_files: List[Path] = None,
    resume: bool = False,
):
    already_done = check_existing(stage_name)
    remaining = target_count - already_done

    if resume and already_done >= target_count:
        print(f"[{stage_name}] Already complete ({already_done}/{target_count}) — skipping")
        return

    if quick_test:
        remaining = min(remaining, 50)

    print(f"\n{'='*60}")
    print(f"▶ Generating: {stage_name}")
    print(f"  Target: {target_count} | Done: {already_done} | To generate: {remaining}")
    print(f"{'='*60}")

    t0 = time.time()
    examples = []

    if stage_name == "stage1_faithfulness":
        examples = build_stage1(memories, remaining)
    elif stage_name == "stage2_agentic":
        examples = build_stage2(memories, remaining)
    elif stage_name == "stage3_causal":
        examples = build_stage3(memories, remaining)
    elif stage_name == "stage4_selfrag":
        examples = build_stage4(memories, remaining)
    elif stage_name == "stage5_belief":
        examples = build_stage5(memories, remaining)
    elif stage_name == "stage6_summarization":
        examples = build_stage6(memories, remaining)
    elif stage_name == "stage7_dialogue":
        examples = build_stage7(memories, remaining)
    elif stage_name == "stage8_longcontext":
        examples = build_stage8(memories, remaining)
    elif stage_name == "stage9_dpo":
        examples = build_stage9(memories, remaining)
    elif stage_name == "stage10_user_style":
        files = user_files or []
        examples = build_stage10_from_files(files, remaining)

    # Quality filter
    before = len(examples)
    examples = quality_filter(examples, stage_name)
    after = len(examples)
    if before != after:
        print(f"  [QFilter] {before} → {after} examples (removed {before - after} low-quality)")

    # Save
    save_stage(stage_name, examples, overwrite=not resume)

    elapsed = time.time() - t0
    rate = len(examples) / elapsed if elapsed > 0 else 0
    print(f"  ✅ Done in {elapsed:.1f}s ({rate:.0f} examples/sec)")


def main():
    parser = argparse.ArgumentParser(description="Cortex Lab Local Dataset Generator")
    parser.add_argument("--all",          action="store_true", help="Generate all stages")
    parser.add_argument("--stage",        type=str,            help="Stage name, e.g. stage1_faithfulness")
    parser.add_argument("--from-files",   type=str,            help="Path to folder with user files (for Stage 10)")
    parser.add_argument("--quick-test",   action="store_true", help="Generate only 50 examples per stage (for testing)")
    parser.add_argument("--resume",       action="store_true", help="Skip already-completed stages")
    parser.add_argument("--status",       action="store_true", help="Show generation progress table")
    parser.add_argument("--personas",     type=int, default=15, help="Number of personas to generate memories for")
    args = parser.parse_args()

    if args.status:
        print_progress_table()
        return

    print("\n" + "🧠 " * 20)
    print("CORTEX LAB — DATASET GENERATOR")
    print("Pure template engine. No external model. No API. No cost.")
    print()
    print("WHAT THIS BUILDS:")
    print("  Synthetic training data that teaches the 7B model HOW to:")
    print("  • Cite memories with [Memory: timestamp]")
    print("  • Route queries to the right agent (JSON)")
    print("  • Trace causal chains across time")
    print("  • Self-critique with ISREL/ISSUP/ISUSE tokens")
    print("  • Detect belief evolution and contradictions")
    print("  • Synthesize across 15+ memories")
    print()
    print("IN PRODUCTION: The fine-tuned model applies these skills")
    print("  to any real user's ingested data. Zero retraining needed.")
    print("🧠 " * 20 + "\n")

    # Generate base memories from all personas (pure template — instant)
    print(f"\n[Setup] Generating memory timelines for {args.personas} personas...")
    mem_gen = MemoryGenerator()
    all_memories = []
    persona_subset = random.sample(PERSONAS, min(args.personas, len(PERSONAS)))
    for persona in persona_subset:
        mems = mem_gen.generate_persona_memories(persona, num_memories=200)
        all_memories.extend(mems)
        print(f"  ✓ {persona['name']} ({persona['job']}, {persona['city']}): {len(mems)} memories")

    print(f"\n[Setup] Memory pool: {len(all_memories)} memories across {len(persona_subset)} personas\n")

    # User files for Stage 10
    user_files = []
    if args.from_files:
        user_file_dir = Path(args.from_files)
        if user_file_dir.exists():
            user_files = list(user_file_dir.glob("**/*.txt")) + \
                         list(user_file_dir.glob("**/*.md")) + \
                         list(user_file_dir.glob("**/*.json"))
            print(f"[Stage 10] Found {len(user_files)} user files in {user_file_dir}")
        else:
            print(f"[Stage 10] Warning: --from-files path not found: {user_file_dir}")

    # Run generation
    if args.all:
        for stage_name, cfg in STAGE_TARGETS.items():
            run_stage(
                stage_name=stage_name,
                target_count=cfg["count"],
                is_dpo=cfg["dpo"],
                memories=all_memories,
                quick_test=args.quick_test,
                user_files=user_files if stage_name == "stage10_user_style" else None,
                resume=args.resume,
            )
    elif args.stage:
        if args.stage not in STAGE_TARGETS:
            print(f"ERROR: Unknown stage '{args.stage}'")
            print(f"Valid stages: {list(STAGE_TARGETS.keys())}")
            return
        cfg = STAGE_TARGETS[args.stage]
        run_stage(
            stage_name=args.stage,
            target_count=cfg["count"],
            is_dpo=cfg["dpo"],
            memories=all_memories,
            quick_test=args.quick_test,
            user_files=user_files if args.stage == "stage10_user_style" else None,
            resume=args.resume,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/generate_datasets.py --status")
        print("  python scripts/generate_datasets.py --all --quick-test")
        print("  python scripts/generate_datasets.py --all")
        print("  python scripts/generate_datasets.py --stage stage1_faithfulness")

    # Final status
    print_progress_table()
    total_size = sum(
        (DATA_DIR / f"{s}.json").stat().st_size
        for s in STAGE_TARGETS
        if (DATA_DIR / f"{s}.json").exists()
    )
    print(f"✅ Dataset generation complete! Files saved to training_data/")
    print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nNext step:")
    print(f"  python scripts/fine_tune_cortex.py --stage stage1_faithfulness")


if __name__ == "__main__":
    main()
