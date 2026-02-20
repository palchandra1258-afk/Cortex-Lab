"""
Cortex Lab — User File Ingestion Pipeline
==========================================
Drop any personal files into raw_data/ and run this script.
It extracts your content and formats it as memory objects for:
  - Stage 10 (user style training)
  - Enriching the memory pool for all other stages

Supported formats:
    .txt      Plain text, journal entries, notes
    .md       Markdown notes, README files
    .json     Chat exports, structured data
    .pdf      PDFs (requires: pip install pdfplumber)
    .py       Python source files
    .csv      CSV data (summarizes each row)
    Chat      WhatsApp/Telegram exports (.txt format)

Usage:
    # Ingest everything in raw_data/
    python scripts/ingest_user_files.py

    # Ingest a specific folder
    python scripts/ingest_user_files.py --input /path/to/my/notes

    # Show what would be extracted (dry run)
    python scripts/ingest_user_files.py --dry-run

    # Reset all extracted memories and re-ingest
    python scripts/ingest_user_files.py --reset
"""

import os
import sys
import json
import re
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
RAW_DIR   = ROOT / "raw_data"
DATA_DIR  = ROOT / "training_data"
OUT_FILE  = DATA_DIR / "user_memories.json"

RAW_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserMemory:
    event_id: str
    timestamp: str
    content: str
    memory_type: str        # episodic | semantic | reflective | belief | code | conversation
    source_file: str
    char_count: int
    entities: List[str]
    topics: List[str]
    emotion: str            # neutral | positive | negative | mixed | reflective
    importance: float
    raw_segment: str        # original text before any processing

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY & TOPIC EXTRACTION (lightweight, no external NLP needed)
# ─────────────────────────────────────────────────────────────────────────────

PERSON_PATTERNS = [
    r'\bmy (?:manager|boss|mentor|colleague|partner|wife|husband|friend|sister|brother|dad|mom|teacher|professor)\b',
    r'\b(?:Dr\.|Mr\.|Ms\.|Prof\.)\s+[A-Z][a-z]+',
    r'\b[A-Z][a-z]{2,10}(?:\s+[A-Z][a-z]{2,10})?\b(?=\s+(?:said|told|mentioned|asked|helped|gave|showed))',
]

TOPIC_KEYWORDS = {
    "career":        ["job", "work", "promotion", "interview", "manager", "salary", "career",
                      "project", "colleague", "office", "meeting", "deadline", "performance"],
    "learning":      ["read", "book", "course", "learned", "study", "research", "understand",
                      "insight", "concept", "paper", "lecture", "education", "skill"],
    "relationships": ["friend", "partner", "family", "relationship", "conversation",
                      "mentor", "love", "marriage", "breakup", "dating", "social"],
    "health":        ["sleep", "exercise", "gym", "diet", "health", "anxiety", "stress",
                      "meditation", "doctor", "mental", "energy", "tired", "rest"],
    "planning":      ["goal", "plan", "decision", "priority", "strategy", "objective",
                      "schedule", "deadline", "budget", "organize", "review"],
    "finance":       ["money", "salary", "invest", "savings", "expense", "budget",
                      "debt", "loan", "bank", "income", "financial"],
    "creativity":    ["write", "design", "create", "art", "music", "build", "project",
                      "idea", "inspiration", "creative", "startup", "product"],
    "reflection":    ["realized", "noticed", "pattern", "think", "believe", "feel",
                      "wonder", "understand myself", "growth", "changed"],
    "code":          ["python", "javascript", "code", "function", "bug", "api",
                      "database", "server", "deploy", "git", "algorithm"],
}

POSITIVE_WORDS = {"great", "amazing", "excited", "happy", "proud", "love", "fantastic",
                  "wonderful", "excellent", "breakthrough", "success", "win", "joy"}
NEGATIVE_WORDS = {"bad", "sad", "frustrated", "angry", "anxious", "worried", "failed",
                  "disappointed", "stressed", "overwhelmed", "difficult", "struggle", "hard"}
REFLECTIVE_WORDS = {"realized", "noticed", "thinking", "wonder", "reflect", "pattern",
                    "understand", "growth", "changed", "learned", "insight"}


def detect_emotion(text: str) -> str:
    text_lower = text.lower()
    words = set(text_lower.split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    ref = len(words & REFLECTIVE_WORDS)

    if ref >= 2:
        return "reflective"
    elif pos > neg and pos >= 2:
        return "positive"
    elif neg > pos and neg >= 2:
        return "negative"
    elif pos > 0 and neg > 0:
        return "mixed"
    return "neutral"


def detect_topics(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if sum(1 for kw in keywords if kw in text_lower) >= 2:
            found.append(topic)
    return found[:4] or ["general"]


def extract_entities(text: str) -> List[str]:
    entities = []
    for pattern in PERSON_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        entities.extend(matches[:3])
    # Deduplicate and clean
    clean = list(set(e.strip() for e in entities if len(e.strip()) > 3))
    return clean[:5]


def estimate_importance(text: str, topics: List[str]) -> float:
    score = 0.5
    # Longer content = more important
    if len(text) > 300:
        score += 0.1
    if len(text) > 600:
        score += 0.1
    # Reflective content = more important
    if "reflective" in topics or any(w in text.lower() for w in REFLECTIVE_WORDS):
        score += 0.15
    # Decision/planning content
    if "planning" in topics or any(w in text.lower() for w in ["decided", "plan", "goal"]):
        score += 0.1
    # Emotional content
    if any(w in text.lower() for w in list(POSITIVE_WORDS) + list(NEGATIVE_WORDS)):
        score += 0.05
    return round(min(1.0, score), 2)


def content_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:12]


def infer_memory_type(text: str, source_file: str) -> str:
    text_lower = text.lower()
    ext = Path(source_file).suffix.lower()

    if ext == ".py":
        return "code"
    if any(kw in text_lower for kw in ["chat", "message", "sent", "received", "replied"]):
        return "conversation"
    if any(kw in text_lower for kw in ["realized", "pattern", "reflecting", "looking back", "i notice"]):
        return "reflective"
    if any(kw in text_lower for kw in ["i believe", "i think", "my view", "my opinion", "i feel"]):
        return "belief"
    if any(kw in text_lower for kw in ["learned", "understood", "studied", "read", "insight"]):
        return "semantic"
    return "episodic"


# ─────────────────────────────────────────────────────────────────────────────
# FILE PARSERS — one per file type
# ─────────────────────────────────────────────────────────────────────────────

def parse_timestamp_from_text(text: str, fallback: Optional[datetime] = None) -> str:
    """Try to extract a date from text content. Fall back to provided datetime."""
    date_patterns = [
        r'\b(\d{4}[-/]\d{2}[-/]\d{2})\b',                      # 2024-01-15
        r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',                      # 15/01/2024
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:200], re.IGNORECASE)
        if match:
            try:
                raw = match.group(0)
                # Try ISO first
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"]:
                    try:
                        return datetime.strptime(raw, fmt).isoformat()
                    except ValueError:
                        pass
            except Exception:
                pass
    if fallback:
        return fallback.isoformat()
    return datetime.now().isoformat()


def split_into_segments(text: str, min_len: int = 80, max_len: int = 600) -> List[str]:
    """Split text into meaningful segments for memory extraction."""
    # Try paragraph splitting first
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    result = []

    for para in paras:
        if len(para) < min_len:
            continue
        if len(para) <= max_len:
            result.append(para)
        else:
            # Split long paragraphs by sentence
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_len:
                    current += " " + sent
                else:
                    if len(current) >= min_len:
                        result.append(current.strip())
                    current = sent
            if len(current) >= min_len:
                result.append(current.strip())

    return result


def parse_txt_md(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse .txt and .md files.
    Returns list of (timestamp_str, content) tuples.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    segments = split_into_segments(text)
    results = []

    for seg in segments:
        ts = parse_timestamp_from_text(seg, fallback=datetime.fromtimestamp(file_path.stat().st_mtime))
        results.append((ts, seg))

    return results


def parse_json_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse JSON files. Handles:
    - List of strings
    - List of {"content": ..., "timestamp": ...}
    - List of {"text": ..., "date": ...}
    - Nested dicts (recursively flatten)
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  [JSON] Invalid JSON in {file_path.name} — skipping")
        return []

    results = []

    def extract_from_dict(d: dict) -> Optional[Tuple[str, str]]:
        content = None
        timestamp = datetime.now().isoformat()

        for key in ["content", "text", "message", "body", "memory", "note"]:
            if key in d and isinstance(d[key], str) and len(d[key]) > 30:
                content = d[key]
                break

        for key in ["timestamp", "date", "time", "created_at", "updated_at"]:
            if key in d and isinstance(d[key], str):
                try:
                    timestamp = parse_timestamp_from_text(d[key], fallback=datetime.now())
                    break
                except Exception:
                    pass

        if content:
            return (timestamp, content)
        return None

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and len(item) > 30:
                results.append((datetime.now().isoformat(), item))
            elif isinstance(item, dict):
                extracted = extract_from_dict(item)
                if extracted:
                    results.append(extracted)
    elif isinstance(data, dict):
        # Try extracting from the dict itself or nested lists
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 30:
                        results.append((datetime.now().isoformat(), item))
                    elif isinstance(item, dict):
                        extracted = extract_from_dict(item)
                        if extracted:
                            results.append(extracted)

    return results


def parse_python_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse .py files — extract docstrings and comments as semantic memories.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    results = []

    # Extract module-level docstring
    module_doc = re.search(r'^"""(.*?)"""', text, re.DOTALL)
    if module_doc:
        content = f"Code file {file_path.name}: {module_doc.group(1).strip()}"
        if len(content) > 50:
            results.append((datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(), content))

    # Extract function docstrings with function names
    func_pattern = re.compile(
        r'def\s+(\w+)\s*\([^)]*\).*?:\s*\n\s*"""(.*?)"""',
        re.DOTALL
    )
    for match in func_pattern.finditer(text):
        func_name = match.group(1)
        docstring = match.group(2).strip()
        if len(docstring) > 30:
            content = f"Function `{func_name}` in {file_path.name}: {docstring[:300]}"
            results.append((datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(), content))

    # Extract inline comments (grouped)
    comment_blocks = []
    current_block = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('#!'):
            current_block.append(stripped[1:].strip())
        else:
            if len(current_block) >= 2:
                block_text = ' '.join(current_block)
                if len(block_text) > 40:
                    comment_blocks.append(block_text)
            current_block = []

    for block in comment_blocks[:5]:
        results.append((datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        f"Code comment in {file_path.name}: {block}"))

    return results


def parse_pdf_file(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse PDF files using pdfplumber.
    Install: pip install pdfplumber
    """
    try:
        import pdfplumber
    except ImportError:
        print(f"  [PDF] pdfplumber not installed. Install with: pip install pdfplumber")
        print(f"  [PDF] Skipping {file_path.name}")
        return []

    results = []
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"

        segments = split_into_segments(full_text)
        ts = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        for seg in segments:
            results.append((ts, seg))

    except Exception as e:
        print(f"  [PDF] Error reading {file_path.name}: {e}")

    return results


def parse_chat_export(file_path: Path) -> List[Tuple[str, str]]:
    """
    Parse WhatsApp/Telegram/iMessage chat exports.
    Handles formats like:
      [2024-01-15, 10:30 AM] John: Hello there!
      2024-01-15 10:30 - John: Hello there!
    Groups consecutive messages into conversation blocks.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split('\n')
    results = []

    # Detect chat format
    wa_pattern = re.compile(
        r'^\[?(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]?\s*-?\s*([^:]+):\s*(.*)'
    )

    current_block = []
    current_ts = None
    BLOCK_SIZE = 5  # Group N messages into one memory

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = wa_pattern.match(line)
        if match:
            date_str, time_str, sender, message = match.groups()
            try:
                for fmt in ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        dt = datetime.strptime(date_str.strip(), fmt)
                        current_ts = dt.isoformat()
                        break
                    except ValueError:
                        continue
            except Exception:
                current_ts = datetime.now().isoformat()

            current_block.append(f"{sender}: {message}")

            if len(current_block) >= BLOCK_SIZE:
                block_text = "\n".join(current_block)
                if len(block_text) > 80:
                    results.append((current_ts or datetime.now().isoformat(), block_text))
                current_block = []
        else:
            # Continuation of previous message
            if current_block:
                current_block[-1] += " " + line

    # Save remaining block
    if len(current_block) >= 2:
        block_text = "\n".join(current_block)
        if len(block_text) > 80:
            results.append((current_ts or datetime.now().isoformat(), block_text))

    return results


def parse_csv_file(file_path: Path) -> List[Tuple[str, str]]:
    """Parse CSV files — summarize each row as a memory."""
    import csv
    results = []
    try:
        with open(file_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i > 500:  # Limit to first 500 rows
                    break
                # Stringify the row
                content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and k)
                if len(content) > 50:
                    ts = datetime.now().isoformat()
                    # Look for date columns
                    for key in row:
                        if any(dk in key.lower() for dk in ["date", "time", "created", "timestamp"]):
                            ts = parse_timestamp_from_text(str(row[key]), fallback=datetime.now())
                            break
                    results.append((ts, content))
    except Exception as e:
        print(f"  [CSV] Error reading {file_path.name}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

PARSER_MAP = {
    ".txt": parse_txt_md,
    ".md":  parse_txt_md,
    ".json": parse_json_file,
    ".py":  parse_python_file,
    ".pdf": parse_pdf_file,
    ".csv": parse_csv_file,
}

# Files that look like chat exports get the chat parser
CHAT_INDICATORS = ["chat", "messages", "whatsapp", "telegram", "conversation"]


def ingest_file(file_path: Path) -> List[UserMemory]:
    """Parse a single file and return UserMemory objects."""
    ext = file_path.suffix.lower()
    stem = file_path.stem.lower()

    # Detect chat files by name
    if ext == ".txt" and any(ind in stem for ind in CHAT_INDICATORS):
        parser = parse_chat_export
    elif ext in PARSER_MAP:
        parser = PARSER_MAP[ext]
    else:
        return []

    try:
        raw_segments = parser(file_path)
    except Exception as e:
        print(f"  [Error] Failed to parse {file_path.name}: {e}")
        return []

    memories = []
    for ts, content in raw_segments:
        if not content or len(content) < 50:
            continue

        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()
        content = content[:800]  # Cap at 800 chars per memory

        topics = detect_topics(content)
        emotion = detect_emotion(content)
        entities = extract_entities(content)
        importance = estimate_importance(content, topics)
        mem_type = infer_memory_type(content, str(file_path))

        mem = UserMemory(
            event_id=f"usr_{content_hash(content)}",
            timestamp=ts,
            content=content,
            memory_type=mem_type,
            source_file=file_path.name,
            char_count=len(content),
            entities=entities,
            topics=topics,
            emotion=emotion,
            importance=importance,
            raw_segment=content,
        )
        memories.append(mem)

    return memories


def ingest_directory(input_dir: Path, dry_run: bool = False) -> List[UserMemory]:
    """Ingest all supported files from a directory recursively."""
    all_memories = []
    supported_exts = set(PARSER_MAP.keys())
    files = []

    for ext in supported_exts:
        files.extend(list(input_dir.glob(f"**/*{ext}")))
        files.extend(list(input_dir.glob(f"**/*{ext.upper()}")))

    # Deduplicate
    files = list(set(files))
    files.sort()

    if not files:
        print(f"[Ingestion] No supported files found in {input_dir}")
        print(f"  Supported: {', '.join(sorted(supported_exts))}")
        return []

    print(f"\n[Ingestion] Found {len(files)} files to process:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  📄 {f.name} ({size_kb:.1f} KB)")

    if dry_run:
        print("\n[DRY RUN] Would process these files. Use without --dry-run to ingest.")
        return []

    print()
    for file_path in files:
        memories = ingest_file(file_path)
        print(f"  ✓ {file_path.name}: {len(memories)} memories extracted")
        all_memories.extend(memories)

    return all_memories


def deduplicate_memories(memories: List[UserMemory]) -> List[UserMemory]:
    """Remove near-duplicate memories based on content hash."""
    seen_hashes = set()
    unique = []
    for mem in memories:
        h = content_hash(mem.content[:200])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(mem)
    return unique


def save_memories(memories: List[UserMemory], output_path: Path, reset: bool = False):
    """Save extracted memories, optionally appending to existing."""
    if output_path.exists() and not reset:
        existing = json.loads(output_path.read_text())
        print(f"\n[Save] Appending {len(memories)} to existing {len(existing)} memories")
        all_mems = existing + [asdict(m) for m in memories]
    else:
        all_mems = [asdict(m) for m in memories]

    output_path.write_text(json.dumps(all_mems, indent=2, ensure_ascii=False))
    size_kb = output_path.stat().st_size / 1024
    print(f"[Save] ✅ {output_path.name}: {len(all_mems)} total memories ({size_kb:.1f} KB)")


def print_summary(memories: List[UserMemory]):
    """Print extraction summary statistics."""
    if not memories:
        return

    from collections import Counter

    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total memories:     {len(memories)}")
    print(f"Total characters:   {sum(m.char_count for m in memories):,}")

    types = Counter(m.memory_type for m in memories)
    print("\nMemory types:")
    for t, c in types.most_common():
        print(f"  {t:<20} {c:>5}")

    topics = Counter(t for m in memories for t in m.topics)
    print("\nTop topics:")
    for t, c in topics.most_common(8):
        print(f"  {t:<20} {c:>5}")

    emotions = Counter(m.emotion for m in memories)
    print("\nEmotion distribution:")
    for e, c in emotions.most_common():
        print(f"  {e:<20} {c:>5}")

    sources = Counter(m.source_file for m in memories)
    print("\nBy source file:")
    for s, c in sources.most_common(10):
        print(f"  {s:<30} {c:>5}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest your personal files as training memories for Cortex Lab"
    )
    parser.add_argument("--input",    type=str, default=str(RAW_DIR),
                        help="Input folder containing your files (default: raw_data/)")
    parser.add_argument("--output",   type=str, default=str(OUT_FILE),
                        help="Output JSON file for extracted memories")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be ingested without processing")
    parser.add_argument("--reset",    action="store_true",
                        help="Clear existing memories before ingesting")
    parser.add_argument("--min-len",  type=int, default=80,
                        help="Minimum character length for a memory segment")
    parser.add_argument("--show",     action="store_true",
                        help="Print the first 5 extracted memories for inspection")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    print("\n" + "📥 " * 20)
    print("CORTEX LAB — USER FILE INGESTION")
    print("📥 " * 20)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_path}")

    if not input_dir.exists():
        print(f"\n[Error] Input directory does not exist: {input_dir}")
        print(f"Create it with: mkdir -p {input_dir}")
        print(f"Then drop your files there and run this script again.")
        sys.exit(1)

    # Ingest
    memories = ingest_directory(input_dir, dry_run=args.dry_run)

    if args.dry_run:
        return

    if not memories:
        print("\n[Warning] No memories extracted.")
        print("Make sure your files are in a supported format (.txt, .md, .json, .pdf, .py, .csv)")
        return

    # Deduplicate
    before = len(memories)
    memories = deduplicate_memories(memories)
    after = len(memories)
    if before != after:
        print(f"\n[Dedup] Removed {before - after} duplicate memories")

    # Show sample
    if args.show:
        print("\nSample extracted memories:")
        for m in memories[:5]:
            print(f"\n  [{m.timestamp[:10]}] ({m.memory_type}) — {m.content[:120]}...")

    # Save
    save_memories(memories, output_path, reset=args.reset)

    # Summary
    print_summary(memories)

    print(f"\n✅ Your memories are ready at: {output_path}")
    print(f"\nNext step:")
    print(f"  python scripts/generate_datasets.py --stage stage10_user_style --from-files {input_dir}")
    print(f"  python scripts/generate_datasets.py --all --quick-test  # Test all stages with your data")


if __name__ == "__main__":
    main()
