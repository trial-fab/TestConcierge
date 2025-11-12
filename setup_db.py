from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
from supabase import Client, create_client
import requests

load_dotenv()

# env helpers
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "google/embeddinggemma-300m")
EMBEDDER_ID = f"hf::{HUGGINGFACE_MODEL}"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY")

DEFAULT_DOCUMENTS_TABLE = os.getenv("SUPABASE_DOCUMENTS_TABLE", "documents")
DEFAULT_CHUNKS_TABLE = os.getenv("SUPABASE_CHUNKS_TABLE", "chunks")
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_supabase_client(url: Optional[str], key: Optional[str]) -> Client:
    url = require_env(url, "SUPABASE_URL")
    key = require_env(key, "SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


# constants & regex
MD_EXT = {".md"}

_WS = re.compile(r"[ \t\f\v]+")
_NL = re.compile(r"\n{3,}")
PARA_BREAK = re.compile(r"\n\s*\n")
BULLET = re.compile(r"^\s*(?:[\u2022\-\*\u25E6]|\d+\.)\s+")
HEADER = re.compile(r"^#{1,6}\s+")
YAML_FRONT = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)
USF_LINK = re.compile(r"https?://(?:www\.)?usf\.edu[^\s\]\)]+", re.I)

# Navigation patterns to strip (reduces duplicate pollution)
SKIP_TO_CONTENT = re.compile(r"\[Skip (?:to|Over) [^\]]+\]\([^\)]+\)", re.I)
BREADCRUMB_NAV = re.compile(r"## Breadcrumb Navigation.*?(?=\n##|\n\*|\Z)", re.S | re.I)
MAIN_NAV_SECTION = re.compile(r"## Main Navigat(?:ion)?.*?(?=\n##|\n\*|\Z)", re.S | re.I)
# Generic footer navigation (About USF, Academics, Admissions, etc.)
FOOTER_NAV = re.compile(
    r"^\* \[(?:About USF|Academics|Admissions|Locations|Campus Life|Research|Administrative Units|"
    r"Regulations & Policies|Human Resources|Work at USF|Emergency & Safety)\]\(https://www\.usf\.edu/[^\)]+\)\s*$",
    re.M
)


# text helpers
def first_usf_url(text: str) -> Optional[str]:
    """Return the first usf.edu URL (without fragment) if present."""
    if not text:
        return None
    m = USF_LINK.search(text)
    if not m:
        return None
    url = m.group(0)
    return url.split("#", 1)[0]

def derive_category(path: Path) -> str:
    """Use top-level folder name as category when available; default 'USF'."""
    try:
        parts = path.parts
        return parts[0] if len(parts) > 1 else "USF"
    except Exception:
        return "USF"

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def strip_navigation(text: str) -> str:
    """
    Remove common USF website navigation elements that pollute vector search.
    Preserves actual content and citation metadata.
    """
    # Strip skip-to-content links
    text = SKIP_TO_CONTENT.sub("", text)

    # Strip breadcrumb navigation sections
    text = BREADCRUMB_NAV.sub("", text)

    # Strip main navigation sections
    text = MAIN_NAV_SECTION.sub("", text)

    # Strip footer navigation (the 66x duplicate)
    text = FOOTER_NAV.sub("", text)

    # Clean up extra whitespace left by removals
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = YAML_FRONT.sub("", s, count=1)  # strip YAML front-matter if present
    s = strip_navigation(s)  # Remove navigation pollution
    s = _WS.sub(" ", s).strip()
    s = _NL.sub("\n\n", s)
    return s.strip()

def reflow_paragraphs(text: str) -> str:
    """
    - Keep real paragraph breaks (blank lines)
    - Join hard-wrapped lines within a paragraph into one line
    - Preserve bullet lines
    - De-hyphenate soft wraps: immuniza-\ntion -> immunization
    """
    text = re.sub(r"-\n(?=[a-z])", "", text)
    paras = PARA_BREAK.split(text)
    out: List[str] = []
    for p in paras:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            out.append("")
            continue
        buf: List[str] = []
        for ln in lines:
            if HEADER.match(ln):
                if buf:
                    out.append(" ".join(buf))
                    buf = []
                out.append(ln)
            elif BULLET.match(ln):
                if buf:
                    out.append(" ".join(buf))
                    buf = []
                out.append(ln)
            else:
                buf.append(ln)
        if buf:
            out.append(" ".join(buf))
        out.append("")
    return "\n".join(out).strip()

def group_faq_blocks(text: str) -> str:
    """Keep each Q/A pair self-contained without merging unrelated blocks."""
    q_start = re.compile(r"^\s*(?:Q[:\-\)]|Question\b|How\b|What\b|When\b|Where\b|Why\b|Can\b|Do\b|Does\b|Is\b|Are\b)", re.I)
    a_mark = re.compile(r"^\s*A[:\-\)]\s*", re.I)
    lines = text.split("\n")
    blocks: List[str] = []
    current_block: List[str] = []
    saw_q = False
    capturing_answer = False

    def _flush():
        nonlocal current_block
        if current_block:
            blocks.append("\n".join(current_block).strip())
            current_block = []

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            _flush()
            blocks.append("")
            capturing_answer = False
            continue
        if q_start.match(stripped):
            saw_q = True
            _flush()
            current_block.append(stripped)
            capturing_answer = True
            continue
        if capturing_answer:
            current_block.append(a_mark.sub("", ln, count=1))
        else:
            _flush()
            blocks.append(ln)

    _flush()
    return "\n".join(blocks) if saw_q else text

def recursive_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf:
                pieces.append(buf)
            if len(p) <= chunk_size:
                buf = p
            else:
                sents = re.split(r"(?<=[.!?])\s+", p)
                sbuf = ""
                for s in sents:
                    if len(sbuf) + len(s) + 1 <= chunk_size:
                        sbuf = (sbuf + " " + s).strip() if sbuf else s
                    else:
                        if sbuf:
                            pieces.append(sbuf)
                        step = max(1, chunk_size - overlap)
                        for i in range(0, len(s), step):
                            pieces.append(s[i:i + chunk_size])
                        sbuf = ""
                if sbuf:
                    pieces.append(sbuf)
                buf = ""
    if buf:
        pieces.append(buf)
    out, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 1 <= chunk_size:
            cur = cur + "\n" + p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    if overlap > 0 and len(out) > 1:
        with_ol = [out[0]]
        for i in range(1, len(out)):
            prev_tail = _tail_snippet(out[i - 1], overlap)
            if prev_tail:
                combined = (prev_tail + "\n\n" + out[i]).strip()
            else:
                combined = out[i]
            with_ol.append(combined)
        out = with_ol
    return out

def glue_short_chunks(chunks: List[str], min_chars: int = 200) -> List[str]:
    """
    Merge very short chunks to avoid fragmentation.
    Reduced from 300 to 200 to keep chunks closer to target size (700).
    """
    if not chunks:
        return chunks
    out: List[str] = []
    for ch in chunks:
        if out and len(out[-1]) < min_chars:
            out[-1] = (out[-1].rstrip() + "\n\n" + ch.lstrip()).strip()
        else:
            out.append(ch)
    return out

def fingerprint(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

def _tail_snippet(text: str, target_chars: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    collected: List[str] = []
    total = 0
    for sentence in reversed(sentences):
        if not sentence:
            continue
        collected.insert(0, sentence)
        total += len(sentence)
        if total >= target_chars:
            break
    return " ".join(collected).strip()


def l2_normalize(vec: List[float]) -> List[float]:
    """Return a unit-length version of vec to keep embeddings comparable."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _format_for_embedding(text: str, title: Optional[str]) -> str:
    """
    Return cleaned text for embedding (no title prefix, no URLs).

    Strips URLs to reduce noise in embeddings while preserving semantic content.
    URLs dilute embedding quality (seen 35-62% URL ratio in some chunks).

    Note: URLs are preserved in stored chunk content for citations.
    Only embedding text is cleaned.
    """
    # Strip URLs - they add noise but no semantic value
    # Keeps: "Visit HART online" but removes: "https://www.hart.org/..."
    cleaned = re.sub(r'https?://\S+', '', text)
    # Remove markdown link syntax that remains after URL removal
    cleaned = re.sub(r'\[([^\]]+)\]\(\)', r'\1', cleaned)
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def iter_md_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        if p.is_file():
            yield p

def md_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.*)$", text, flags=re.M)
    title = (m.group(1).strip() if m else None) or fallback
    return title[:200]

# embedding + supabase
def _hf_request(payload: dict) -> list:
    model = require_env(HUGGINGFACE_MODEL, "HUGGINGFACE_EMBEDDING_MODEL")
    token = require_env(os.getenv("HUGGINGFACEHUB_API_TOKEN"), "HUGGINGFACEHUB_API_TOKEN")
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"Hugging Face error ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Hugging Face response: {data}")
    return data

def embed_texts(texts: List[str], titles: Optional[List[str]] = None, batch: int = 8) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), max(1, batch)):
        subset = texts[i:i + batch]
        formatted: List[str] = []
        for j, chunk in enumerate(subset):
            title = None
            if titles and (i + j) < len(titles):
                title = titles[i + j]
            # Format chunk for embedding (strips URLs for cleaner embeddings)
            # Original chunk text (with URLs) is preserved in 'subset' for storage
            formatted.append(_format_for_embedding(chunk, title))
        payload = {"inputs": formatted, "options": {"wait_for_model": True}}
        data = _hf_request(payload)
        # The inference API returns one embedding per input (list of floats) or dict with "embedding".
        parsed: List[List[float]] = []
        for item in data:
            vec = item.get("embedding") if isinstance(item, dict) else item
            if not isinstance(vec, list):
                raise RuntimeError(f"Invalid embedding payload: {item}")
            parsed.append(l2_normalize([float(x) for x in vec]))
        if len(parsed) != len(subset):
            raise RuntimeError("Mismatch between inputs and embeddings from Hugging Face.")
        vectors.extend(parsed)
    return vectors

def ensure_document(
    client: Client,
    documents_table: str,
    doc_fp: str,
    title: str,
    source_path: str,
    category: str,
) -> tuple[str, bool]:
    """Return (document_id, existed)."""
    try:
        resp = (
            client.table(documents_table)
            .select("id, checksum")
            .eq("checksum", doc_fp)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", []) or []
        if rows:
            return rows[0]["id"], True
    except Exception:
        pass

    doc_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat(timespec="seconds")
    record = {
        "id": doc_id,
        "title": title,
        "source_path": source_path,
        "category": category,
        "checksum": doc_fp,
        "created_at": now,
        "updated_at": now,
    }
    client.table(documents_table).insert(record).execute()
    return doc_id, False

def delete_existing_chunks(client: Client, chunks_table: str, document_id: str) -> None:
    client.table(chunks_table).delete().eq("document_id", document_id).execute()

def insert_chunks(
    client: Client,
    chunks_table: str,
    document_id: str,
    chunks: List[str],
    titles: List[str],
    metas: List[Dict[str, object]],
    embeddings: List[List[float]],
) -> None:
    rows = []
    for idx, (content, title, meta, embed) in enumerate(zip(chunks, titles, metas, embeddings)):
        rows.append({
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "chunk_index": idx,
            "content": content,  # Original text with URLs intact (for citations)
            "section_title": title,
            "metadata": meta,
            "embedding": embed,
            "chunk_fp": meta["chunk_fp"],
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        })
    if rows:
        client.table(chunks_table).upsert(rows, on_conflict="document_id,chunk_index").execute()


def purge_existing_corpus(client: Client, documents_table: str, chunks_table: str) -> None:
    """Remove all existing chunks/documents ahead of a full refresh."""
    print(f"[refresh] Clearing existing data from '{chunks_table}' and '{documents_table}' …")
    client.table(chunks_table).delete().neq("id", ZERO_UUID).execute()
    client.table(documents_table).delete().neq("id", ZERO_UUID).execute()
    print("[refresh] Tables emptied.")

def main():
    ap = argparse.ArgumentParser(
        description="RAG ingestion (MD → chunks → Azure embeddings → Supabase pgvector)"
    )
    ap.add_argument("--source", default="data/raw", help="Folder containing .md files (recurses)")
    ap.add_argument("--supabase-url", default=SUPABASE_URL, help="Supabase project URL")
    ap.add_argument("--supabase-key", default=SUPABASE_SERVICE_KEY, help="Supabase service role key")
    ap.add_argument("--documents-table", default=DEFAULT_DOCUMENTS_TABLE, help="Supabase documents table")
    ap.add_argument("--chunks-table", default=DEFAULT_CHUNKS_TABLE, help="Supabase chunks table")
    ap.add_argument("--chunk", type=int, default=700, help="Target chunk size (chars)")
    ap.add_argument("--overlap", type=int, default=220, help="Chunk overlap (chars)")
    ap.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    ap.add_argument(
        "--skip-unchanged",
        action="store_true",
        default=False,
        help="Skip docs whose checksum already exists in Supabase",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse and report without writing to Supabase",
    )
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"[!] Source not found: {src}")

    client = None if args.dry_run else get_supabase_client(args.supabase_url, args.supabase_key)

    if not args.dry_run:
        purge_existing_corpus(client, args.documents_table, args.chunks_table)

    scanned = added = skipped = 0

    for path in iter_md_files(src):
        scanned += 1
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if not raw.strip():
                print(f"[skip empty] {path}")
                skipped += 1
                continue

            text = group_faq_blocks(reflow_paragraphs(clean_text(raw)))
            if not text.strip():
                print(f"[skip empty] {path}")
                skipped += 1
                continue

            doc_fp = sha1(" ".join(text.split()))
            base_id = f"{path.relative_to(src)}".replace("\\", "/")
            title = md_title(text, fallback=path.stem)
            category = derive_category(path.relative_to(src))
            canonical = first_usf_url(raw) or ""

            if args.dry_run:
                print(f"[dry] {base_id} checksum={doc_fp[:12]} len={len(text)}")
                continue

            document_id, existed = ensure_document(
                client,
                args.documents_table,
                doc_fp,
                title,
                base_id,
                category,
            )

            if existed and args.skip_unchanged:
                print(f"[skip unchanged] {path}")
                skipped += 1
                continue

            raw_chunks = recursive_chunks(text, args.chunk, args.overlap)
            raw_chunks = glue_short_chunks(raw_chunks, min_chars=200)

            seen = set()
            chunks: List[str] = []
            metas: List[Dict[str, object]] = []
            for ch in raw_chunks:
                fp = fingerprint(ch)
                if fp in seen:
                    continue
                seen.add(fp)
                chunks.append(ch)
                metas.append({
                    "source": base_id,
                    "filename": path.name,
                    "relpath": base_id,
                    "section_title": title,
                    "len": len(ch),
                    "doc_fp": doc_fp,
                    "chunk_fp": fp,
                    "category": category,
                    "canonical": canonical,
                    "embedder": EMBEDDER_ID,
                    "document_id": document_id,
                })

            if not chunks:
                print(f"[skip no-chunks] {path}")
                skipped += 1
                continue

            if existed:
                delete_existing_chunks(client, args.chunks_table, document_id)

            titles = [title] * len(chunks)
            embeddings = embed_texts(chunks, titles=titles, batch=args.batch)
            metas_for_rows = []
            for idx, meta in enumerate(metas):
                enriched = {**meta, "chunk_index": idx}
                metas_for_rows.append(enriched)

            insert_chunks(
                client,
                args.chunks_table,
                document_id,
                chunks,
                titles,
                metas_for_rows,
                embeddings,
            )

            added += len(chunks)
            print(f"[ok] {path} → {len(chunks)} chunks")

        except Exception as e:
            print(f"[error] {path}: {e}")
            skipped += 1

    print("\n=== Ingestion Summary ===")
    print(f"Scanned files : {scanned}")
    print(f"Chunks added  : {added}")
    print(f"Skipped       : {skipped}")
    if not args.dry_run:
        print(f"Supabase URL  : {args.supabase_url}")
        print(f"Documents tbl : {args.documents_table}")
        print(f"Chunks tbl    : {args.chunks_table}")
        print(f"Embed model   : {require_env(HUGGINGFACE_MODEL, 'HUGGINGFACE_EMBEDDING_MODEL')}")
        print(f"Chunk/Overlap : {args.chunk}/{args.overlap}")

if __name__ == "__main__":
    main()
