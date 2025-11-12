import json
import math
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Protocol
from urllib.parse import urlparse

import requests

# Prevent transformers from importing heavy optional backends (TF, Flax) that may be unavailable.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

# Lazy import for heavy ML libraries to speed up cold start
# CrossEncoder imported in _get_cross_encoder() only when needed

from utils.azure_llm import stream_chat
from utils.supabase_client import get_supabase_client

from dotenv import load_dotenv
load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "google/embeddinggemma-300m")

AZURE_ORCHESTRATOR_DEPLOYMENT = os.getenv("AZURE_PHI4_ORCHESTRATOR") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

SUPABASE_MATCH_FUNCTION = os.getenv("SUPABASE_MATCH_FUNCTION", "match_document_chunks")
SUPABASE_DEFAULT_MATCH_COUNT = int(os.getenv("SUPABASE_MATCH_COUNT", "10"))
# Increased to 100 - transportation chunks score ~0.40 but compete with residency docs at 0.42-0.54
# Wider funnel ensures relevant chunks make it to reranking stage
SUPABASE_INITIAL_MATCH_COUNT = int(os.getenv("SUPABASE_INITIAL_MATCH_COUNT", "100"))
SUPABASE_CHUNKS_TABLE = os.getenv("SUPABASE_CHUNKS_TABLE", "chunks")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CONTEXT_MAX_SNIPPETS = int(os.getenv("RAG_CONTEXT_SNIPPETS", "10"))
NEIGHBOR_RADIUS = int(os.getenv("RAG_NEIGHBOR_RADIUS", "1"))
LOW_SCORE_RETRY_THRESHOLD = float(os.getenv("RAG_LOW_SCORE_RETRY_THRESHOLD", "0.2"))
RETRY_SCALE = int(os.getenv("RAG_RETRIEVE_RETRY_MULTIPLIER", "2"))

class MCPClientProtocol(Protocol):
    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

def get_system_prompt() -> str:
    text = os.getenv("RAG_SYSTEM_PROMPT")
    if not text:
        raise RuntimeError("Missing required environment variable: RAG_SYSTEM_PROMPT")
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text

def require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# Token estimator
_WORD_OR_PUNC = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_OR_PUNC.findall(str(text)))

@lru_cache(maxsize=128)
def embed_query(text: str) -> tuple:
    """
    Embed query text using HuggingFace API with caching.
    Returns tuple for hashability (lru_cache requirement).
    """
    token = require_env(HUGGINGFACEHUB_API_TOKEN, "HUGGINGFACEHUB_API_TOKEN")
    model = require_env(HUGGINGFACE_MODEL, "HUGGINGFACE_EMBEDDING_MODEL")
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Hugging Face error ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected Hugging Face response: {data}")
    first = data[0]
    vec = first.get("embedding") if isinstance(first, dict) else first
    if not isinstance(vec, list):
        raise RuntimeError(f"Invalid embedding payload: {first}")
    floats = [float(x) for x in vec]
    normalized = _l2_normalize(floats)
    return tuple(normalized)  # Return tuple for hashability

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """
    Lazy-load the cross-encoder once per process.
    Imports sentence_transformers only when needed to speed up cold start.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_MODEL)


def _rerank_hits(query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not hits:
        return []

    # Rerank top 20 candidates - balances quality and speed
    # Increased from 12 to catch more relevant docs that were filtered too early
    candidates = hits[:min(20, len(hits))]

    ce = _get_cross_encoder()
    # Use full 1200 chars for better semantic matching (chunks avg ~700-960 chars)
    # Increased from 800 to avoid truncating important content at end of chunks
    pairs = [(query, h.get("doc", "")[:1200]) for h in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda t: float(t[1]), reverse=True)
    reranked: List[Dict[str, Any]] = []
    for hit, score in ranked[:max(1, top_k)]:
        updated = dict(hit)
        updated["rerank_score"] = float(score)
        reranked.append(updated)
    return reranked


def _fetch_neighbor_chunks(base_hits: List[Dict[str, Any]], radius: int) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if radius <= 0:
        return {}
    doc_map: Dict[str, set[int]] = defaultdict(set)
    for hit in base_hits:
        doc_id = hit.get("document_id") or (hit.get("meta") or {}).get("document_id")
        chunk_index = hit.get("chunk_index") or (hit.get("meta") or {}).get("chunk_index")
        if doc_id is None or chunk_index is None:
            continue
        for delta in range(1, radius + 1):
            if chunk_index - delta >= 0:
                doc_map[doc_id].add(chunk_index - delta)
            doc_map[doc_id].add(chunk_index + delta)
    if not doc_map:
        return {}

    client = get_supabase_client()
    neighbors: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for doc_id, indices in doc_map.items():
        wanted = sorted(i for i in indices if i is not None and i >= 0)
        if not wanted:
            continue
        try:
            resp = (
                client.table(SUPABASE_CHUNKS_TABLE)
                .select("id, document_id, chunk_index, content, section_title, metadata, filename, category, canonical")
                .eq("document_id", doc_id)
                .in_("chunk_index", wanted)
                .execute()
            )
        except Exception:
            continue
        rows = getattr(resp, "data", []) or []
        for row in rows:
            meta = row.get("metadata") or {}
            merged_meta = {
                **meta,
                "section_title": row.get("section_title") or meta.get("section_title"),
                "filename": row.get("filename") or meta.get("filename"),
                "category": row.get("category") or meta.get("category"),
                "canonical": row.get("canonical") or meta.get("canonical"),
                "chunk_id": row.get("id") or meta.get("id"),
                "document_id": row.get("document_id") or meta.get("document_id"),
                "chunk_index": row.get("chunk_index"),
            }
            neighbors[(doc_id, row.get("chunk_index"))] = {
                "doc": row.get("content") or "",
                "meta": merged_meta,
                "score": None,
                "document_id": merged_meta.get("document_id"),
                "chunk_index": merged_meta.get("chunk_index"),
                "is_neighbor": True,
            }
    return neighbors


def _expand_with_neighbors(hits: List[Dict[str, Any]], *, max_snippets: int) -> List[Dict[str, Any]]:
    if not hits:
        return []
    base = hits[:max_snippets]
    neighbor_lookup = _fetch_neighbor_chunks(base, NEIGHBOR_RADIUS)
    combined: List[Dict[str, Any]] = []
    seen: set[Tuple[Optional[str], Optional[int], Optional[str]]] = set()

    def _add(hit: Dict[str, Any]) -> None:
        meta = hit.get("meta") or {}
        key = (
            hit.get("document_id") or meta.get("document_id"),
            hit.get("chunk_index") or meta.get("chunk_index"),
            meta.get("chunk_id"),
        )
        if key in seen:
            return
        seen.add(key)
        combined.append(hit)

    for hit in base:
        _add(hit)
        doc_id = hit.get("document_id") or (hit.get("meta") or {}).get("document_id")
        chunk_index = hit.get("chunk_index") or (hit.get("meta") or {}).get("chunk_index")
        if doc_id is None or chunk_index is None:
            continue
        for delta in range(1, NEIGHBOR_RADIUS + 1):
            for idx in (chunk_index - delta, chunk_index + delta):
                neighbor = neighbor_lookup.get((doc_id, idx))
                if neighbor:
                    _add(neighbor)
        if len(combined) >= max_snippets:
            break
    return combined[:max_snippets]

def retrieve_matches(
    query: str,
    match_count: Optional[int] = None,
    extra_filter: Optional[Dict[str, Any]] = None,
    *,
    initial_override: Optional[int] = None,
    embedding_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    embedding_source = embedding_text if embedding_text is not None else query

    # Use clean embeddings (no title prefix) for better semantic matching
    # This matches the updated ingestion format for consistent retrieval
    embedding = list(embed_query(embedding_source))  # Convert tuple to list

    desired = match_count or SUPABASE_DEFAULT_MATCH_COUNT
    initial = max(initial_override or SUPABASE_INITIAL_MATCH_COUNT, desired)
    payload: Dict[str, Any] = {
        "query_embedding": embedding,
        "match_count": initial,
    }
    filter_env = os.getenv("SUPABASE_MATCH_FILTER")
    filt = extra_filter or {}
    if filter_env:
        try:
            filt.update(json.loads(filter_env))
        except json.JSONDecodeError:
            pass
    if filt:
        payload["filter"] = filt
    resp = client.rpc(SUPABASE_MATCH_FUNCTION, payload).execute()
    data = getattr(resp, "data", []) or []
    hits: List[Dict[str, Any]] = []
    for item in data:
        meta = item.get("metadata") or {}
        doc = item.get("content") or item.get("chunk") or meta.get("content") or ""
        sim = item.get("similarity") or item.get("score")
        doc_id = item.get("document_id") or meta.get("document_id")
        chunk_index = item.get("chunk_index") or meta.get("chunk_index")
        hits.append(
            {
                "doc": doc,
                "meta": {
                    **meta,
                    "section_title": item.get("section_title") or meta.get("section_title"),
                    "filename": item.get("filename") or meta.get("filename"),
                    "category": item.get("category") or meta.get("category"),
                    "canonical": item.get("canonical") or meta.get("canonical"),
                    "chunk_id": item.get("id") or meta.get("id"),
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                },
                "score": sim,
                "document_id": doc_id,
                "chunk_index": chunk_index,
            }
        )

    return _rerank_hits(query, hits, top_k=desired)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_relevant_sentence(doc: str, query: str) -> str:
    doc = (doc or "").strip()
    query = (query or "").strip()
    if not doc or not query:
        return ""
    sentences = _SENTENCE_SPLIT.split(doc)
    lowered = query.lower()
    for sentence in sentences:
        if lowered in sentence.lower():
            return sentence.strip()
    return sentences[0].strip() if sentences else ""


def format_context(hits: List[Dict[str, Any]], *, limit: Optional[int] = None, query: str = "") -> str:
    if not hits:
        return "No relevant context found."
    cap = limit or CONTEXT_MAX_SNIPPETS
    blocks = []
    for i, h in enumerate(hits[:cap], 1):
        meta = h.get("meta") or {}
        title = meta.get("section_title") or meta.get("title") or meta.get("filename") or "Section"
        block = f"Source {i}: {title}\n{h.get('doc', '')}"
        quote = _extract_relevant_sentence(h.get("doc", ""), query)
        if quote:
            block += f"\n> Quote: {quote}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""

    def short_url(u: str) -> str:
        if not u or not isinstance(u, str) or not u.startswith("http"):
            return ""
        p = urlparse(u)
        disp = (p.netloc + p.path).rstrip("/")
        return disp

    lines, seen = [], set()
    i = 0
    for h in hits:
        m = h.get("meta") or {}
        title = m.get("section_title") or m.get("filename") or "Untitled"
        cat = m.get("category") or "Orientation"
        file = m.get("filename") or ""
        canon = (m.get("canonical") or "").strip()

        key = (title, canon)
        if key in seen:
            continue
        seen.add(key)

        i += 1
        suffix = f" · {file}" if file else ""
        if canon:
            disp = short_url(canon)
            lines.append(f"{i}. [{title}]({canon}) — {cat}{suffix}" + (f"\n    ↳ {disp}" if disp else ""))
        else:
            lines.append(f"{i}. {title} — {cat}{suffix}")

        if i >= min(5, CONTEXT_MAX_SNIPPETS):
            break

    return "\n".join(lines)

def _augment_query(user_text: str) -> str:
    lowered = (user_text or "").lower()
    keywords: List[str] = []

    if "orientation" in lowered:
        keywords.extend([
            "orientation",
            "orientation dates",
            "orientation fees",
            "orientation schedule",
            "myorientation",
        ])
        if "international" in lowered:
            keywords.extend([
                "international orientation",
                "glo-bull beginnings",
                "international student orientation",
                "mybullspath",
            ])
        if any(term in lowered for term in ("freshman", "first-year", "ftic")):
            keywords.append("first-year orientation")

    if "international" in lowered and "orientation" not in lowered:
        keywords.extend([
            "international student services",
            "glo-bull beginnings",
        ])

    if keywords:
        dedup = []
        seen = set()
        for word in keywords:
            if word not in seen:
                dedup.append(word)
                seen.add(word)
        return f"{user_text}\n\nRelated keywords: {', '.join(dedup)}"
    return user_text


def generate_with_rag(
    user_text: str,
    match_count: Optional[int] = None,
    mcp_client: Optional[MCPClientProtocol] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    augmented_query = _augment_query(user_text)
    desired = match_count or CONTEXT_MAX_SNIPPETS
    if mcp_client:
        base_hits = mcp_client.retrieve_context(user_text, match_count=max(desired, CONTEXT_MAX_SNIPPETS))
    else:
        base_hits = retrieve_matches(
            augmented_query,
            match_count=max(desired, CONTEXT_MAX_SNIPPETS),
            embedding_text=user_text,
        )
        low_confidence = (
            not base_hits
            or (base_hits and base_hits[0].get("rerank_score", 0.0) < LOW_SCORE_RETRY_THRESHOLD)
        )
        if low_confidence and RETRY_SCALE > 1:
            base_hits = retrieve_matches(
                augmented_query,
                match_count=max(desired, CONTEXT_MAX_SNIPPETS),
                initial_override=SUPABASE_INITIAL_MATCH_COUNT * RETRY_SCALE,
                embedding_text=user_text,
            )
    expanded_hits = _expand_with_neighbors(base_hits or [], max_snippets=CONTEXT_MAX_SNIPPETS)
    system = get_system_prompt()
    ctx = format_context(expanded_hits, limit=CONTEXT_MAX_SNIPPETS, query=user_text)

    messages = [
        {"role": "system", "content": f"{system}\n\nCONTEXT:\n{ctx}"},
        {"role": "user", "content": user_text},
    ]

    response_text = ""
    for delta in stream_chat(AZURE_ORCHESTRATOR_DEPLOYMENT, messages):
        response_text += delta
        yield ("delta", {"text": response_text})

    sources_block = build_sources_block(expanded_hits)
    final = (response_text or "").rstrip()
    if sources_block:
        final += "\n\n**Sources**\n" + sources_block
    yield ("final", {"text": final, "hits": expanded_hits})
