"""
RAG Doctor retrieval: text chunks + figures.

    out = retrieve_context(question)

    out["chunks"]   -> the 10 reranked chunks (CONTEXT_K)
    out["figures"]  -> 0..5 figures (MAX_FIGURES) that pass FIGURE_THRESHOLD; each has
                       id, image_path, paper, page, label, caption, score
    out["context"]  -> chunks + figure captions (with ids) formatted as text for the LLM prompt
    out["figure_candidates"] -> every scored figure (for tuning the threshold)

Flow:
    question -> embed ONCE
        -> Chroma text chunks (TOP_K) -> LLM rerank -> top CONTEXT_K
        -> Chroma figure captions     -> caption similarity
                                         + evidence from the returned chunks
                                           (same paper / explicit "Figure N" citation)
                                      -> keep those scoring >= FIGURE_THRESHOLD (max MAX_FIGURES)

There is no separate image reranker: the answering LLM decides which of the returned
figures to show (see answer.py).

CLI (prints figure scores so you can tune FIGURE_THRESHOLD):
    python retrieve.py "difference between naive rag and advanced rag"
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from litellm import completion, embedding
from pydantic import BaseModel, Field

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma"

COLLECTION_NAME = os.getenv("RAG_DOCTOR_COLLECTION", "rag_doctor")
FIGURE_COLLECTION_NAME = os.getenv("RAG_DOCTOR_FIGURE_COLLECTION", "rag_doctor_figures")
EMBEDDING_MODEL = os.getenv("RAG_DOCTOR_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-small")
RERANKER_MODEL = os.getenv("RAG_DOCTOR_RERANKER_MODEL", "openrouter/deepseek/deepseek-v4.1-flash")
TOP_K = int(os.getenv("RAG_DOCTOR_TOP_K", "20"))
CONTEXT_K = 10               # reranked chunks returned for the answer LLM

N_FIGURE_CANDIDATES = 25     # figures scored per query (the figure set is small)
MAX_FIGURES = 5              # never return more than this (fewer if they don't pass the threshold)
FIGURE_THRESHOLD = 0.30      # minimum final figure score to return
DOC_BONUS = 0.05             # figure's paper appears among the context chunks
CITE_BONUS = 0.15            # a context chunk explicitly cites this figure ("Figure 3")

CITE = re.compile(r"\bfig(?:ure)?s?\.?\s*(\d+)", re.I)


class RankOrder(BaseModel):
    order: list[str] = Field(description="Chunk IDs ordered from most relevant to least relevant")


# --------------------------------------------------
# Chroma / LLM helpers
# --------------------------------------------------

@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(name: str = COLLECTION_NAME):
    return _client().get_collection(name)


def embed_query(question: str) -> list[float]:
    response = embedding(model=EMBEDDING_MODEL, input=[question])
    return response.data[0]["embedding"]


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = RERANKER_MODEL,
    response_format: type[BaseModel] | None = None,
) -> str:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format=response_format,
    )
    return response.choices[0].message.content


# --------------------------------------------------
# Text retrieval + rerank 
# --------------------------------------------------

def retrieve(
    question: str,
    top_k: int = TOP_K,
    query_vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    collection = get_collection()

    if query_vector is None:
        query_vector = embed_query(question)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result.get("distances", [[]])[0]
    ids = result["ids"][0]

    return [
        {
            "rank": rank,
            "id": chunk_id,
            "document": document or "",
            "metadata": metadata or {},
            "distance": distances[rank - 1] if distances else None,
        }
        for rank, (chunk_id, document, metadata) in enumerate(
            zip(ids, documents, metadatas), start=1
        )
    ]


def rerank(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = """
You are a document re-ranker.

You are provided with a question and a list of relevant chunks of text
retrieved from a knowledge base.

The chunks are provided in their original retrieval order. This order is
approximately based on relevance, but you may improve it.

Your task is ONLY to reorder the provided chunks by relevance to the question.
Return the chunk IDs from most relevant to least relevant.
Include every chunk ID exactly once.
Do not add, remove, or modify any chunk.
Reply only with the JSON structure requested.
"""

    user_prompt = f"""
The user has asked the following question:

{question}

Order all the chunks by relevance to the question,
from most relevant to least relevant.

Include every chunk ID exactly once.

Here are the chunks:
"""
    for chunk in chunks:
        user_prompt += f"# CHUNK ID: {chunk['id']}\n\n{chunk['document']}\n\n"

    raw = _call_llm(system_prompt, user_prompt, model=RERANKER_MODEL, response_format=RankOrder)
    evaluation = RankOrder.model_validate_json(raw)

    expected_ids = {chunk["id"] for chunk in chunks}
    if set(evaluation.order) != expected_ids or len(evaluation.order) != len(expected_ids):
        raise ValueError("Reranker did not return every chunk ID exactly once.")

    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    reranked = [chunks_by_id[chunk_id] for chunk_id in evaluation.order]

    return [{**chunk, "rank": rank} for rank, chunk in enumerate(reranked, start=1)]


def rerank_safe(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Production wrapper: a bad rerank response degrades to vector order instead of failing the request."""
    try:
        return rerank(question, chunks)
    except Exception as exc:
        print(f"[rerank] failed ({exc}); using vector order")
        return chunks


# --------------------------------------------------
# Figures (no LLM involved: embedding score + evidence from the returned chunks)
# --------------------------------------------------

def find_figures(
    query_vector: list[float],
    context_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (figures_to_return, all_scored_candidates)."""
    try:
        collection = get_collection(FIGURE_COLLECTION_NAME)
    except Exception as exc:
        print(f"[figures] collection '{FIGURE_COLLECTION_NAME}' unavailable ({exc}); run index_figures.py")
        return [], []

    count = collection.count()
    if not count:
        return [], []

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(N_FIGURE_CANDIDATES, count),
        include=["documents", "metadatas", "distances"],
    )

    top_docs = {c["metadata"].get("doc_id") for c in context_chunks}
    cited = {
        (c["metadata"].get("doc_id"), f"Figure {num}")
        for c in context_chunks
        for num in CITE.findall(c["document"])
    }

    candidates = []
    for fig_id, caption, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1 - distance  # collection uses cosine space
        in_top_doc = meta["paper"] in top_docs
        is_cited = (meta["paper"], meta["label"]) in cited
        score = similarity + (DOC_BONUS if in_top_doc else 0) + (CITE_BONUS if is_cited else 0)
        candidates.append(
            {
                **meta,
                "id": fig_id,
                "caption": caption,
                "similarity": round(similarity, 3),
                "score": round(score, 3),
                "in_top_doc": in_top_doc,
                "cited": is_cited,
            }
        )

    candidates.sort(key=lambda f: -f["score"])
    figures = [f for f in candidates if f["score"] >= FIGURE_THRESHOLD][:MAX_FIGURES]
    return figures, candidates


# --------------------------------------------------
# Public entry point
# --------------------------------------------------

def format_context(chunks: list[dict[str, Any]], figures: list[dict[str, Any]]) -> str:
    """Chunks + figure captions as one text block, ready to drop into the LLM prompt."""
    text = "\n\n".join(
        f"[{chunk['rank']}] "
        f"{chunk['metadata'].get('doc_id', 'unknown')} | "
        f"{chunk['metadata'].get('section_path', 'unknown')}\n"
        f"{chunk['document']}"
        for chunk in chunks
    )

    if figures:
        lines = []
        for fig in figures:
            page = f", p.{fig['page']}" if fig.get("page", -1) >= 0 else ""
            lines.append(
                f"- id={fig['id']} | {fig['label'] or 'Figure'} ({fig['paper']}{page}): {fig['caption'][:400]}"
            )
        text += "\n\nAVAILABLE FIGURES:\n" + "\n".join(lines)

    return text


def retrieve_context(question: str) -> dict[str, Any]:
    query_vector = embed_query(question)  # embedded once, used for both collections

    chunks = retrieve(question, TOP_K, query_vector)
    if not chunks:
        return {"chunks": [], "figures": [], "context": "", "figure_candidates": []}

    ranked = rerank_safe(question, chunks)
    context_chunks = ranked[:CONTEXT_K]

    try:  # figures are optional: never let them break text retrieval
        figures, candidates = find_figures(query_vector, context_chunks)
    except Exception as exc:
        print(f"[figures] skipped ({exc})")
        figures, candidates = [], []

    return {
        "chunks": context_chunks,
        "figures": figures,
        "context": format_context(context_chunks, figures),
        "figure_candidates": candidates,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "difference between naive rag and advanced rag"
    out = retrieve_context(question)

    print(f"Query: {question}\n\nContext chunks:")
    for chunk in out["chunks"]:
        meta = chunk["metadata"]
        print(f"  {chunk['rank']}. {meta.get('doc_id')} | {meta.get('section_path')}: {chunk['document'][:70]!r}")

    print("\nFigure candidates (top 8):")
    for fig in out["figure_candidates"][:8]:
        returned = "RETURNED" if fig in out["figures"] else "hidden  "
        flags = f"doc={'Y' if fig['in_top_doc'] else 'n'} cited={'Y' if fig['cited'] else 'n'}"
        print(
            f"  {fig['score']:.3f} (sim {fig['similarity']:.3f}) {returned} {flags} "
            f"{fig['paper']} {fig['label']}: {fig['caption'][:50]}"
        )