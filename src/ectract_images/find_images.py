"""
STEP 2 - given a question, return the figures whose caption matches it.

    python find_figures.py "difference between naive rag and advanced rag"
        -> prints the top matches WITH scores, so you can choose THRESHOLD

In your app:
    from find_figures import find_figures
    for f in find_figures(query):
        st.image(f["image_path"], caption=f"{f['paper']} p.{f['page']}: {f['caption']}")
An empty list means "no relevant figure", so nothing is displayed.
"""
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

FIGURES_JSON = "data/test_images/figures.json"
THRESHOLD = 0.30     # minimum similarity to show a figure; tune it with the command above
MAX_FIGURES = 2      # never show more than this

load_dotenv()
_key = os.environ.get("OPENROUTER_API_KEY")
if not _key:
    raise RuntimeError("OPENROUTER_API_KEY not found. Put it in your .env file.")
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_key)


def embed(texts):
    """Same embedding model as your pipeline. Returns unit-length vectors."""
    resp = _client.embeddings.create(model="openai/text-embedding-3-small", input=list(texts))
    vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


_cache = {}


def _load():
    if not _cache:
        figures = json.load(open(FIGURES_JSON))
        _cache["figures"] = figures
        _cache["vecs"] = embed([f["caption"] for f in figures]) if figures else np.zeros((0, 1))
    return _cache["figures"], _cache["vecs"]


def find_figures(query, threshold=THRESHOLD, top=MAX_FIGURES):
    figures, vecs = _load()
    if not figures or not query.strip():
        return []
    scores = vecs @ embed([query])[0]
    best = np.argsort(-scores)[:top]
    return [{**figures[i], "score": round(float(scores[i]), 3)} for i in best if scores[i] >= threshold]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "difference between naive rag and advanced rag"
    figures, vecs = _load()
    scores = vecs @ embed([q])[0]
    print(f"Query: {q}\n")
    for i in np.argsort(-scores)[:5]:
        shown = "SHOWN " if scores[i] >= THRESHOLD else "hidden"
        print(f"  {scores[i]:.3f} {shown} {figures[i]['paper']} {figures[i]['label']}: {figures[i]['caption'][:60]}")