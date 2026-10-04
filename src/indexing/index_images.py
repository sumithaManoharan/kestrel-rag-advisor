"""
Embed figure captions (figures.json) into their OWN Chroma collection.

    python index_figures.py

Run it after the PDF parsing step. It is independent of embed.py, so rebuilding
the text collection never touches the figures (and vice versa).
"""
import json
import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import embedding

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # same layout as embed.py
IMAGES_DIR = PROJECT_ROOT / os.getenv("RAG_DOCTOR_IMAGES_DIR", "data/images")  # PNGs + figures.json
FIGURES_JSON = IMAGES_DIR / "figures.json"
CHROMA_DIR = PROJECT_ROOT / "vectorstore" / "chroma"
COLLECTION_NAME = os.getenv("RAG_DOCTOR_FIGURE_COLLECTION", "rag_doctor_figures")
EMBEDDING_MODEL = os.getenv(
    "RAG_DOCTOR_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-small"
)
BATCH_SIZE = 100

figures = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
if not figures:
    raise SystemExit(f"No figures found in {FIGURES_JSON}")

# Small collection, so rebuilding from scratch each run is cheap and avoids stale entries.
client = PersistentClient(path=str(CHROMA_DIR))
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.create_collection(
    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}  # similarity = 1 - distance
)

captions = [f["caption"] for f in figures]
embeddings = []
for start in range(0, len(captions), BATCH_SIZE):
    resp = embedding(model=EMBEDDING_MODEL, input=captions[start:start + BATCH_SIZE])
    embeddings.extend(item["embedding"] for item in resp.data)

collection.upsert(
    ids=[f["id"] for f in figures],
    documents=captions,
    embeddings=embeddings,
    metadatas=[
        {
            "paper": f["paper"],
            "page": f["page"] if f["page"] is not None else -1,  # Chroma rejects None
            "label": f["label"],
            "image_path": f["image_path"],
        }
        for f in figures
    ],
)

print(f"Done! {collection.count()} figures indexed in '{COLLECTION_NAME}'.")