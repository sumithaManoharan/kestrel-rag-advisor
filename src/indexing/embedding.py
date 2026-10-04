from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import embedding
from pathlib import Path
import json
import os


load_dotenv(override=True)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "vectorstore" / "chroma"
COLLECTION_NAME = "rag_doctor"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "RAG_DOCTOR_EMBEDDING_MODEL",
    "openrouter/openai/text-embedding-3-small"
)


# --------------------------------------------------
# Load Chroma
# --------------------------------------------------

client = PersistentClient(path=str(CHROMA_DIR))
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# Read chunks
# --------------------------------------------------

documents = []
metadatas = []
ids = []

print("Reading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)

        documents.append(chunk["text"])

        metadatas.append({
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "section_path": " > ".join(
                chunk.get("section_path", [])
            ),
        })

        ids.append(
            f'{chunk["doc_id"]}_{chunk["chunk_id"]}'
        )


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

BATCH_SIZE = 100

print(
    f"Generating embeddings for {len(documents)} chunks..."
)

embeddings = []

for start in range(0, len(documents), BATCH_SIZE):

    batch_documents = documents[start:start + BATCH_SIZE]

    print(
        f"Embedding chunks "
        f"{start + 1}-{min(start + BATCH_SIZE, len(documents))}"
    )

    response = embedding(
        model=EMBEDDING_MODEL,
        input=batch_documents,
    )

    batch_embeddings = [
        item["embedding"]
        for item in response.data
    ]

    embeddings.extend(batch_embeddings)


# --------------------------------------------------
# Add embeddings to Chroma
# --------------------------------------------------

print("Adding embeddings to Chroma...")

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)


print(
    f"Done! Collection contains {collection.count()} chunks."
)

