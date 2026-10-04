from pathlib import Path

import chromadb
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "vectorstore" / "chroma"


# --------------------------------------------------
# Connect to Chroma
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection("rag_doctor")


# --------------------------------------------------
# Get vectors + metadata
# --------------------------------------------------

data = collection.get(
    include=[
        "embeddings",
        "documents",
        "metadatas",
    ]
)

vectors = np.array(data["embeddings"])
documents = data["documents"]
metadatas = data["metadatas"]


print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimensions: {vectors.shape[1]}")


# --------------------------------------------------
# Get document types / paper IDs
# --------------------------------------------------

doc_types = [
    metadata.get("doc_id", "unknown")
    if metadata
    else "unknown"
    for metadata in metadatas
]


# --------------------------------------------------
# Create colors for each paper
# --------------------------------------------------

unique_types = list(dict.fromkeys(doc_types))

color_map = {
    doc_type: i
    for i, doc_type in enumerate(unique_types)
}

colors = [
    color_map[doc_type]
    for doc_type in doc_types
]


# --------------------------------------------------
# Reduce embeddings to 3D using t-SNE
# --------------------------------------------------

tsne = TSNE(
    n_components=3,
    random_state=42,
    perplexity=min(30, len(vectors) - 1),
)

reduced_vectors = tsne.fit_transform(vectors)


# --------------------------------------------------
# Hover text
# --------------------------------------------------

hover_text = []

for doc_type, document, metadata in zip(
    doc_types,
    documents,
    metadatas,
):

    metadata = metadata or {}

    chunk_id = metadata.get("chunk_id", "unknown")
    section = metadata.get("section_path", "unknown")

    hover_text.append(
        f"Paper: {doc_type}"
        f"<br>Chunk: {chunk_id}"
        f"<br>Section: {section}"
        f"<br>Text: {document[:150]}..."
    )


# --------------------------------------------------
# Create 3D scatter plot
# --------------------------------------------------

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode="markers",

            marker=dict(
                size=5,
                color=colors,
                opacity=0.8,
                colorscale="Viridis",
            ),

            text=hover_text,
            hoverinfo="text",
        )
    ]
)


# --------------------------------------------------
# Layout
# --------------------------------------------------

fig.update_layout(
    title="3D Chroma Vector Store Visualization",

    scene=dict(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        zaxis_title="t-SNE 3",
    ),

    width=900,
    height=700,

    margin=dict(
        r=10,
        b=10,
        l=10,
        t=40,
    ),
)


# --------------------------------------------------
# Show
# --------------------------------------------------

fig.show()