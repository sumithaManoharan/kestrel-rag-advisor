# Kestrel 🦅

**A research-grounded advisor for Retrieval-Augmented Generation.**
Describe a RAG problem, compare techniques, or ask how to implement a strategy. Kestrel hovers over a library of RAG papers, then dives to the passages (and figures) that answer you, with sources.

[**Try the live demo**](https://huggingface.co/spaces/sumi0702/Kestrel) · [pngs/Screencast from 2026-10-05 02-45-54.webm](#demo)

![Kestrel screenshot](pngs/image.png)

## Demo

<!-- Drag and drop your video file into this README in GitHub's web editor.
     GitHub uploads it and inserts a link like the one below on its own line. Replace this line with it. -->
https://github.com/user-attachments/assets/TODO-your-video-id

## What it does

- **Diagnoses RAG problems.** "My RAG returns irrelevant chunks for vague queries" gets a diagnosis, why it happens, and strategies to try.
- **Adapts to your intent.** Troubleshooting, recommendations, optimization, comparisons, implementation and conceptual questions each get a different answer structure.
- **Stays grounded.** Answers are based on retrieved paper excerpts, and the model separates established evidence from its own engineering suggestions.
- **Shows the figures.** When an architecture diagram or results chart helps, the answer pins the original figure from the paper right in the chat.
- **Handles follow-ups.** Short follow-ups like "how does that compare?" are rewritten into standalone search queries.

## How it works

```mermaid
flowchart LR
    Q[Your question] --> R[Rewrite follow-ups<br/>into a standalone query<br/>DeepSeek]
    R --> S[Retrieve passages, then<br/>rerank to the top 10<br/>DeepSeek]
    R --> F[Find candidate figures<br/>up to 5, from a figure index]
    S --> A[Answer model gpt-oss<br/>streams the answer and<br/>chooses which figures to show]
    F --> A
    A --> M[Figure markers<br/>resolved to images]
    M --> U[Answer + pinned figures<br/>in the chat]
```

1. **Rewrite.** For follow-up messages, a small, fast model (DeepSeek, the same one used for reranking) turns the latest message into a standalone query. The first message is used as is.
2. **Retrieve and rerank.** `retrieve_context(query)` finds candidate passages, and DeepSeek reranks them down to the 10 best.
3. **Find candidate figures.** The same retrieval step also returns up to 5 candidate figures from a separate figure index, each with a caption and an id.
4. **Answer.** The answer model gets the passages and the figure list, and streams its response.
5. **Pin figures.** The answer model shows a figure by writing `[[figure:<id>]]` where it belongs. The app strips the markers, drops any invented ids, and renders the chosen figures in the chat.

## Design decisions

- **The answer model picks the figures.** There is no separate image-selection model. Retrieval supplies a few candidate figures, and the same model that writes the answer decides which ones help (usually zero, one or two) and where they belong, so images appear right next to the text they support.
- **Unknown ids are dropped silently.** If the model invents a figure id, it is ignored rather than breaking the answer.
- **Streaming with live status.** The UI shows each pipeline stage (Rewrite, Retrieve, Reason, Illustrate) as it runs, plus passage count, figure count and response time.
- **Evidence over confidence.** The system prompt tells the model to mark which parts come from the research and which are its own suggestions.

## Tech stack

Python · Gradio · LiteLLM · OpenRouter · ChromaDB · [uv](https://docs.astral.sh/uv/)

| Role | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Query rewriting | DeepSeek (same fast model as the reranker) |
| Reranking | DeepSeek (`deepseek-v4.1-flash` by default) |
| Answer generation and figure choice | `gpt-oss-120b` by default |

Models are called through LiteLLM, and the answer model can be swapped by changing one environment variable.

## Getting started

```bash
git clone https://github.com/TODO-your-username/TODO-repo-name.git
cd TODO-repo-name
uv sync

cp .env.example .env     # then add your API key(s)
```

Build the index (first time only):

```bash
# TODO: replace with your ingestion command(s), for example:
# uv run python -m src.ingestion.<your_ingestion_script>
```

Run the app:

```bash
uv run python app.py
```

Then open the local URL that Gradio prints (usually http://127.0.0.1:7860).

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | API key for the answer model (via OpenRouter) | none |
| `RAG_DOCTOR_MODEL` | Answer model, in LiteLLM format | `openrouter/openai/gpt-oss-120b` |
| `RAG_DOCTOR_IMAGES_DIR` | Folder holding the extracted figure images | `data/images` |

## Project structure

<!-- TODO: adjust to match your repo -->
```
.
├── app.py                # Kestrel UI (Gradio)
├── answer.py             # Retrieval + answer workflow and prompt
├── src/
│   └── ingestion/        # Paper processing, indexing, retrieve_context
├── data/
│   └── images/           # Extracted figures
├── docs/                 # Screenshot and demo assets
├── .env.example
└── README.md
```

## Results

<!-- TODO (optional but strong): add a few numbers, such as retrieval hit rate on a small question set,
     or a before/after comparison with and without reranking. -->

## Limitations and next steps

- Answers are only as good as the papers in the index. TODO: number of papers and date range.
- No automated evaluation yet. TODO: add a small question set with expected sources.
- Possible next steps: more papers, a citation panel showing the exact passages used, and user feedback buttons.

## Credits

The knowledge base is built from published research papers on RAG. All papers, text excerpts and figures belong to their original authors and are shown here for educational and portfolio purposes. TODO: list the main papers.

## License

MIT. See [LICENSE](LICENSE).
