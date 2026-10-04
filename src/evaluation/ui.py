"""
Gradio dashboard for RAG Doctor evaluation.

Run:
    uv run python src/evaluation/evaluator.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr

from eval import RESULTS_FILE, run_evaluation


def load_results():
    if not RESULTS_FILE.exists():
        return None

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_card(title: str, value, description: str, value_color: str = "#2563eb") -> str:
    value_text = "—" if value is None else str(value)

    return f"""
    <div style="
        border:1px solid #e5e7eb;
        border-radius:14px;
        padding:18px;
        min-width:170px;
        flex:1;
        background:white;
        box-shadow:0 2px 8px rgba(0,0,0,0.06);
    ">
        <div style="
            font-size:13px;
            color:#6b7280;
            font-weight:600;
            margin-bottom:6px;
        ">
            {html.escape(title)}
        </div>

        <div style="
            font-size:30px;
            font-weight:800;
            margin:4px 0 8px;
            color:{value_color};
        ">
            {html.escape(value_text)}
        </div>

        <div style="
            font-size:12px;
            line-height:1.4;
            color:#9ca3af;
        ">
            {html.escape(description)}
        </div>
    </div>
    """


def retrieval_html(summary: dict) -> str:
    retrieval = summary["retrieval"]

    return f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;">

        {metric_card(
            "Precision@5",
            retrieval["mean_precision_at_5"],
            "Share of the top 5 judged chunks that are relevant.",
            "#2563eb"
        )}

        {metric_card(
            "Recall@10",
            retrieval["mean_recall_at_10"],
            "Relevant ground-truth chunks found in top 10.",
            "#059669"
        )}

        {metric_card(
            "MRR",
            retrieval["mean_mrr"],
            "How early the first highly relevant chunk appears.",
            "#d97706"
        )}

        {metric_card(
            "nDCG@5",
            retrieval["mean_ndcg_at_5"],
            "Rewards highly relevant chunks appearing near the top.",
            "#7c3aed"
        )}

    </div>
    """


def answer_html(summary: dict) -> str:
    answer = summary["answer"]

    labels = {
        "problem_understanding": ("Problem understanding", "#2563eb"),
        "faithfulness": ("Faithfulness", "#059669"),
        "relevance": ("Relevance", "#0891b2"),
        "completeness": ("Completeness", "#7c3aed"),
        "recommendation_quality": ("Recommendation quality", "#d97706"),
        "experiment_quality": ("Experiment quality", "#db2777"),
        "citation_support": ("Citation support", "#475569"),
    }

    cards = []

    for key, (label, color) in labels.items():
        cards.append(
            metric_card(
                label,
                f"{answer.get(key, 0):.2f} / 5",
                "LLM-as-judge mean score.",
                color,
            )
        )

    return f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;">
        {''.join(cards)}
    </div>
    """


def chunks_markdown(case_id: str, results: list[dict]) -> str:
    case = next(
        (item for item in results if item["id"] == case_id),
        None,
    )

    if case is None:
        return "Select a test case."

    if "error" in case:
        return f"### Error\n\n`{case['error']}`"

    output = [
        f"## {case['id']}",
        "",
        f"**Problem:** {case['question']}",
        "",
        "### Generated answer",
        "",
        case["answer"],
        "",
        "### Retrieved chunks",
        "",
    ]

    for chunk in case["retrieved_chunks"]:
        metadata = chunk.get("metadata", {})

        doc_id = metadata.get("doc_id", "unknown")
        section = metadata.get("section_path", "unknown")

        output.extend(
            [
                f"#### Rank {chunk['rank']} — relevance {chunk['relevance']}/3",
                f"**Paper:** `{doc_id}`  ",
                f"**Section:** `{section}`  ",
                f"**Chunk ID:** `{chunk['id']}`  ",
                f"**Judge:** {chunk['relevance_reason']}",
                "",
                "> " + chunk["document"][:1800].replace("\n", "\n> "),
                "",
            ]
        )

    return "\n".join(output)


def refresh_dashboard():
    data = load_results()

    if not data:
        return (
            "<p>No evaluation results yet. Click <b>Run evaluation</b>.</p>",
            "<p>No answer evaluation yet.</p>",
            gr.update(choices=[], value=None),
            "No retrieved chunks yet.",
        )

    summary = data["summary"]
    results = data["results"]

    case_ids = [item["id"] for item in results]

    first_case = case_ids[0] if case_ids else None

    return (
        retrieval_html(summary),
        answer_html(summary),
        gr.update(choices=case_ids, value=first_case),
        chunks_markdown(first_case, results) if first_case else "No results.",
    )


def run_and_refresh():
    run_evaluation()
    return refresh_dashboard()


def show_case(case_id):
    data = load_results()

    if not data:
        return "No evaluation results yet."

    return chunks_markdown(case_id, data["results"])


with gr.Blocks(
    title="RAG Doctor Evaluator",
) as demo:

    gr.Markdown(
        """
        # 🩺 RAG Doctor
        ### Evaluation Dashboard

        Evaluate your RAG system from three perspectives:

        **Retrieval quality → Answer quality → Evidence inspection**
        """
        )

    run_button = gr.Button(
        "Run evaluation",
        variant="primary",
    )

    gr.Markdown("## 1. Retrieval Evaluation")

    gr.Markdown(
        """
        **Precision@K** — how much of the retrieved context is relevant.  
        **Recall@K** — how much of the known relevant evidence was retrieved.  
        **MRR** — how early the first highly relevant chunk appears.  
        **nDCG@K** — whether highly relevant chunks are ranked near the top.
        """
    )

    retrieval_output = gr.HTML()

    gr.Markdown("## 2. LLM-as-Judge — Answer Evaluation")

    gr.Markdown(
        """
        The LLM judge scores whether the generated answer understands the
        problem, stays faithful to retrieved evidence, gives useful strategies,
        and proposes a useful experiment.
        """
    )

    answer_output = gr.HTML()

    gr.Markdown("## 3. Retrieved Chunks")

    gr.Markdown(
        "Select a test case to inspect the actual chunks used to generate the answer."
    )

    case_selector = gr.Dropdown(
        choices=[],
        label="Test case",
    )

    chunks_output = gr.Markdown()

    run_button.click(
        fn=run_and_refresh,
        outputs=[
            retrieval_output,
            answer_output,
            case_selector,
            chunks_output,
        ],
    )

    case_selector.change(
        fn=show_case,
        inputs=case_selector,
        outputs=chunks_output,
    )

    demo.load(
        fn=refresh_dashboard,
        outputs=[
            retrieval_output,
            answer_output,
            case_selector,
            chunks_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
