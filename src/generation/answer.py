"""
RAG Doctor chat (Gradio): a small test harness for the retrieval + answer workflow.

    python answer.py

Per message:
    1. Follow-up messages are rewritten into a standalone search query (first message is used as is).
    2. retrieve_context(query) -> 10 reranked chunks + up to 5 candidate figures (with captions and ids).
    3. The answer LLM gets the chunks and the figure list and streams its answer. It shows a figure by
       writing [[figure:<id>]] where the figure belongs - no separate image-selection model.
    4. The markers are removed from the text and the figures the model chose appear in the chat.
"""

from __future__ import annotations

import os
import re
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from litellm import completion

from src.ingestion.context import RERANKER_MODEL, retrieve_context  # adjust to where retrieve.py lives, e.g. src.retrieval.retrieve

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / os.getenv("RAG_DOCTOR_IMAGES_DIR", "data/images")  # same folder as index_figures.py; Gradio may only serve files from here (see launch)

LLM_MODEL = os.getenv("RAG_DOCTOR_MODEL", "openrouter/openai/gpt-oss-120b")
REWRITE_MODEL = RERANKER_MODEL      # small fast model, only used to rewrite follow-up questions
MAX_HISTORY_MESSAGES = 12           # last 6 exchanges are sent to the LLM
SHOW_UNREFERENCED_FIGURES = False   # True = also show retrieved figures the model did not pick (testing aid)

SYSTEM_PROMPT = """
You are RAG Doctor, an engineering advisor for Retrieval-Augmented Generation.
You are a careful, evidence-grounded RAG engineering advisor.

The user may:
- describe a RAG problem,
- ask for a recommendation,
- ask how to improve their RAG,
- compare RAG techniques,
- or ask how to implement a RAG strategy.

First determine the user's intent.

Possible intents:
1. troubleshooting
2. recommendation
3. optimization
4. comparison
5. implementation
6. conceptual

Adapt the response structure to the intent.

For troubleshooting:
- Diagnosis
- Why It Happens
- Recommended Strategies
- Implementation
- Evidence
- Sources

For recommendation:
- Recommended Strategy
- Why
- Alternatives
- Suggested Experiment
- Evidence
- Sources

For comparison:
- Option A
- Option B
- Trade-offs
- Recommendation for the stated use case
- Evidence
- Sources

Do not force irrelevant sections into the response.
If the message is just conversational (a greeting, thanks, a short clarification), reply briefly
without the structured format.

Base recommendations on the retrieved research context.
If the research does not provide enough evidence for a specific
recommendation, clearly distinguish established evidence from
your own general engineering suggestion.

This is a conversation: use the earlier messages together with the newly retrieved context.

FIGURES
The retrieved context may end with an AVAILABLE FIGURES list; each entry has an id.
If a figure would genuinely help the user understand your answer (for example an architecture or
pipeline diagram when explaining how something works or comparing approaches, or a results chart
when discussing performance), display it by writing [[figure:ID]] on its own line at the point in
the answer where it belongs, copying the id exactly.
Only show figures that directly support the answer - usually zero, one or two. Never invent ids.
If no figure helps, do not write any figure marker. Do not mention or explain this marker syntax.
"""

# [[figure:2312.10997_fig3]]
MARKER = re.compile(r"\[\[\s*figure\s*:\s*([^\]\s]+)\s*\]\]", re.I)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def condense_question(message: str, history: list[dict[str, str]]) -> str:
    """Make a follow-up like 'and how does that compare?' searchable on its own."""
    if not history:
        return message

    recent = "\n".join(f"{m['role'].upper()}: {m['content'][:600]}" for m in history[-6:])
    prompt = (
        "Rewrite the user's latest message as a standalone question that can be understood without "
        "the conversation, keeping all technical terms. If it is already self-contained, return it "
        "unchanged. Output only the question.\n\n"
        f"CONVERSATION:\n{recent}\n\nLATEST MESSAGE:\n{message}"
    )
    try:
        response = completion(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip() or message
    except Exception as exc:
        print(f"[chat] question rewrite failed ({exc}); using the raw message")
        return message


def visible_text(text: str) -> str:
    """Streaming view of the answer: hide complete markers and a half-streamed one."""
    text = MARKER.sub("", text)
    return re.sub(r"\[\[[^\]]*$", "", text)


def resolve_figures(text: str, figures_by_id: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Replace each marker with a short reference; return the clean text and the figures the model chose."""
    shown: list[dict[str, Any]] = []

    def replace(match: re.Match) -> str:
        fig = figures_by_id.get(match.group(1))
        if fig is None:  # invented or malformed id: drop it silently
            return ""
        if all(fig["id"] != s["id"] for s in shown):
            shown.append(fig)
        return f"*({fig['label'] or 'Figure'}, {fig['paper']} - shown below)*"

    return MARKER.sub(replace, text).strip(), shown


def resolve_image(image_path: str) -> Path | None:
    path = Path(image_path)
    for candidate in ([path] if path.is_absolute() else [PROJECT_ROOT / path, Path.cwd() / path]):
        if candidate.exists():
            return candidate.resolve()
    return None


def figure_messages(fig: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_image(fig["image_path"])
    if path is None:
        print(f"[chat] image file not found: {fig['image_path']}")
        return [{"role": "assistant", "content": f"*(image file not found: {fig['image_path']})*"}]

    page = f", p.{fig['page']}" if fig.get("page", -1) >= 0 else ""
    return [
        {"role": "assistant", "content": f"**{fig['label'] or 'Figure'}** ({fig['paper']}{page}): {fig['caption'][:300]}"},
        {"role": "assistant", "content": {"path": str(path)}},
    ]


# --------------------------------------------------
# Chat handler
# --------------------------------------------------
# `display` = what the Chatbot shows (text + image messages).
# `history` = plain {"role", "content": str} turns that are sent to the LLM.
# Both are kept in gr.State so this works the same across Gradio versions.

def respond(message, display, history):
    message = (message or "").strip()
    if not message:
        yield "", display, display, history
        return

    display = display + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "*Searching the papers...*"},
    ]
    yield "", display, display, history

    try:
        query = condense_question(message, history)
        print(f"[chat] retrieval query: {query}")

        result = retrieve_context(query)
        figures_by_id = {fig["id"]: fig for fig in result["figures"]}
        print(
            f"[chat] {len(result['chunks'])} chunks | "
            f"figures: {[(f['id'], f['score']) for f in result['figures']]}"
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history[-MAX_HISTORY_MESSAGES:]
            + [{
                "role": "user",
                "content": f"USER QUESTION:\n{message}\n\nRETRIEVED RESEARCH CONTEXT:\n{result['context']}",
            }]
        )

        answer = ""
        for chunk in completion(model=LLM_MODEL, messages=messages, temperature=0, stream=True):
            if not chunk.choices:
                continue
            answer += chunk.choices[0].delta.content or ""
            display[-1] = {"role": "assistant", "content": visible_text(answer) or "*Thinking...*"}
            yield "", display, display, history

        text, shown = resolve_figures(answer, figures_by_id)
        if SHOW_UNREFERENCED_FIGURES:
            shown += [fig for fig in result["figures"] if all(fig["id"] != s["id"] for s in shown)]

        text = text or "(the model returned no answer)"
        display[-1] = {"role": "assistant", "content": text}
        for fig in shown:
            display += figure_messages(fig)

    except Exception as exc:
        traceback.print_exc()
        display[-1] = {"role": "assistant", "content": f"Error: {exc}"}
        yield "", display, display, history
        return

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
    yield "", display, display, history


def clear_chat():
    return "", [], [], []


# --------------------------------------------------
# UI
# --------------------------------------------------

def make_chatbot() -> gr.Chatbot:
    try:
        return gr.Chatbot(type="messages", label="RAG Doctor", height=650)  # Gradio 5
    except TypeError:
        return gr.Chatbot(label="RAG Doctor", height=650)  # Gradio 6 removed `type`


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="RAG Doctor") as demo:
        gr.Markdown("## RAG Doctor\nAsk about RAG problems, techniques or trade-offs.")
        chatbot = make_chatbot()
        box = gr.Textbox(placeholder="Ask a question and press Enter", show_label=False)
        clear = gr.Button("Clear chat")

        display_state = gr.State([])
        history_state = gr.State([])

        outputs = [box, chatbot, display_state, history_state]
        box.submit(respond, [box, display_state, history_state], outputs)
        clear.click(clear_chat, None, outputs)

    return demo


if __name__ == "__main__":
    build_demo().launch(allowed_paths=[str(IMAGES_DIR)])