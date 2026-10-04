"""
Kestrel - a RAG advisor that hovers over the research, then dives for the answer.

Portfolio UI that reuses the retrieval + answer logic from answer.py.
Put this file next to answer.py and run:

    python app.py
"""

from __future__ import annotations

import random
import time
import traceback

import gradio as gr
from litellm import completion

from src.generation.answer import (
    IMAGES_DIR,
    LLM_MODEL,
    MAX_HISTORY_MESSAGES,
    SYSTEM_PROMPT,
    condense_question,
    figure_messages,
    resolve_figures,
    retrieve_context,
    visible_text,
)

APP_NAME = "Kestrel"
TAGLINE = "Hovers over the RAG research, then dives straight to the passage that answers your question."

EXAMPLES = [
    "My RAG returns irrelevant chunks for long, vague queries. What should I try?",
    "Compare Self-RAG and Corrective RAG for a support chatbot",
    "How do I add hybrid search with a reranker?",
    "Why does my model hallucinate even when the context is right?",
    "Is GraphRAG worth it for a pile of legal contracts?",
    "How should I chunk PDFs full of tables?",
]

STEPS = [
    ("✏️ Rewrite", "Making your question searchable"),
    ("🔎 Retrieve", "Pulling passages and figures from the papers"),
    ("🧠 Reason", "Writing the answer from the evidence"),
    ("📌 Illustrate", "Pinning up the figures"),
]

WAITING = [
    "Flipping through the papers...",
    "Squinting at footnotes...",
    "Highlighting the good parts...",
    "Cross-checking the evidence...",
]

# --------------------------------------------------
# Visual identity: ink-blue desk, highlighter yellow, sticky-note accents
# --------------------------------------------------

INK = "#13112b"
INK_2 = "#1d1a3f"
HIGHLIGHT = "#ffe14d"
CORAL = "#f4874b"  # kestrel chestnut
MINT = "#5eead4"
LAVENDER = "#cfc9ff"

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.indigo,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill=INK,
    body_text_color=LAVENDER,
    background_fill_primary=INK_2,
    background_fill_secondary=INK,
    block_background_fill=INK_2,
    block_border_color="#2c2860",
    block_label_text_color=LAVENDER,
    input_background_fill="#100e24",
    input_border_color="#3a3580",
    input_border_color_focus=HIGHLIGHT,
    button_primary_background_fill=HIGHLIGHT,
    button_primary_background_fill_hover="#ffeb7d",
    button_primary_text_color=INK,
    button_secondary_background_fill="transparent",
    button_secondary_text_color=LAVENDER,
    button_secondary_border_color="#3a3580",
)

LETTERS = "".join(f'<span style="--i:{i}">{c}</span>' for i, c in enumerate(APP_NAME))

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&display=swap');
:root{--hl:#ffe14d;--cx:#f4874b;--mint:#5eead4;--lav:#cfc9ff}

/* living background: soft glows + a slowly drifting dot grid */
.gradio-container{max-width:1100px!important;margin:auto;
  background:radial-gradient(circle at 12% 0%,#4a3cc055,transparent 45%),
             radial-gradient(circle at 92% 25%,#f4874b26,transparent 40%),
             radial-gradient(#ffffff16 1.2px,transparent 1.2px) 0 0/26px 26px,#13112b!important;
  animation:drift 6s linear infinite}
@keyframes drift{to{background-position:0 0,0 0,26px 26px}}

/* hero */
#hero{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap;padding:30px 8px 14px}
#hero h1{display:flex;margin:0;font-family:'Bricolage Grotesque',sans-serif;font-weight:800;
  font-size:clamp(3.6rem,11vw,7rem);letter-spacing:-.03em;line-height:1;color:#fff}
#hero h1 span{display:inline-block;cursor:default;transition:transform .2s;
  animation:drop .8s cubic-bezier(.3,1.6,.5,1) backwards;animation-delay:calc(var(--i)*.08s)}
#hero h1 span:nth-child(3n+1){text-shadow:5px 5px 0 var(--cx)}
#hero h1 span:nth-child(3n+2){text-shadow:5px 5px 0 var(--hl)}
#hero h1 span:nth-child(3n){text-shadow:5px 5px 0 var(--mint)}
#hero h1 span:hover{transform:translateY(-14px) rotate(-8deg)}
@keyframes drop{from{opacity:0;transform:translateY(-60px) rotate(14deg)}}
#hero p{font-size:1.1rem;margin:16px 0 0;max-width:42ch;color:var(--lav)}

/* scene: kestrel hovers, locks on, dives (9s loop) */
#scene{position:relative;width:240px;height:215px;flex:0 0 auto}
.paper{position:absolute;left:50%;bottom:0;width:170px;height:90px;margin-left:-85px;background:#fff;
  border-radius:8px;transform:rotate(-4deg);box-shadow:5px 7px 0 #0007;padding:12px;display:flex;flex-direction:column;gap:8px}
.paper b{display:block;height:6px;border-radius:3px;background:#d5d2f0}
.paper b.hit{background:var(--hl);animation:shine 1.2s ease-in-out infinite alternate}
@keyframes shine{to{box-shadow:0 0 12px 2px var(--hl)}}
.target{position:absolute;left:54px;bottom:19px;width:56px;height:56px;z-index:1}
.target i{position:absolute;inset:0;border-radius:50%;border:2px solid var(--cx);animation:ping 2.4s ease-out infinite}
.target i+i{animation-delay:1.2s}
.target::before{content:'';position:absolute;inset:-8px;border:2px dashed var(--hl);border-radius:50%;animation:spin 8s linear infinite}
@keyframes ping{from{transform:scale(.4);opacity:1}to{transform:scale(1.6);opacity:0}}
@keyframes spin{to{transform:rotate(360deg)}}
.bird{position:absolute;left:50%;top:0;width:150px;margin-left:-75px;z-index:2;
  filter:drop-shadow(0 10px 6px #0007);animation:hover 9s ease-in-out infinite}
@keyframes hover{
  0%,30%,100%{transform:translateY(0) rotate(-2deg)} 15%,45%{transform:translateY(-12px) rotate(2deg)}
  58%{transform:translateY(-6px)} 66%,72%{transform:translateY(62px) scale(.82)} 85%{transform:translateY(-4px)}}
.wl{transform-origin:58px 46px;animation:flapL .4s ease-in-out infinite alternate}
.wr{transform-origin:62px 46px;animation:flapR .4s ease-in-out infinite alternate}
@keyframes flapL{to{transform:rotate(7deg)}} @keyframes flapR{to{transform:rotate(-7deg)}}
.say{position:absolute;left:-4px;top:0;width:112px;height:32px;background:#fff;color:#000;
  font-weight:800;font-size:.8rem;border-radius:12px 12px 2px 12px;box-shadow:3px 4px 0 #0007;z-index:3}
.say::after{content:'';position:absolute;right:-6px;bottom:7px;border:6px solid transparent;
  border-left-color:#fff;border-right:0}
.say span{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;animation:say 9s infinite}
#hero .say span{color:#000!important}
.say span:nth-child(2){animation-delay:-6s} .say span:nth-child(3){animation-delay:-3s}
@keyframes say{0%,30%{opacity:1}33%,100%{opacity:0}}

/* sticky notes: 3-column grid, pinned, drop in once, wiggle on hover */
#notes{display:grid!important;grid-template-columns:repeat(3,1fr);gap:20px 16px;padding:10px 4px}
@media(max-width:700px){#notes{grid-template-columns:1fr 1fr}}
.sticky{position:relative;color:#13112b!important;border:none!important;text-align:left!important;
  font-weight:700!important;line-height:1.3!important;min-height:96px;white-space:normal!important;
  border-radius:4px 4px 16px 4px!important;box-shadow:4px 6px 0 #0007!important;
  animation:slap .55s cubic-bezier(.3,1.5,.5,1) backwards;transition:transform .18s,box-shadow .18s!important}
.sticky::before{content:'';position:absolute;top:-8px;left:50%;width:14px;height:14px;border-radius:50%;
  background:radial-gradient(circle at 35% 30%,#fffa,#e23 55%,#811);box-shadow:0 3px 0 #0006}
.sticky:hover{transform:rotate(0) translateY(-7px) scale(1.04)!important;box-shadow:7px 13px 0 #0007!important}
@keyframes slap{from{opacity:0;transform:translateY(-50px) rotate(14deg)}}
.s1{background:#ffe14d!important;transform:rotate(-1.6deg);animation-delay:.9s}
.s2{background:#5eead4!important;transform:rotate(1.2deg);animation-delay:1s}
.s3{background:#f4874b!important;transform:rotate(-.8deg);animation-delay:1.1s}
.s4{background:#b9a8ff!important;transform:rotate(1.5deg);animation-delay:1.2s}
.s5{background:#ffb86b!important;transform:rotate(-1.2deg);animation-delay:1.3s}
.s6{background:#9ae6ff!important;transform:rotate(.9deg);animation-delay:1.4s}

/* pipeline cards */
.steps{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.step{position:relative;overflow:hidden;flex:1 1 190px;padding:12px 14px;border-radius:12px;
  background:#1d1a3f;border:2px solid #2c2860;color:#8d88c9;transition:all .3s}
.step b{display:block;font-family:'Bricolage Grotesque',sans-serif;color:#b9b4f0}
.step small{color:#8d88c9}
.step.active{background:#2a2560;border-color:var(--hl);transform:translateY(-4px);box-shadow:0 8px 0 #0005}
.step.active b,.step.active small{color:#fff}
.step.active::after{content:'';position:absolute;left:0;bottom:0;height:4px;width:40%;background:var(--hl);
  border-radius:2px;animation:slide 1s ease-in-out infinite alternate}
@keyframes slide{to{left:60%}}
.step.done{border-color:var(--mint);background:#16303b}
.step.done b::after{content:' \2713';color:var(--mint)}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.stat{font-family:'JetBrains Mono',monospace;font-size:.8rem;padding:4px 10px;border-radius:999px;
  background:#5eead422;color:var(--mint);animation:pop .45s cubic-bezier(.3,1.7,.5,1) backwards}
.stat.hot{background:#ffe14d22;color:var(--hl)} .stat.pink{background:#f4874b22;color:var(--cx)}
@keyframes pop{from{opacity:0;transform:scale(.4)}}

/* chat + buttons */
#kchat{border:2px solid #3a3580!important;border-radius:18px!important;box-shadow:0 0 0 5px #ffe14d14,0 18px 40px #0007}
.gradio-container button.primary{font-weight:800;box-shadow:3px 4px 0 #b8941a;transition:transform .15s}
.gradio-container button.primary:hover{transform:translate(-1px,-2px) rotate(-1deg)}
#footer{text-align:center;font-size:.85rem;opacity:.7;padding:18px 0 6px}
#footer code{color:var(--hl)}

@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

HERO = f"""
<div id="hero">
  <div>
    <h1>{LETTERS}</h1>
    <p>{TAGLINE}</p>
  </div>
  <div id="scene" aria-hidden="true">
    <div class="paper"><b style="width:90%"></b><b style="width:70%"></b><b class="hit" style="width:62%"></b><b style="width:80%"></b></div>
    <div class="target"><i></i><i></i></div>
    <svg class="bird" viewBox="0 0 120 100" xmlns="http://www.w3.org/2000/svg">
      <g class="wl"><path d="M58 40C40 30 18 34 3 52c19-3 34 0 51 10z" fill="#f4874b"/>
        <path d="M20 44l-6 10M32 41l-5 14M44 42l-3 15" stroke="#3a2a4f" stroke-width="2.5" stroke-linecap="round"/></g>
      <g class="wr"><path d="M62 40C80 30 102 34 117 52c-19-3-34 0-51 10z" fill="#f4874b"/>
        <path d="M100 44l6 10M88 41l5 14M76 42l3 15" stroke="#3a2a4f" stroke-width="2.5" stroke-linecap="round"/></g>
      <path d="M50 70h20l7 26H43z" fill="#f4874b"/><path d="M45 88h30l2 8H43z" fill="#3a2a4f"/>
      <ellipse cx="60" cy="54" rx="10" ry="22" fill="#e8743a"/>
      <circle cx="60" cy="30" r="9" fill="#9fb3d9"/><path d="M56 24l4-9 4 9z" fill="#ffe14d"/>
      <circle cx="56" cy="29" r="1.8" fill="#13112b"/><circle cx="64" cy="29" r="1.8" fill="#13112b"/>
    </svg>
    <div class="say"><span>Hovering...</span><span>Target locked!</span><span>Diving in!</span></div>
  </div>
</div>
"""

FOOTER = """
<div id="footer">
  Built with <code>Gradio</code> + <code>LiteLLM</code> + hybrid retrieval, reranking and figure search.
  Answers cite the papers they came from.
</div>
"""


def status_html(active: int, stats: dict | None = None, note: str = "") -> str:
    """active: index of the running step, len(STEPS) = all done, -1 = idle."""
    cells = []
    for i, (title, desc) in enumerate(STEPS):
        cls = "done" if i < active else "active" if i == active else ""
        text = note if (i == active and note) else desc
        cells.append(f'<div class="step {cls}"><b>{title}</b><small>{text}</small></div>')
    html = f'<div class="steps">{"".join(cells)}</div>'
    if stats:
        html += '<div class="stats">' + "".join(
            f'<span class="stat {kind}">{label}</span>' for label, kind in stats.items()
        ) + "</div>"
    return html


# --------------------------------------------------
# Chat handler (same flow as answer.py, plus live status)
# --------------------------------------------------

def respond(message, display, history):
    message = (message or "").strip()
    if not message:
        yield "", display, display, history, status_html(-1)
        return

    t0 = time.time()
    display = display + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": f"*{random.choice(WAITING)}*"},
    ]
    yield "", display, display, history, status_html(0)

    try:
        query = condense_question(message, history)
        yield "", display, display, history, status_html(1)

        result = retrieve_context(query)
        figures_by_id = {fig["id"]: fig for fig in result["figures"]}
        stats = {
            f"{len(result['chunks'])} passages": "",
            f"{len(result['figures'])} candidate figures": "hot",
        }
        yield "", display, display, history, status_html(2, stats)

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
            yield "", display, display, history, status_html(2, stats)

        yield "", display, display, history, status_html(3, stats)

        text, shown = resolve_figures(answer, figures_by_id)
        text = text or "(the model returned no answer)"
        display[-1] = {"role": "assistant", "content": text}
        for fig in shown:
            display += figure_messages(fig)

        stats[f"{len(shown)} figures pinned"] = "pink"
        stats[f"{time.time() - t0:.1f}s"] = ""

    except Exception as exc:
        traceback.print_exc()
        display[-1] = {
            "role": "assistant",
            "content": f"Something broke: `{exc}`. Check the terminal log, then try again.",
        }
        yield "", display, display, history, status_html(-1)
        return

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
    yield "", display, display, history, status_html(len(STEPS), stats)


def clear_chat():
    return "", [], [], [], status_html(-1)


def make_chatbot() -> gr.Chatbot:
    for kwargs in (
        {"type": "messages", "allow_tags": False},  # Gradio 5.x (recent)
        {"type": "messages"},                       # older Gradio 5.x
        {"allow_tags": False},                      # Gradio 6 (no `type`)
        {},
    ):
        try:
            return gr.Chatbot(label=APP_NAME, height=620, elem_id="kchat", **kwargs)
        except TypeError:
            continue


# --------------------------------------------------
# UI
# --------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title=f"{APP_NAME} | RAG advisor", theme=theme, css=CSS) as demo:
        gr.HTML(HERO)

        gr.Markdown("**Try a sticky note** or write your own question. 🎲 picks one for you.")
        with gr.Row(elem_id="notes"):
            notes = [
                gr.Button(q, elem_classes=["sticky", f"s{i + 1}"])
                for i, q in enumerate(EXAMPLES)
            ]

        status = gr.HTML(status_html(-1))
        chatbot = make_chatbot()

        with gr.Row():
            box = gr.Textbox(
                placeholder="Describe your RAG problem, then press Enter",
                show_label=False, scale=8, container=False,
            )
            send = gr.Button("Ask", variant="primary", scale=1)
            surprise = gr.Button("🎲", scale=0, min_width=56)
            clear = gr.Button("Clear", scale=0, min_width=80)

        gr.HTML(FOOTER)

        display_state = gr.State([])
        history_state = gr.State([])
        outputs = [box, chatbot, display_state, history_state, status]

        box.submit(respond, [box, display_state, history_state], outputs)
        send.click(respond, [box, display_state, history_state], outputs)
        for btn, q in zip(notes, EXAMPLES):
            btn.click(lambda q=q: q, None, box).then(
                respond, [box, display_state, history_state], outputs
            )
        surprise.click(lambda: random.choice(EXAMPLES), None, box).then(
            respond, [box, display_state, history_state], outputs
        )
        clear.click(clear_chat, None, outputs)

    return demo


if __name__ == "__main__":
    build_demo().launch(allowed_paths=[str(IMAGES_DIR)],inbrowser=True)