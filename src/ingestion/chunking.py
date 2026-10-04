#!/usr/bin/env python3
"""
chunker.py - structure-aware chunking for Docling markdown exports of research papers.

Pipeline:  clean -> parse blocks -> rebuild heading hierarchy -> pack per section

Usage:
    python chunker.py paper.md --out chunks.jsonl
    python chunker.py paper.md --target 300 --overlap 0      # try other settings

In code:
    from chunker import chunk_markdown, ChunkConfig
    doc = chunk_markdown(md_text, doc_id="2407.12883", cfg=ChunkConfig(target_tokens=300))
    doc.chunks     # embed `embed_text`, show/feed `text` to the LLM
    doc.sections   # {section_id: {"path": [...], "text": "..."}}  -> for small-to-big retrieval
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics as st
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


# --------------------------------------------------------------------------- config
def approx_tokens(s: str) -> int:
    """Cheap default (~4 chars/token). Swap in your embedding model's tokenizer:

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        cfg = ChunkConfig(count=lambda s: len(tok.encode(s, add_special_tokens=False)))
    """
    return max(1, len(s) // 4)


@dataclass
class ChunkConfig:
    target_tokens: int = 400       # pack prose up to roughly this size
    max_tokens: int = 512          # hard ceiling for any single chunk
    min_tokens: int = 60           # tiny trailing pieces get merged into the previous chunk
    overlap_tokens: int = 50       # sentence carry-over when a section is split by size (0 = off)
    inline_tokens: int = 150       # tables/code smaller than this stay inside the surrounding prose chunk
    count: Callable[[str], int] = approx_tokens
    drop_sections: tuple = ("references", "bibliography", "contents", "table of contents",
                            "acknowledgments", "acknowledgements")
    drop_footnotes: bool = True    # page footnotes that Docling injects mid-flow
    prepend_header: bool = True    # embed_text = "<title> | <section path>\n\n<text>"


# --------------------------------------------------------------------------- data
@dataclass
class Block:
    kind: str                      # heading | para | list | table | code
    text: str
    caption: str = ""              # tables only


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    kind: str                      # text | table | code
    section_id: str
    section_path: list[str]
    text: str
    embed_text: str
    n_tokens: int


@dataclass
class ChunkedDoc:
    doc_id: str
    title: str
    chunks: list[Chunk] = field(default_factory=list)
    sections: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- 1. clean + parse
_NOISE = re.compile(r"<!--\s*(?:image|formula-not-decoded)\s*-->")
_TABLE_CAPTION = re.compile(r"^Table\s+\d+\s*[:.]")
_FOOTNOTE = re.compile(r"^(?:\d{1,2}|[\u2217\u2020\u2021*])\s+[A-Z]")


def clean(md: str) -> str:
    md = html.unescape(md)                       # &amp; &lt; &gt; -> & < >
    md = _NOISE.sub("", md)                      # placeholders carry no information
    return re.sub(r"\n{3,}", "\n\n", md)


def parse_blocks(md: str) -> list[Block]:
    lines, i, blocks, para = md.split("\n"), 0, [], []

    def flush_para():
        if para:
            blocks.append(Block("para", " ".join(l.strip() for l in para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):                              # fenced code
            flush_para()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            blocks.append(Block("code", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif re.match(r"^#{1,6}\s+", line):                     # heading
            flush_para()
            blocks.append(Block("heading", re.sub(r"^#{1,6}\s+", "", line).strip()))
            i += 1
        elif line.lstrip().startswith("|"):                     # table
            flush_para()
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(Block("table", "\n".join(lines[i:j])))
            i = j
        elif re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):          # list
            flush_para()
            j = i
            while j < len(lines) and re.match(r"^\s*(?:[-*]|\d+\.)\s+", lines[j]):
                j += 1
            blocks.append(Block("list", "\n".join(lines[i:j])))
            i = j
        elif not line.strip():
            flush_para()
            i += 1
        else:
            para.append(line)
            i += 1
    flush_para()

    # attach "Table N: ..." captions (Docling puts them right before the table)
    merged = []
    for b in blocks:
        if b.kind == "table" and merged and merged[-1].kind == "para" \
                and _TABLE_CAPTION.match(merged[-1].text):
            b.caption = merged.pop().text
        merged.append(b)
    return merged


# --------------------------------------------------------------------------- 2. heading hierarchy
# Docling exports every heading as "##", so depth is rebuilt from section numbering.
_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")             # "3.2 Title"
_APP = re.compile(r"^([A-Z](?:\.\d+)*)\s+(?=[A-Z])")       # "A.1 Title" (appendix)
_KNOWN_TOP = {"abstract", "introduction", "related work", "conclusion", "conclusions",
              "discussion", "references", "bibliography", "contents", "appendix",
              "acknowledgments", "acknowledgements", "limitations", "reproducibility",
              "ethics statement", "code of ethics and ethics statement"}


def heading_depth(text: str) -> int | None:
    """Depth from numbering; 1 for known top-level names; None if unrecognised."""
    m = _NUM.match(text) or _APP.match(text)
    if m:
        return m.group(1).count(".") + 1
    return 1 if text.strip().lower() in _KNOWN_TOP else None


def doc_is_numbered(blocks: list[Block]) -> bool:
    heads = [b.text for b in blocks if b.kind == "heading"]
    n = sum(1 for h in heads if _NUM.match(h) or _APP.match(h))
    return n >= 3 and n / max(len(heads), 1) >= 0.25


# --------------------------------------------------------------------------- 3. splitting helpers
_ABBR = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bFig|\bTab|\bSec|\bEq|\bvs|\bcf|\bNo)\.$")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    out: list[str] = []
    for p in parts:                                  # undo splits after "et al." / "Fig." ...
        if out and _ABBR.search(out[-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def tail_sentences(text: str, cfg: ChunkConfig) -> str:
    """Last few sentences of `text` (<= overlap_tokens) to seed the next chunk."""
    out, n = [], 0
    for s in reversed(split_sentences(text)):
        t = cfg.count(s)
        if n + t > cfg.overlap_tokens:
            break
        out.insert(0, s)
        n += t
    return " ".join(out)


def split_long_para(text: str, cfg: ChunkConfig) -> list[str]:
    if cfg.count(text) <= cfg.max_tokens:
        return [text]
    pieces, cur = [], []
    for s in split_sentences(text):
        if cur and cfg.count(" ".join(cur + [s])) > cfg.target_tokens:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_lines(lines: list[str], cfg: ChunkConfig, prefix: list[str] | None = None) -> list[list[str]]:
    """Group lines into pieces <= max_tokens; `prefix` (caption/header rows) repeats in each."""
    prefix = prefix or []
    flat: list[str] = []
    for ln in lines:                                 # a single huge line (one-line prompt) -> sentence split
        flat.extend(split_long_para(ln, cfg) if cfg.count(ln) > cfg.max_tokens else [ln])
    pieces, cur = [], []
    for ln in flat:
        if cur and cfg.count("\n".join(prefix + cur + [ln])) > cfg.max_tokens:
            pieces.append(prefix + cur)
            cur = []
        cur.append(ln)
    if cur:
        pieces.append(prefix + cur)
    return pieces


# --------------------------------------------------------------------------- 4. tables
def normalize_table(b: Block) -> tuple[list[str], list[str]]:
    """Strip Docling's whitespace padding + repeated merged cells. Returns (header_lines, body_rows)."""
    rows = []
    for raw in b.text.splitlines():
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) and any(cells):
            continue                                            # separator row
        if len(cells) > 2 and len(set(cells)) == 1 and not re.fullmatch(r"[\d.,%-]+", cells[0]):
            cells = cells[:1]                                   # spanned cell repeated in every column
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return [], []
    ncol = rows[0].count("|") - 1
    header = ([b.caption] if b.caption else []) + [rows[0], "|" + "---|" * max(ncol, 1)]
    return header, rows[1:]


# --------------------------------------------------------------------------- 5. main pipeline
def chunk_markdown(md: str, doc_id: str, cfg: ChunkConfig | None = None) -> ChunkedDoc:
    cfg = cfg or ChunkConfig()
    blocks = parse_blocks(clean(md))
    numbered = doc_is_numbered(blocks)

    title = next((b.text for b in blocks if b.kind == "heading"), doc_id)
    doc = ChunkedDoc(doc_id=doc_id, title=title)

    # --- walk blocks, grouping them into sections keyed by their heading path
    sections: list[dict] = []          # {"path": [...], "blocks": [...]}
    stack: list[str] = []
    last_numbered_depth = 0
    cur = {"path": [], "blocks": []}
    seen_title = False

    for b in blocks:
        if b.kind == "heading":
            if not seen_title:                       # first heading = paper title
                seen_title = True
                continue
            depth = heading_depth(b.text)
            if numbered and depth is None:           # false heading ("## Query", "## DEBUG") -> inline label
                cur["blocks"].append(Block("para", f"**{b.text}**"))
                continue
            if depth is None:                        # un-numbered paper: every heading is top level
                depth = 1
            if _NUM.match(b.text) or _APP.match(b.text):
                last_numbered_depth = depth
            stack = stack[:depth - 1] + [b.text]
            cur = {"path": list(stack), "blocks": []}
            sections.append(cur)
            continue
        if b.kind == "para" and cfg.drop_footnotes and _FOOTNOTE.match(b.text) and cfg.count(b.text) < 60:
            continue
        if not sections or cur is not sections[-1]:  # front matter (before first real section)
            if cur not in sections:
                sections.append(cur)
        cur["blocks"].append(b)

    # --- pack each section into chunks
    n_chunk = 0
    for s_idx, sec in enumerate(sections):
        path = sec["path"]
        root = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", path[0]).strip().lower() if path else ""
        if root in cfg.drop_sections or not sec["blocks"]:
            continue
        section_id = f"s{s_idx:03d}"
        header = f"{title} | {' > '.join(path) if path else 'front matter'}"
        pieces = pack_section(sec["blocks"], cfg)
        pieces = merge_small(pieces, cfg)
        doc.sections[section_id] = {"path": path, "text": "\n\n".join(
            render_atomic(b) if b.kind in ("table", "code") else b.text for b in sec["blocks"])}
        for kind, text in pieces:
            doc.chunks.append(Chunk(
                chunk_id=f"{doc_id}::{n_chunk:04d}", doc_id=doc_id, kind=kind,
                section_id=section_id, section_path=path, text=text,
                embed_text=f"{header}\n\n{text}" if cfg.prepend_header else text,
                n_tokens=cfg.count(text)))
            n_chunk += 1
    return doc


def render_atomic(b: Block) -> str:
    if b.kind == "table":
        header, body = normalize_table(b)
        return "\n".join(header + body)
    return "```\n" + b.text + "\n```"


def pack_section(blocks: list[Block], cfg: ChunkConfig) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    seed = ""

    def joined(extra: list[str] = []) -> str:
        return "\n\n".join(([seed] if seed else []) + buf + extra)

    def flush(carry: bool):
        nonlocal buf, seed
        if not buf:
            return
        text = joined()
        out.append(("text", text))
        seed = tail_sentences(text, cfg) if carry and cfg.overlap_tokens else ""
        buf = []

    for b in blocks:
        if b.kind in ("table", "code"):
            full = render_atomic(b)
            if cfg.count(full) > cfg.inline_tokens:  # big table/code: its own chunk(s), never merged with prose
                flush(carry=False)
                seed = ""
                if b.kind == "table":
                    header, body = normalize_table(b)
                    out += [("table", "\n".join(p)) for p in split_lines(body, cfg, prefix=header)]
                else:
                    out += [("code", "```\n" + "\n".join(p) + "\n```")
                            for p in split_lines(b.text.split("\n"), cfg)]
                continue
            pieces = [full]                          # small table/code: keep with the surrounding prose
        elif b.kind == "para":
            pieces = split_long_para(b.text, cfg)
        else:
            pieces = ["\n".join(p) for p in split_lines(b.text.split("\n"), cfg)]
        for piece in pieces:
            if buf and cfg.count(joined([piece])) > cfg.target_tokens:
                flush(carry=True)
            buf.append(piece)
    flush(carry=False)
    return out


def merge_small(pieces: list[tuple[str, str]], cfg: ChunkConfig) -> list[tuple[str, str]]:
    """Fold tiny prose leftovers into the previous prose chunk when that stays under max_tokens."""
    out: list[tuple[str, str]] = []
    for kind, text in pieces:
        if (out and kind == "text" and out[-1][0] == "text" and cfg.count(text) < cfg.min_tokens
                and cfg.count(out[-1][1] + "\n\n" + text) <= cfg.max_tokens):
            out[-1] = ("text", out[-1][1] + "\n\n" + text)
        else:
            out.append((kind, text))
    return out


# --------------------------------------------------------------------------- CLI
def report(doc: ChunkedDoc) -> None:
    toks = [c.n_tokens for c in doc.chunks]
    kinds = {k: sum(c.kind == k for c in doc.chunks) for k in ("text", "table", "code")}
    q = sorted(toks)
    print(f"{doc.doc_id}: {len(toks)} chunks {kinds} | sections kept: {len(doc.sections)}")
    print(f"tokens  min {q[0]}  p10 {q[len(q)//10]}  median {st.median(q):.0f}  "
          f"p90 {q[int(len(q)*.9)]}  max {q[-1]}")
    print(f"chunks < 60 tok: {sum(t < 60 for t in toks)}   > 512 tok: {sum(t > 512 for t in toks)}")


def main(): 
    markdown_folder = Path("data/processed/markdown")
    output_file = Path("data/processed/all_chunks.jsonl") 
    cfg = ChunkConfig( target_tokens=400, max_tokens=512, overlap_tokens=50 ) 

    with open(output_file, "w", encoding="utf-8") as f: 
        for md_file in markdown_folder.glob("*.md"):
             print(f"\nProcessing: {md_file.name}")
             doc = chunk_markdown( md_file.read_text(encoding="utf-8"), doc_id=md_file.stem, cfg=cfg ) 
             report(doc) 
             
             for chunk in doc.chunks: 
                f.write( json.dumps(asdict(chunk), ensure_ascii=False) + "\n" ) 
    print(f"\nAll chunks saved to: {output_file}") 
                
if __name__ == "__main__": 
    main()