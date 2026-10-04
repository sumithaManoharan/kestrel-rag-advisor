"""
Ingestion step 1: arXiv PDFs -> markdown + figure images + figures.json.

    python -m src.ingestion.parse_docs --download      # first run: fetch missing PDFs, then parse
    python -m src.ingestion.parse_docs                 # parse whatever is in data/raw
    python -m src.ingestion.parse_docs --only 2312.10997 2310.11511
    python -m src.ingestion.parse_docs --force         # ignore the cache, reconvert every PDF

Each PDF goes through Docling ONCE (the slow step). The converted document is cached as JSON, so
re-running after you change the caption logic, size filter or naming only re-exports (seconds).

Reads / writes (all relative to the project root):
    data/raw/<id>.pdf                      input PDFs
    data/cache/docling/<id>.json           cached Docling document (pixels embedded)
    data/processed/markdown/<id>.md        -> chunking.py
    data/images/<id>_fig_<N>.png           figures, named by the figure number in the caption
    data/images/figures.json               -> index_figures.py   (id, paper, page, label, caption, image_path)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode, TextItem

load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "docling"
MD_DIR = PROJECT_ROOT / "data" / "processed" / "markdown"
IMAGE_DIR = PROJECT_ROOT / os.getenv("RAG_DOCTOR_IMAGES_DIR", "data/images")  # same variable as index_figures.py / answer.py
FIGURES_JSON = IMAGE_DIR / "figures.json"

MIN_SIZE = 150       # ignore pictures smaller than this in either direction (logos, icons)
IMAGES_SCALE = 2.0   # render pictures at 2x
DO_OCR = False       # arXiv PDFs have a text layer; OCR mostly slows conversion down. True = previous behaviour.

ARXIV_IDS = [
    "2312.10997", "2310.11511", "2401.15884", "2407.01449", "2407.12883", "2005.11401", "2004.04906",
    "2407.01219", "2501.09136", "2401.05856", "2510.13975", "2407.12216", "2604.00865", "2608.20627",
    "2307.03172", "2401.14887", "2309.15217", "2311.09476", "2408.08067", "2409.12941", "2212.10496",
    "2402.03216", "2304.09542", "2312.06648", "2401.18059", "2305.06983", "2503.09516", "2404.16130",
    "2408.09869", "2410.10594",
]

FIG = re.compile(r"^\s*fig(?:ure)?s?\.?\s*(\d+)", re.I)  # "Figure 3: ...", "Fig. 3. ..."


# --------------------------------------------------
# 1. Download
# --------------------------------------------------

def download_docs(paper_ids: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for pid in paper_ids:
        out = RAW_DIR / f"{pid}.pdf"
        if out.exists():
            continue
        print(f"downloading {pid}")
        try:
            response = requests.get(f"https://arxiv.org/pdf/{pid}", timeout=60)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            part = out.with_suffix(".part")  # write then rename, so a crash never leaves a broken PDF
            part.write_bytes(response.content)
            part.replace(out)
        except Exception as exc:
            print(f"  FAILED {pid}: {exc}")
        time.sleep(1)  # be polite to arXiv


# --------------------------------------------------
# 2. Convert once (cached)
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_converter() -> DocumentConverter:
    """Built lazily: if every PDF is cached, the Docling models are never loaded."""
    opts = PdfPipelineOptions()
    opts.images_scale = IMAGES_SCALE
    opts.generate_picture_images = True  # without this Docling keeps no picture pixels
    opts.do_ocr = DO_OCR
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def load_cached(stem: str) -> DoclingDocument | None:
    path = CACHE_DIR / f"{stem}.json"
    if not path.exists():
        return None
    try:
        doc = DoclingDocument.load_from_json(path)
        if doc.pictures and all(pic.get_image(doc) is None for pic in doc.pictures):
            raise ValueError("cached JSON has no picture pixels")
        return doc
    except Exception as exc:
        print(f"  cache unusable ({exc}); reconverting")
        return None


def get_document(pdf: Path, force: bool) -> DoclingDocument:
    if not force:
        doc = load_cached(pdf.stem)
        if doc is not None:
            print("  loaded from cache")
            return doc

    print("  converting with Docling (slow step)...")
    doc = get_converter().convert(str(pdf)).document
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(CACHE_DIR / f"{pdf.stem}.json", image_mode=ImageRefMode.EMBEDDED)
    except Exception as exc:
        print(f"  could not write cache ({exc}); continuing without it")
    return doc


# --------------------------------------------------
# 3. Export
# --------------------------------------------------

def get_caption(doc, pic, items, pos_by_ref, taken: set[str]) -> str:
    """Caption Docling linked to the picture; if none, the nearest 'Figure N' text that no other picture owns."""
    caption = (pic.caption_text(doc) or "").strip()
    if not caption:
        pos = pos_by_ref.get(pic.self_ref)
        if pos is not None:
            for j in (pos + 1, pos - 1, pos + 2, pos - 2):  # arXiv captions sit below the figure
                if 0 <= j < len(items):
                    item = items[j]
                    if isinstance(item, TextItem) and FIG.match(item.text) and item.self_ref not in taken:
                        taken.add(item.self_ref)  # one caption text belongs to one picture only
                        caption = item.text.strip()
                        break
    return " ".join(caption.split())


def project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def export_figures(doc, stem: str) -> list[dict[str, Any]]:
    items = [item for item, _ in doc.iterate_items()]
    pos_by_ref = {item.self_ref: i for i, item in enumerate(items)}
    taken = {ref.cref for pic in doc.pictures for ref in getattr(pic, "captions", [])}  # captions Docling already linked

    records, used, written = [], set(), set()
    for n, pic in enumerate(doc.pictures, start=1):
        img = pic.get_image(doc)
        if img is None or img.width < MIN_SIZE or img.height < MIN_SIZE:
            continue
        caption = get_caption(doc, pic, items, pos_by_ref, taken)
        if not caption:
            continue

        m = FIG.match(caption)
        base = f"{stem}_fig_{m.group(1)}" if m else f"{stem}_fig_u{n}"  # named by the caption's figure number
        fig_id, k = base, 1
        while fig_id in used:  # several pictures under one caption (sub-figures)
            k += 1
            fig_id = f"{base}_{k}"
        used.add(fig_id)

        path = IMAGE_DIR / f"{fig_id}.png"
        img.save(path)
        written.add(path)
        records.append({
            "id": fig_id,
            "paper": stem,
            "page": pic.prov[0].page_no if pic.prov else None,
            "label": f"Figure {m.group(1)}" if m else "",
            "caption": caption,
            "image_path": project_relative(path),
        })

    # only after a successful export: drop leftovers from earlier runs / other extractors for this paper
    for old in IMAGE_DIR.glob(f"{stem}_*"):
        if old.is_file() and old not in written:
            old.unlink()
    return records


def write_figures_json(new_records: list[dict[str, Any]], processed: set[str]) -> int:
    """Merge into figures.json: papers processed now are replaced, all other papers are kept."""
    existing: list[dict[str, Any]] = []
    if FIGURES_JSON.exists():
        try:
            existing = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"existing figures.json unreadable ({exc}); starting fresh")

    merged = [r for r in existing if r.get("paper") not in processed] + new_records
    merged.sort(key=lambda r: (r["paper"], r["id"]))

    tmp = FIGURES_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FIGURES_JSON)
    return len(merged)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv PDFs, parse them once with Docling, export markdown + figures.")
    parser.add_argument("--download", action="store_true", help="download missing PDFs first (ARXIV_IDS, or --only)")
    parser.add_argument("--force", action="store_true", help="ignore the Docling cache and reconvert")
    parser.add_argument("--only", nargs="+", metavar="ID", help="process only these paper ids")
    args = parser.parse_args()

    if args.download:
        download_docs(args.only or ARXIV_IDS)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if args.only:
        wanted = set(args.only)
        pdfs = [p for p in pdfs if p.stem in wanted]
        for missing in sorted(wanted - {p.stem for p in pdfs}):
            print(f"no PDF for {missing} in {RAW_DIR} (use --download)")
    if not pdfs:
        print(f"No PDFs to process in {RAW_DIR}. Use --download.")
        return

    MD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    done: set[str] = set()
    failed: list[str] = []

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf.stem}")
        try:  # one bad PDF must not kill a 30-paper run
            doc = get_document(pdf, args.force)
            (MD_DIR / f"{pdf.stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
            paper_records = export_figures(doc, pdf.stem)
        except Exception as exc:
            traceback.print_exc()
            print(f"  FAILED {pdf.stem}: {exc}")
            failed.append(pdf.stem)
            continue
        records += paper_records
        done.add(pdf.stem)
        print(f"  {len(paper_records)} figure(s)")

    total = write_figures_json(records, done)

    print(f"\nDone: {len(done)}/{len(pdfs)} papers, {len(records)} figures this run, {total} in {FIGURES_JSON}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()