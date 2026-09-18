"""
Timestamp-Linked Ask-the-Manual Engine — Ingestion & Retrieval starter
========================================================================

Indexes an SOP PDF alongside a video transcript (Whisper word-level
timestamps) and links each procedural step to BOTH a PDF page and a
video timestamp range, so a single retrieval returns a text answer,
a page citation, and the exact moment to seek to in the video.

This is a starter/demo implementation: it's structured so each stage
(parse -> transcribe -> align -> index -> query) is a separate,
readable function you can swap out for the real tech stack
(LlamaIndex + Qdrant/Chroma + Whisper) without changing the shape
of the pipeline.

Run:
    pip install sentence-transformers pytesseract pdf2image numpy
    python ingest.py

Requires (installed separately, not via pip):
    - Tesseract OCR engine  https://github.com/UB-Mannheim/tesseract/wiki
    - Poppler (for pdf2image)  https://github.com/oschwartz10612/poppler-windows/releases
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np



def correct_orientation(image):
    """Detect and fix page rotation using Tesseract's OSD, before OCR."""
    import pytesseract
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int(
            [line for line in osd.split("\n") if "Rotate:" in line][0].split(":")[1].strip()
        )
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
    except Exception:
        pass
    return image


# ---------------------------------------------------------------------
# 1. Data model — every chunk can carry BOTH a PDF address and a
#    video address. That dual-address node is the core idea.
# ---------------------------------------------------------------------

@dataclass
class Node:
    id: str
    text: str
    source_type: str  # "pdf" or "video"
    pdf_page: Optional[int] = None
    pdf_section: Optional[str] = None
    video_start_ms: Optional[int] = None
    video_end_ms: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    linked_node_id: Optional[str] = None
    link_confidence: float = 0.0


# ---------------------------------------------------------------------
# 2. PDF ingestion (page-aware chunking)
#    This manual is a SCANNED PDF (no text layer), so extraction goes
#    through OCR in the __main__ block below, not plain PyMuPDF text
#    extraction. This function just needs {page_number: page_text} —
#    it doesn't care whether that text came from a text layer or OCR.
# ---------------------------------------------------------------------

def parse_sop_pdf(pages: dict[int, str]) -> list[Node]:
    """pages: {page_number: raw_page_text}."""
    nodes = []
    for page_num, page_text in pages.items():
        # split on "Annexure 4X" headings (X = a letter, e.g. 4A, 4B, 4M)
        sections = re.split(r"(?=Annexure\s+\d+[A-Z]\b)", page_text)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading_match = re.match(
                r"(Annexure\s+\d+[A-Z]\s*\n?[^\n]*)", section
            )
            heading = heading_match.group(1).replace("\n", " ").strip() if heading_match else None
            nodes.append(
                Node(
                    id=f"pdf-p{page_num}-{len(nodes)}",
                    text=section,
                    source_type="pdf",
                    pdf_page=page_num,
                    pdf_section=heading,
                )
            )
    return nodes


# ---------------------------------------------------------------------
# 3. Video transcript ingestion (Whisper word-level timestamps)
#    Real version:
#        import whisper
#        model = whisper.load_model("small")
#        result = model.transcribe(video_path, word_timestamps=True)
#        words = [w for seg in result["segments"] for w in seg["words"]]
#    Then group `words` the same way group_words_into_segments() does.
# ---------------------------------------------------------------------

def group_words_into_segments(
    words: list[dict], max_gap_ms: int = 800, max_segment_ms: int = 15000
) -> list[Node]:
    """
    words: [{"word": str, "start_ms": int, "end_ms": int}, ...]
    Groups words into segments on pause boundaries, so each segment
    reads like one spoken step rather than one word or one full video.
    """
    segments: list[Node] = []
    current_words: list[dict] = []

    def flush():
        if not current_words:
            return
        text = " ".join(w["word"] for w in current_words).strip()
        segments.append(
            Node(
                id=f"video-{len(segments)}",
                text=text,
                source_type="video",
                video_start_ms=current_words[0]["start_ms"],
                video_end_ms=current_words[-1]["end_ms"],
            )
        )

    for w in words:
        if current_words:
            gap = w["start_ms"] - current_words[-1]["end_ms"]
            span = w["end_ms"] - current_words[0]["start_ms"]
            if gap > max_gap_ms or span > max_segment_ms:
                flush()
                current_words = []
        current_words.append(w)
    flush()
    return segments


# ---------------------------------------------------------------------
# 4. Embeddings
#    Real version: sentence-transformers or the LlamaIndex embedding
#    wrapper around your model of choice. Kept pluggable here.
# ---------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True)
    except ImportError:
        # Deterministic fallback so the pipeline is runnable without
        # the extra dependency installed — replace with the real
        # embedding model before this goes anywhere near production.
        rng = np.random.default_rng(42)
        dim = 64
        vecs = np.array(
            [rng.normal(size=dim) + hash(t) % 7 for t in texts]
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------
# 5. Alignment — link every PDF chunk to its best-matching video
#    segment(s) at INDEX time, so query time does zero extra work.
#    This is the piece that makes the "seek to the right frame"
#    feature possible at all.
# ---------------------------------------------------------------------

def align_pdf_to_video(
    pdf_nodes: list[Node], video_nodes: list[Node], min_confidence: float = 0.35
) -> None:
    pdf_embeds = embed_texts([n.text for n in pdf_nodes])
    video_embeds = embed_texts([n.text for n in video_nodes])

    for i, pdf_node in enumerate(pdf_nodes):
        pdf_node.embedding = pdf_embeds[i]
        best_score = -1.0
        best_video_node = None
        for j, video_node in enumerate(video_nodes):
            score = cosine_sim(pdf_embeds[i], video_embeds[j])
            if score > best_score:
                best_score = score
                best_video_node = video_node

        if best_video_node and best_score >= min_confidence:
            pdf_node.linked_node_id = best_video_node.id
            pdf_node.link_confidence = round(best_score, 3)
            pdf_node.video_start_ms = best_video_node.video_start_ms
            pdf_node.video_end_ms = best_video_node.video_end_ms
        # else: this SOP step has no confident video match —
        # worth surfacing as a "coverage gap" in the product.


# ---------------------------------------------------------------------
# 6. Query — retrieve, and return answer + page citation + timestamp
#    together, since the link was already resolved at index time.
# ---------------------------------------------------------------------

def query(question: str, pdf_nodes: list[Node], top_k: int = 1) -> list[dict]:
    q_embed = embed_texts([question])[0]
    scored = [
        (cosine_sim(q_embed, n.embedding), n)
        for n in pdf_nodes
        if n.embedding is not None
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, node in scored[:top_k]:
        results.append(
            {
                "answer_text": node.text,
                "score": round(score, 3),
                "citation": {"pdf_page": node.pdf_page, "section": node.pdf_section},
                "video": {
                    "start_ms": node.video_start_ms,
                    "end_ms": node.video_end_ms,
                    "confidence": node.link_confidence,
                }
                if node.video_start_ms is not None
                else None,
            }
        )
    return results


# ---------------------------------------------------------------------
# 7. Demo run against the real scanned manual (OCR'd) + placeholder
#    video narration (swap for real Whisper output once you have a
#    recorded video).
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import pytesseract
    from pdf2image import convert_from_path

    # --- EDIT THESE TWO PATHS to match your own install locations ---
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin\poppler-26.09.0\Library\bin"
    # ------------------------------------------------------------------

    pdf_path = "data/Repair_Methodology.pdf"

    print("Rasterizing PDF pages...")
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    print(f"Rasterized {len(images)} pages, running OCR (this takes a minute)...")
    sample_pdf_pages: dict[int, str] = {}
    for page_num, image in enumerate(images, start=1):
        image = correct_orientation(image)
        text = pytesseract.image_to_string(image)
        print(f"  page {page_num}: {len(text)} characters")
        sample_pdf_pages[page_num] = text

    sample_words = [
        {"word": "First,", "start_ms": 15000, "end_ms": 15300},
        {"word": "chip", "start_ms": 15300, "end_ms": 15550},
        {"word": "back", "start_ms": 15550, "end_ms": 15750},
        {"word": "the", "start_ms": 15750, "end_ms": 15850},
        {"word": "loose", "start_ms": 15850, "end_ms": 16150},
        {"word": "concrete", "start_ms": 16150, "end_ms": 16600},
        {"word": "around", "start_ms": 16600, "end_ms": 16900},
        {"word": "the", "start_ms": 16900, "end_ms": 17000},
        {"word": "corroded", "start_ms": 17000, "end_ms": 17450},
        {"word": "bar.", "start_ms": 17450, "end_ms": 17700},
        # gap > 800ms -> new segment
        {"word": "Now", "start_ms": 19000, "end_ms": 19200},
        {"word": "wire-brush", "start_ms": 19200, "end_ms": 19750},
        {"word": "the", "start_ms": 19750, "end_ms": 19850},
        {"word": "steel", "start_ms": 19850, "end_ms": 20150},
        {"word": "until", "start_ms": 20150, "end_ms": 20400},
        {"word": "it's", "start_ms": 20400, "end_ms": 20550},
        {"word": "fully", "start_ms": 20550, "end_ms": 20800},
        {"word": "clean,", "start_ms": 20800, "end_ms": 21150},
        {"word": "then", "start_ms": 21150, "end_ms": 21350},
        {"word": "apply", "start_ms": 21350, "end_ms": 21650},
        {"word": "the", "start_ms": 21650, "end_ms": 21750},
        {"word": "zinc", "start_ms": 21750, "end_ms": 22050},
        {"word": "primer", "start_ms": 22050, "end_ms": 22450},
        {"word": "in", "start_ms": 22450, "end_ms": 22550},
        {"word": "two", "start_ms": 22550, "end_ms": 22750},
        {"word": "coats.", "start_ms": 22750, "end_ms": 23100},
    ]

    print("Building embeddings and aligning PDF <-> video (first run downloads the model)...")
    pdf_nodes = parse_sop_pdf(sample_pdf_pages)
    print(f"Found {len(pdf_nodes)} PDF chunks after splitting on Annexure headings.")

    for n in pdf_nodes:
        preview = n.text.replace("\n", " ")[:80]
        print(f"  [page {n.pdf_page}] section={n.pdf_section!r} :: {preview}")

    video_nodes = group_words_into_segments(sample_words)
    align_pdf_to_video(pdf_nodes, video_nodes)

    results = query("How do I treat exposed corroded reinforcement?", pdf_nodes, top_k=5)
    print(json.dumps(results, indent=2))