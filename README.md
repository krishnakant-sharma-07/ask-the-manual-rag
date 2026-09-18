# Ask the Manual — Timestamp-Linked Engine

A retrieval engine that indexes an SOP PDF alongside a training video and links every procedural step to **both** a PDF page citation **and** an exact video timestamp — so a single question returns a text answer, a page reference, and a video already seeked to the moment that step is demonstrated.

Built as a technical proof of concept, tested end to end against a real 15-page structural repair methodology manual (corrosion treatment, injection grouting, termite infestation, steel retrofitting).

**[Try the live demo →](#)** *(link goes here once GitHub Pages is set up — see below)*

---

## Why this exists

Most "chat with your manual" tools stop at a text answer. A field worker asking *"how do I treat exposed corroded reinforcement?"* doesn't want three paragraphs back — they want the ten seconds of video where someone's hands actually do it. The gap between *"here's the answer"* and *"here's the moment"* is what this project closes.

## How it works

```
SOP PDF ──► OCR (Tesseract) ──► page-aware, section-aware chunks
                                          │
Training video ──► Whisper (word-level timestamps) ──► spoken segments
                                          │
                              Semantic alignment
                    (each PDF chunk linked to its best-matching
                     video segment, AT INDEX TIME — not query time)
                                          │
                              Vector store (embeddings)
                                          │
        Question ──► retrieval ──► answer + page citation + video timestamp,
                                    all returned together
```

The hard part isn't the search — it's linking a PDF chunk to a video timestamp *before* anyone asks a question, so retrieval does zero extra work at query time. See [`ingest/ingest.py`](ingest/ingest.py) for the full pipeline: PDF chunking, transcript segmenting, embedding, and alignment.

## What's in this repo

| Path | What it is |
|---|---|
| `ingest/ingest.py` | The core pipeline — OCR extraction, chunking, embedding, alignment, and retrieval. Runnable standalone. |
| `web/demo.html` | A self-contained interactive demo — search a question, see the matched answer, citation, and a timeline that jumps to the matched moment. No build step, no dependencies. |
| `api/` | Reserved for a FastAPI layer wiring the ingestion pipeline to the demo UI (in progress). |

## Running it locally

**Requirements:** Python 3.10+, [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki), [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (Windows) for PDF rasterization.

```bash
python -m venv venv
venv\Scripts\activate      # or source venv/bin/activate on Mac/Linux
pip install sentence-transformers pytesseract pdf2image numpy

python ingest/ingest.py
```

Edit the `tesseract_cmd` and `POPPLER_PATH` variables near the top of the `__main__` block to match your local install paths.

To try the interactive demo, just open `web/demo.html` directly in a browser — no server needed.

## Status

- [x] Real PDF ingestion via OCR, with automatic per-page orientation correction
- [x] Page-and-section-aware chunking
- [x] Semantic embedding and PDF↔video alignment
- [x] Working retrieval with citation + confidence scoring
- [x] Interactive demo UI
- [ ] Real Whisper transcription (currently using placeholder narration for the video side)
- [ ] FastAPI endpoint connecting the pipeline to the demo UI live
- [ ] Coverage-gap detection for SOP steps with no matching video

## Author

Krishnakant Sharma — [LinkedIn](https://linkedin.com/in/krishnakant-sharma-33153626b) · [krish511sharma@gmail.com]()
