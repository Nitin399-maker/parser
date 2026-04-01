import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://llmfoundry.straivedemo.com/openrouter/v1/chat/completions"
MODEL = "google/gemini-3.1-flash-image-preview"

IMAGE_AREA_THRESHOLD = 50_000  # px² — minimum area to be considered a "large" image
IMAGE_MIN_DIMENSION = 100      # px  — each side must be at least this many pixels
CONTENT_BATCH_SIZE = 5  # pages per LLM call for combined table+image extraction
MAX_RETRIES = 1
RETRY_DELAY = 2.0
HTTP_TIMEOUT = 180     # seconds — large page images need more time

# Cost tracking (accumulated across all LLM calls)
_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_usd = 0.0


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict], retries: int = MAX_RETRIES) -> str:
    """Synchronous OpenRouter call with retry logic. Accumulates usage/cost."""
    global _total_input_tokens, _total_output_tokens, _total_cost_usd
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "messages": messages}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract usage and cost if present
            usage = data.get("usage", {})
            print("LLM call usage:", usage)  # Debug print for usage details
            _total_input_tokens += usage.get("prompt_tokens", 0)
            _total_output_tokens += usage.get("completion_tokens", 0)
            
            cost = usage.get("cost_details", {}).get("upstream_inference_cost", 0.0)
            _total_cost_usd += cost
            
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError("LLM call failed after all retries")


async def _call_llm_async(messages: list[dict]) -> str:
    """Run blocking LLM call in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm, messages)


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

def extract_file_metadata(pdf_path: str) -> dict[str, Any]:
    """Extract file-level metadata from a PDF."""
    path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    doc.close()

    def _clean(val: str | None) -> str:
        return val.strip() if val else ""

    return {
        "file_name": path.name,
        "total_pages": fitz.open(pdf_path).page_count,
        "author": _clean(meta.get("author")),
        "creation_date": _clean(meta.get("creationDate")),
        "modification_date": _clean(meta.get("modDate")),
        "file_size": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Page metadata
# ---------------------------------------------------------------------------

def extract_page_metadata(page: fitz.Page) -> dict[str, Any]:
    """Extract page-level metadata."""
    rect = page.rect
    return {
        "page_number": page.number + 1,
        "width": round(rect.width, 2),
        "height": round(rect.height, 2),
    }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(page: fitz.Page) -> str:
    """Extract clean, ordered text from a page."""
    blocks = page.get_text("blocks", sort=True)
    paragraphs: list[str] = []
    for block in blocks:
        if block[6] != 0:  # skip image blocks
            continue
        raw = block[4]
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def detect_heading(page: fitz.Page) -> str:
    """Infer the best heading from font size, weight, and position."""
    blocks = page.get_text("dict", sort=True).get("blocks", [])
    candidates: list[tuple[float, float, str]] = []  # (font_size, y_pos, text)

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text or len(text) < 2:
                    continue
                size = span.get("size", 0)
                flags = span.get("flags", 0)
                bold_bonus = 2.0 if flags & 2**4 else 0.0  # bold flag
                score = size + bold_bonus
                y0 = block["bbox"][1]
                candidates.append((score, y0, text))

    if not candidates:
        return ""

    # Pick the span with the highest score; break ties by y position (top first)
    best = max(candidates, key=lambda c: (c[0], -c[1]))
    return best[2]


# ---------------------------------------------------------------------------
# Table detection (pdfplumber)
# ---------------------------------------------------------------------------

def detect_tables_pdfplumber(pdf_path: str) -> dict[int, bool]:
    """
    Returns a mapping of {page_number (1-based): has_table (bool)}
    using pdfplumber for detection only.
    """
    result: dict[int, bool] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            result[i] = len(tables) > 0
    return result


# ---------------------------------------------------------------------------
# Image detection (PyMuPDF)
# ---------------------------------------------------------------------------

def detect_images(page: fitz.Page, doc: fitz.Document) -> list[dict[str, Any]]:
    """
    Detect large images on a page. Returns list of base64-encoded images
    with metadata.
    """
    images: list[dict[str, Any]] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        width = img_info[2]
        height = img_info[3]
        # Skip small images: icons, logos, decorative lines, etc.
        if width * height < IMAGE_AREA_THRESHOLD:
            continue
        if width < IMAGE_MIN_DIMENSION or height < IMAGE_MIN_DIMENSION:
            continue
        try:
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append({"base64": b64, "ext": ext, "width": width, "height": height})
        except Exception as exc:
            logger.warning("Could not extract image xref=%d: %s", xref, exc)
    return images


# ---------------------------------------------------------------------------
# LLM: combined table + image extraction (single request per batch)
# ---------------------------------------------------------------------------

def _render_page_png(page: fitz.Page) -> str:
    """Render a PDF page to a base64-encoded PNG at 2× zoom."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


async def process_content_llm(
    pdf_path: str,
    pages_with_tables: set[int],
    pages_with_images: set[int],
) -> dict[int, dict[str, Any]]:
    """
    Union of pages that have tables or images (or both).
    Each page is rendered to PNG and sent to the LLM in batches.
    One combined prompt extracts tables AND image description in a single call.

    Returns:
        {
            page_number: {
                "tables": [...],          # list of tables (list of list of str)
                "page_images": [          # one entry per significant image/visual
                    {
                        "summary": "...",
                        "details": "...",
                        "markdown": "..."   # markdown of any text/data inside the image; "" if none
                    }
                ]
            }
        }
    """
    all_pages = sorted(pages_with_tables | pages_with_images)
    if not all_pages:
        return {}

    doc = fitz.open(pdf_path)
    results: dict[int, dict[str, Any]] = {}

    system_msg = (
        "You are a precise document analysis assistant. "
        "You will be shown one or more PDF page images. "
        "For each page, extract ALL tables and describe ALL significant images. "
        "Return ONLY a valid JSON array — no markdown, no explanation — in this exact format:\n"
        "[\n"
        "  {\n"
        '    "page_number": <int>,\n'
        '    "markdown_table": "<all tables on this page rendered as pipe-delimited GitHub markdown, each table having a header row and separator row (e.g. |---|---|); if multiple tables exist separate them with a blank line; empty string if the page has no tables>",\n'
        '    "page_images": [\n'
        '      {"summary": "<one sentence>", "details": "<2-3 sentences>", "markdown": "<markdown representation of any text, data, or structured content visible inside the image; empty string if the image contains no readable text or data>"}\n'
        "    ]\n"
        "  }\n"
        "]\n\n"
        "RULES:\n"
        "- must give the page number same as mentioned in the user request while sending the pages of the pdf.\n"
        "- If a page has no tables, set \"markdown_table\" to an empty string \"\".\n"
        "- Each table must be rendered as a full GitHub-style pipe-delimited markdown table with a header row and separator row. Do not truncate any cells.\n"
        "- If a page has no significant images, set \"page_images\" to [].\n"
        "- If an image contains a table, extract all tables with all data.\n"
        "- For each image entry, always include the \"markdown\" key. "
        "Set it to a markdown string if the image contains text, tables, charts with labels, "
        "captions, or any readable data. Set it to an empty string \"\" if the image is purely graphical with no readable text.\n"
        "- CRITICAL JSON ESCAPING RULES — failure to follow these will cause a Python json.loads parse error:\n"
        "  1. The ONLY valid backslash escape sequences in a JSON string are: \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX\n"
        "  2. NEVER write \\| or \\& or \\; or \\( or \\) or \\[ or \\] or \\v or \\i or any other backslash sequence not listed above — these are INVALID JSON escapes and will crash the parser.\n"
        "  3. A literal pipe | ampersand & semicolon ; parenthesis ( ) bracket [ ] needs NO backslash — just write the character as-is.\n"
        "  4. A literal backslash in your text MUST be written as \\\\ (two backslashes). A single lone \\ is never valid in JSON.\n"
        "  5. You MAY use \\n (backslash + n) inside a JSON string to represent a line break between rows.\n"
        "  6. NEVER embed a raw newline byte (ASCII 0x0A) or raw tab byte (0x09) directly inside a JSON string.\n"
        "  7. Any double-quote inside a string value MUST be escaped as \\\".\n"
        "- Inside table cells (pipe-delimited markdown), replace any literal newlines with a single space so each row stays on one \\n-separated line.\n"
        "- Some pages may contain rotated or sideways content (e.g. landscape tables, text rotated 90° or 270°). Always read and extract content in its correct logical reading orientation regardless of rotation.\n"
        "- The markdown field must contain only exact text visible inside the image. Do not add summaries or explanations inside the markdown field.\n"
        "- The output must be parseable by Python's json.loads without errors.\n"
        "- Must not hallucinate values inside tables and images."
        "- sometimes the cell of the table or header contain content in multiple lines, in that case must not include the next line content in next raw keep it the same raw content."
    )

    async def _process_batch(batch: list[int]) -> None:
        messages_content: list[dict] = []
        for pno in batch:
            page = doc[pno - 1]
            img_b64 = _render_page_png(page)
            messages_content.append({"type": "text", "text": f"Page {pno}:"})
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })

        messages_content.append({
            "type": "text",
            "text": (
                "For each page shown above, extract all tables and describe all significant "
                "images. Return the JSON array exactly as described in the system message."
            ),
        })

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": messages_content},
        ]

        try:
            raw = await _call_llm_async(messages)
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw.strip())
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            # Remove literal control characters that are illegal inside JSON strings.
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
            # Fix invalid JSON escape sequences the LLM may emit (e.g. \| \& \; \v \i).
            # Valid JSON escapes after \ are: " \ / b f n r t u
            # Replace any \x where x is NOT one of those with just x (drop the backslash).
            raw = re.sub(r'\\(?!["\\/bfnrtu])', '', raw)
            parsed: list[dict] = json.loads(raw)
            for item in parsed:
                pno = item.get("page_number")
                if pno is None:
                    continue
                def _strip_br(text: str) -> str:
                    return re.sub(r"<[Bb][Rr]\s*/?>", " ", text)

                raw_images = item.get("page_images", [])
                page_images = [
                    {
                        "summary": img.get("summary", ""),
                        "details": img.get("details", ""),
                        "markdown": _strip_br(img.get("markdown", "")),
                    }
                    for img in raw_images
                ]
                results[pno] = {
                    "markdown_table": _strip_br(item.get("markdown_table", "")),
                    "page_images": page_images,
                }
        except Exception as exc:
            logger.error("Content LLM batch failed for pages %s: %s", batch, exc)
            for pno in batch:
                results[pno] = {"markdown_table": "", "page_images": []}

    batches = [
        all_pages[i: i + CONTENT_BATCH_SIZE]
        for i in range(0, len(all_pages), CONTENT_BATCH_SIZE)
    ]
    # Fire all batches concurrently — every batch is a separate LLM request
    # sent simultaneously (e.g. 50 pages / 5 per batch = 10 requests at once).
    await asyncio.gather(*[_process_batch(b) for b in batches])

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def build_pages_context(
    pdf_path: str,
    content_results: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble the pages_context list from all extracted data."""
    doc = fitz.open(pdf_path)
    pages_context: list[dict[str, Any]] = []

    for page in doc:
        pno = page.number + 1
        content = content_results.get(pno, {})
        pages_context.append({
            "page_metadata": extract_page_metadata(page),
            "page_heading": detect_heading(page),
            "page_tables": content.get("markdown_table", ""),
            "page_text": extract_text(page),
            "page_images": content.get("page_images", []),
            "page_tables": content.get("markdown_table", ""),
        })

    doc.close()

    return pages_context


async def parse_pdf(pdf_path: str) -> dict[str, Any]:
    """
    Full PDF parsing pipeline.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Structured JSON-serialisable dict.
    """
    logger.info("Starting PDF parse: %s", pdf_path)

    # ── 1. File metadata ───────────────────────────────────────────────────
    file_metadata = extract_file_metadata(pdf_path)
    total_pages = file_metadata["total_pages"]
    logger.info("Total pages: %d", total_pages)

    # ── 2. Detect tables (pdfplumber) ──────────────────────────────────────
    logger.info("Detecting tables with pdfplumber …")
    table_map = detect_tables_pdfplumber(pdf_path)
    pages_with_tables = {pno for pno, has in table_map.items() if has}
    logger.info("Pages with tables: %s", sorted(pages_with_tables))

    # ── 3. Detect images (PyMuPDF) ─────────────────────────────────────────
    doc = fitz.open(pdf_path)
    pages_with_images: set[int] = set()
    for page in doc:
        pno = page.number + 1
        if detect_images(page, doc):
            pages_with_images.add(pno)
    doc.close()
    logger.info("Pages with images: %s", sorted(pages_with_images))

    # ── 4. Single combined LLM pass (tables + images together) ────────────
    union_pages = sorted(pages_with_tables | pages_with_images)
    logger.info(
        "Union pages (tables + images, %d total): %s",
        len(union_pages),
        union_pages,
    )
    _llm_start = time.time()
    content_results = await process_content_llm(
        pdf_path, pages_with_tables, pages_with_images
    )
    _llm_latency = time.time() - _llm_start

    # ── 5. Assemble output ─────────────────────────────────────────────────
    pages_context = await build_pages_context(pdf_path, content_results)

    result = {
        "file_metadata": file_metadata,
        "pages_context": pages_context,
    }

    logger.info("PDF parsing complete.")
    _total_tokens = _total_input_tokens + _total_output_tokens
    summary_lines = [
        "",
        "✅ LLM Summary",
        f"   Total input tokens  : {_total_input_tokens:,}",
        f"   Total output tokens : {_total_output_tokens:,}",
        f"   Total tokens        : {_total_tokens:,}",
        f"   Total cost          : ${_total_cost_usd:.4f}",
        f"   Latency             : {int(_llm_latency)} seconds",
        "",
    ]
    for line in summary_lines:
        logger.info(line)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf> [output.json]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else Path(pdf_file).stem + ".json"

    output = asyncio.run(parse_pdf(pdf_file))

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Output written to {out_file}")
