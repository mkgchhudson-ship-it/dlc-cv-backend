"""
Direct Labour Consult — CV Analysis Backend
============================================
Production-ready Flask API.

Fixes applied:
  - Anthropic client initialised without unsupported kwargs
  - Universal CV text extraction pipeline with fallbacks
  - Structured JSON response format throughout
  - Full request logging to console (visible in Render logs)
  - 500-proof error handling on every path

Files are NEVER stored — deleted immediately after processing.
"""

import io
import json
import logging
import os
import re
import tempfile
import traceback
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("dlc_cv")

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])          # tighten to your domain in production

UPLOAD_DIR     = tempfile.gettempdir()
ALLOWED_EXTS   = {"pdf", "doc", "docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_TEXT_CHARS = 12_000             # trim very long CVs before sending to Claude


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_ext(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS
    )


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


# ── Text extraction pipeline ─────────────────────────────────────────────────

def _extract_pdf_pymupdf(path: str) -> str:
    """Step 1 — PyMuPDF (fastest, best for text PDFs)."""
    import fitz  # PyMuPDF
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def _extract_pdf_pdfminer(path: str) -> str:
    """Step 2 — pdfminer.six (better for complex layouts)."""
    from pdfminer.high_level import extract_text as pm_extract
    return (pm_extract(path) or "").strip()


def _extract_pdf_ocr(path: str) -> str:
    """Step 3 — pytesseract OCR (scanned/image PDFs)."""
    import fitz
    import pytesseract
    from PIL import Image

    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_parts.append(pytesseract.image_to_string(img))
    return "\n".join(text_parts).strip()


def _extract_docx(path: str) -> str:
    """Step 4 — python-docx for .docx files."""
    from docx import Document
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    # Also pull table cells
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts).strip()


def _extract_doc_fallback(path: str) -> str:
    """Step 5 — basic .doc extraction via textract or raw byte scan."""
    try:
        import textract
        return textract.process(path).decode("utf-8", errors="ignore").strip()
    except Exception:
        pass
    # Last resort: scan raw bytes for printable ASCII runs
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("latin-1", errors="ignore")
        # Keep only printable characters and whitespace
        cleaned = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
        cleaned = re.sub(r" {3,}", " ", cleaned)
        return cleaned.strip()
    except Exception:
        return ""


def extract_text(path: str, ext: str) -> tuple[str, str]:
    """
    Universal extraction pipeline.
    Returns (text, method_used).
    Never raises — always returns something or empty string.
    """
    if ext in ("doc", "docx"):
        if ext == "docx":
            try:
                text = _extract_docx(path)
                if text:
                    return text, "python-docx"
            except Exception as exc:
                log.warning("python-docx failed: %s", exc)
        # .doc or docx fallback
        try:
            text = _extract_doc_fallback(path)
            if text:
                return text, "doc-fallback"
        except Exception as exc:
            log.warning("doc-fallback failed: %s", exc)
        return "", "none"

    # PDF pipeline
    try:
        text = _extract_pdf_pymupdf(path)
        if text and len(text) >= 80:
            return text, "pymupdf"
        log.info("PyMuPDF returned sparse text (%d chars), trying pdfminer", len(text))
    except Exception as exc:
        log.warning("PyMuPDF failed: %s", exc)

    try:
        text = _extract_pdf_pdfminer(path)
        if text and len(text) >= 80:
            return text, "pdfminer"
        log.info("pdfminer returned sparse text (%d chars), trying OCR", len(text))
    except Exception as exc:
        log.warning("pdfminer failed: %s", exc)

    try:
        text = _extract_pdf_ocr(path)
        if text:
            return text, "ocr-tesseract"
        log.warning("OCR returned empty text")
    except Exception as exc:
        log.warning("OCR failed: %s", exc)

    return "", "none"


# ── Claude analysis ───────────────────────────────────────────────────────────

def analyse_cv(cv_text: str, name: str) -> dict:
    """
    Send extracted CV text to Claude and return structured analysis dict.
    Raises on unrecoverable errors so the caller can handle them.
    """
    # ── Fix: clean initialisation — no unsupported kwargs ──
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Trim to avoid token bloat
    if len(cv_text) > MAX_TEXT_CHARS:
        cv_text = cv_text[:MAX_TEXT_CHARS] + "\n[... document continues ...]"

    prompt = f"""You are a senior HR consultant at Direct Labour Consult, an HR advisory firm
based in Gaborone, Botswana. Review this CV with the same rigour a hiring manager
applies when screening candidates for competitive roles.

Candidate name: {name}

CV TEXT:
---
{cv_text}
---

Return ONLY a raw JSON object — no markdown, no code fences, no preamble.
Exact structure required:

{{
  "candidate_name": "{name}",
  "overall_score": <integer 0-100>,
  "market_readiness": "<Excellent | Strong | Developing | Needs Improvement>",
  "executive_summary": "<2-3 honest sentences on this CV's current market position>",
  "sections": {{
    "professional_summary": {{
      "score": <integer 0-100>,
      "feedback": "<specific feedback on clarity, strength, positioning>",
      "strengths": ["<strength>", "<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "work_experience": {{
      "score": <integer 0-100>,
      "feedback": "<feedback on impact, achievements, structure>",
      "strengths": ["<strength>", "<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "skills": {{
      "score": <integer 0-100>,
      "feedback": "<feedback on relevance, keyword alignment>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "formatting_readability": {{
      "score": <integer 0-100>,
      "feedback": "<feedback on layout, hierarchy, readability>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>"]
    }},
    "ats_compatibility": {{
      "score": <integer 0-100>,
      "feedback": "<feedback on ATS keyword density, formatting, searchability>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }}
  }},
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "critical_improvements": ["<action 1>", "<action 2>", "<action 3>"],
  "rewritten_section": {{
    "section_name": "<name of section rewritten>",
    "original_excerpt": "<original text, max 80 words>",
    "rewritten": "<professionally rewritten version demonstrating best practice>"
  }},
  "advisory_note": "<1-2 sentences of personalised DLC advisory guidance>"
}}

Rules:
- Be direct, specific, and honest. Avoid vague praise.
- Every improvement must be concrete and actionable.
- The rewrite must be meaningfully better — not just rephrased.
- Score 0-100: 75+ = strong, 50-74 = developing, <50 = needs work."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — used by Render to confirm the service is alive."""
    return jsonify({
        "status": "ok",
        "service": "DLC CV Analyser",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Accepts multipart/form-data: name, email, cv (file)
    Always returns structured JSON.
    """
    req_id    = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    tmp_path  = None

    # ── 1. Validate form fields ──────────────────────────────────────
    name  = (request.form.get("name")  or "").strip()
    email = (request.form.get("email") or "").strip()

    if not name:
        log.warning("[%s] rejected — missing name", req_id)
        return jsonify({"success": False, "error": "Name is required."}), 400

    if not email or "@" not in email:
        log.warning("[%s] rejected — invalid email: %s", req_id, email)
        return jsonify({"success": False, "error": "A valid email address is required."}), 400

    if "cv" not in request.files:
        log.warning("[%s] rejected — no file uploaded", req_id)
        return jsonify({"success": False, "error": "No CV file was uploaded."}), 400

    file = request.files["cv"]
    if not file or not file.filename:
        log.warning("[%s] rejected — empty filename", req_id)
        return jsonify({"success": False, "error": "No file was selected."}), 400

    if not _allowed_ext(file.filename):
        log.warning("[%s] rejected — bad extension: %s", req_id, file.filename)
        return jsonify({
            "success": False,
            "error": "Only PDF and Word documents (.pdf, .doc, .docx) are accepted.",
        }), 400

    ext = _get_ext(file.filename)

    log.info(
        "[%s] request received | name=%s | email=%s | file=%s | ts=%s",
        req_id, name, email, file.filename, timestamp,
    )

    try:
        # ── 2. Save to temp ──────────────────────────────────────────
        tmp_path = os.path.join(UPLOAD_DIR, f"{req_id}_{uuid.uuid4()}.{ext}")
        file.save(tmp_path)

        file_size = os.path.getsize(tmp_path)
        if file_size > MAX_FILE_BYTES:
            log.warning("[%s] rejected — file too large: %d bytes", req_id, file_size)
            return jsonify({
                "success": False,
                "error": "File exceeds the 10 MB size limit.",
            }), 400

        log.info("[%s] saved to temp: %s (%d bytes)", req_id, tmp_path, file_size)

        # ── 3. Extract text ──────────────────────────────────────────
        cv_text, method = extract_text(tmp_path, ext)
        log.info("[%s] extraction method=%s | chars=%d", req_id, method, len(cv_text))

        if not cv_text or len(cv_text) < 80:
            log.warning("[%s] extraction yielded insufficient text (%d chars)", req_id, len(cv_text))
            return jsonify({
                "success": False,
                "error": (
                    "Could not extract readable text from this file. "
                    "Please ensure the CV is not a scanned image and contains selectable text. "
                    "If it is a scanned document, please save it as a Word file and resubmit."
                ),
            }), 400

        # ── 4. Analyse with Claude ───────────────────────────────────
        log.info("[%s] sending to Claude for analysis", req_id)
        result = analyse_cv(cv_text, name)
        result["email"] = email
        result["request_id"] = req_id

        log.info(
            "[%s] SUCCESS | name=%s | score=%s | readiness=%s | method=%s",
            req_id,
            name,
            result.get("overall_score"),
            result.get("market_readiness"),
            method,
        )

        return jsonify({"success": True, "data": result})

    # ── Specific error handling ──────────────────────────────────────
    except json.JSONDecodeError as exc:
        log.error("[%s] JSON decode error from Claude: %s", req_id, exc)
        return jsonify({
            "success": False,
            "error": "Analysis produced an unexpected response format. Please try again.",
        }), 500

    except KeyError as exc:
        if "ANTHROPIC_API_KEY" in str(exc):
            log.error("[%s] ANTHROPIC_API_KEY environment variable is not set", req_id)
            return jsonify({
                "success": False,
                "error": "Server configuration error. Please contact support.",
            }), 500
        log.error("[%s] KeyError: %s", req_id, exc)
        return jsonify({"success": False, "error": "An internal error occurred."}), 500

    except Exception as exc:
        # Catch Anthropic API errors by string match (avoids import version issues)
        exc_type = type(exc).__name__
        exc_str  = str(exc)

        if "APIError" in exc_type or "APIStatusError" in exc_type:
            log.error("[%s] Anthropic API error: %s", req_id, exc_str)
            return jsonify({
                "success": False,
                "error": "AI service error. Please try again in a moment.",
            }), 502

        if "RateLimitError" in exc_type:
            log.error("[%s] Anthropic rate limit hit", req_id)
            return jsonify({
                "success": False,
                "error": "Service is temporarily busy. Please try again in 30 seconds.",
            }), 429

        if "AuthenticationError" in exc_type:
            log.error("[%s] Anthropic authentication error", req_id)
            return jsonify({
                "success": False,
                "error": "Server authentication error. Please contact support.",
            }), 500

        log.error(
            "[%s] Unhandled exception: %s — %s\n%s",
            req_id, exc_type, exc_str, traceback.format_exc(),
        )
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred. Please try again.",
        }), 500

    finally:
        # ── Always delete the temp file ──────────────────────────────
        if tmp_path:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    log.info("[%s] temp file deleted", req_id)
            except Exception as cleanup_exc:
                log.warning("[%s] could not delete temp file: %s", req_id, cleanup_exc)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
