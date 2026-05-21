"""
Direct Labour Consult — CV Diagnostic Backend  V15B
====================================================
Render.com deployment  |  Python 3.11+

Routes:
  GET  /              → 302 redirect to main DLC site
  GET  /health        → health check JSON
  GET  /version       → version verification endpoint
  POST /analyze       → single CV analysis (CV Diagnostic product)
  POST /analyze-batch → multi-CV batch analysis (Recruiter Console product)

Extraction pipeline (priority order):
  1. PyMuPDF  (fitz)          — fast digital PDF
  2. pdfminer                 — fallback digital PDF
  3. pytesseract / pdf2image  — OCR for scanned PDFs
  4. python-docx              — DOCX files
  5. binary grep              — legacy DOC files

Response envelope:
  SUCCESS: { "success": true,  "data": { ...analysis... } }
  ERROR:   { "success": false, "error": "human-readable message" }
"""

print("=== DLC BACKEND V15B ACTIVE ===")

# ── Standard library ──────────────────────────────────────────────────────
import gc
import io
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────
from anthropic import Anthropic
from flask import Flask, jsonify, redirect, request
from flask_cors import CORS

# ── Optional heavy libs (graceful degradation) ────────────────────────────
try:
    import fitz                          # PyMuPDF
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_OK = True
except ImportError:
    PDFMINER_OK = False

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except ImportError:
    DOCX_OK = False


# ══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("app")


# ══════════════════════════════════════════════════════════════════════════
#  APP + CORS
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "https://directlabourconsult.com",
    "https://www.directlabourconsult.com",
    "https://dlc-cv-backend.onrender.com",
    "https://*.pages.dev",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]}})


# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

MAX_FILE_BYTES      = 10 * 1024 * 1024   # 10 MB
MAX_OCR_PAGES       = 8
MIN_TEXT_CHARS      = 120
MAX_TEXT_CHARS      = 18_000
OCR_DPI             = 250
MAX_BATCH_FILES     = 25
# ── CRITICAL FIX: 2048 was truncating Claude's JSON response (~3500 tokens)
MAX_TOKENS_SINGLE   = 4096
MAX_TOKENS_BATCH    = 2048               # shorter per-candidate prompt

ALLOWED_EXTENSIONS  = {"pdf", "doc", "docx"}
ANTHROPIC_MODEL     = "claude-opus-4-5"

BACKEND_VERSION     = "DLC_BACKEND_V15B"


# ══════════════════════════════════════════════════════════════════════════
#  ANTHROPIC CLIENT
# ══════════════════════════════════════════════════════════════════════════

def _build_anthropic_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Add it in Render → Environment."
        )
    return Anthropic(api_key=api_key)


try:
    _client = _build_anthropic_client()
    log.info("Anthropic client initialised OK  model=%s", ANTHROPIC_MODEL)
except EnvironmentError as _e:
    _client = None
    log.error("Anthropic client NOT ready: %s", _e)


# ══════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _clean(raw: str) -> str:
    """Normalise and denoise raw extracted text."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKD", raw)
    text = text.replace("\f", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [l for l in text.split("\n") if len(l.strip()) >= 2 or l.strip() == ""]
    return "\n".join(lines).strip()


def _sufficient(text: str) -> bool:
    return bool(text) and len(re.sub(r"\s+", "", text)) >= MIN_TEXT_CHARS


def _try_pymupdf(data: bytes) -> str:
    if not PYMUPDF_OK:
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return _clean(text)
    except Exception as exc:
        log.warning("PyMuPDF failed: %s", exc)
        return ""


def _try_pdfminer(data: bytes) -> str:
    if not PDFMINER_OK:
        return ""
    try:
        text = pdfminer_extract(io.BytesIO(data)) or ""
        return _clean(text)
    except Exception as exc:
        log.warning("pdfminer failed: %s", exc)
        return ""


def _try_ocr(data: bytes) -> tuple:
    if not OCR_OK:
        raise RuntimeError("OCR not available.")
    log.info("OCR starting …")
    try:
        images = convert_from_bytes(
            data, dpi=OCR_DPI, first_page=1, last_page=MAX_OCR_PAGES
        )
    except Exception as exc:
        raise RuntimeError(f"pdf2image conversion failed: {exc}") from exc

    parts = []
    done = 0
    for img in images:
        try:
            parts.append(pytesseract.image_to_string(img, lang="eng",
                                                      config="--oem 1 --psm 3"))
            done += 1
        except Exception as exc:
            log.warning("Tesseract error page %d: %s", done + 1, exc)
        finally:
            img.close()
            gc.collect()

    return _clean("\n".join(parts)), done


def _try_docx(data: bytes) -> str:
    if not DOCX_OK:
        return ""
    try:
        doc = DocxDocument(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for tbl in doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return _clean("\n".join(parts))
    except Exception as exc:
        log.warning("python-docx failed: %s", exc)
        return ""


def _try_doc_legacy(data: bytes) -> str:
    try:
        text = "".join(
            chr(b) if 32 <= b < 127 else (" " if b in (10, 13) else "")
            for b in data
        )
        runs = re.findall(r"[A-Za-z][A-Za-z0-9 \-\.,@:/()\n]{3,}", text)
        return _clean(" ".join(runs))
    except Exception as exc:
        log.warning("DOC legacy parser failed: %s", exc)
        return ""


def extract_cv_text(filename: str, data: bytes) -> tuple:
    """Returns (extracted_text, method_used). Raises ValueError on failure."""
    ext = Path(filename).suffix.lower().lstrip(".")

    if ext == "pdf":
        text = _try_pymupdf(data)
        if _sufficient(text):
            log.info("Extraction: PyMuPDF  chars=%d", len(text))
            return text, "pymupdf"

        text = _try_pdfminer(data)
        if _sufficient(text):
            log.info("Extraction: pdfminer  chars=%d", len(text))
            return text, "pdfminer"

        log.info("Digital extraction insufficient (%d chars) — attempting OCR", len(text))
        try:
            text, pages = _try_ocr(data)
            if _sufficient(text):
                log.info("Extraction: OCR  pages=%d  chars=%d", pages, len(text))
                return text, "ocr ({} pages)".format(pages)
        except RuntimeError as exc:
            log.warning("OCR unavailable: %s", exc)

        raise ValueError(
            "We could not extract readable text from this PDF. "
            "Please save your CV directly from Microsoft Word or Google Docs "
            "(File → Save as PDF) and try again."
        )

    if ext == "docx":
        text = _try_docx(data)
        if _sufficient(text):
            log.info("Extraction: python-docx  chars=%d", len(text))
            return text, "python-docx"
        raise ValueError(
            "Could not read your DOCX file. "
            "Please re-save it in Microsoft Word and try again."
        )

    if ext == "doc":
        text = _try_doc_legacy(data)
        if _sufficient(text):
            log.info("Extraction: doc-legacy  chars=%d", len(text))
            return text, "doc-legacy"
        raise ValueError(
            "Could not read your DOC file. "
            "Please open it in Word and save as DOCX or PDF, then try again."
        )

    raise ValueError("Unsupported file type: .{}".format(ext))


# ══════════════════════════════════════════════════════════════════════════
#  INDUSTRIAL-GRADE JSON PARSER  (V15B — safe_parse_json_v2)
#  Never lets a raw parse error reach the frontend.
# ══════════════════════════════════════════════════════════════════════════

def _strip_control_chars(s: str) -> str:
    """Remove control characters that break JSON parsers."""
    # Keep tab (9), newline (10), carriage return (13) — everything else below 32 → space
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)


def _strip_markdown_fences(s: str) -> str:
    """Strip ```json ... ``` code fences."""
    s = re.sub(r"^```(?:json)?\s*\n?", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> str:
    """
    Find the first complete top-level {...} block using bracket counting.
    Handles cases where Claude prepends/appends non-JSON text.
    """
    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(s[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]

    # Bracket counting didn't close — return from first { to end
    return s[start:]


def _repair_truncated_json(s: str) -> str:
    """
    Attempt to close a truncated JSON object by:
    1. Removing any trailing partial string/value
    2. Closing open arrays with ]
    3. Closing open objects with }
    """
    # Remove trailing partial token (incomplete string, number, keyword)
    s = s.rstrip()

    # Remove trailing comma
    s = re.sub(r",\s*$", "", s)

    # Count unclosed brackets
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1

    # Close open string if needed
    if in_string:
        s += '"'

    # Close open arrays first, then objects
    s += "]" * max(0, depth_arr)
    s += "}" * max(0, depth_obj)

    return s


def _build_fallback_result(raw: str) -> dict:
    """
    Last-resort: extract whatever fields we can via regex, return partial result.
    This ensures the frontend always gets a dict, never a raw exception.
    """
    def _grab(pattern: str, default):
        m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return m.group(1).strip().strip('"')
        return default

    score = _grab(r'"overall_score"\s*:\s*(\d+)', 0)
    name  = _grab(r'"candidate_name"\s*:\s*("(?:[^"\\]|\\.)*")', "Unknown")
    summary = _grab(r'"executive_summary"\s*:\s*("(?:[^"\\]|\\.)*")',
                    "Analysis completed. Full report unavailable — please retry.")

    return {
        "candidate_name":       name,
        "overall_score":        int(score) if str(score).isdigit() else 0,
        "market_readiness":     "Developing",
        "executive_summary":    summary,
        "top_strengths":        ["Analysis partially available — please retry for full report."],
        "critical_improvements":["Please retry for full diagnostic."],
        "sections": {
            "first_impression":   {"score": 0, "feedback": "Partial result — please retry.",
                                   "strengths": [], "improvements": []},
            "value_signal":       {"score": 0, "feedback": "Partial result — please retry.",
                                   "strengths": [], "improvements": []},
            "evidence_of_impact": {"score": 0, "feedback": "Partial result — please retry.",
                                   "strengths": [], "improvements": []},
            "role_alignment":     {"score": 0, "feedback": "Partial result — please retry.",
                                   "strengths": [], "improvements": []},
            "ats_compatibility":  {"score": 0, "feedback": "Partial result — please retry.",
                                   "strengths": [], "improvements": []},
        },
        "rewritten_section": {
            "section_name":     "Note",
            "original_excerpt": "",
            "rewritten":        "Full rewrite unavailable — please retry for complete report.",
        },
        "advisory_note": "Analysis was incomplete due to a response length issue. Please retry — your submission is valid.",
        "_partial": True,
    }


def safe_parse_json_v2(raw: str, context: str = "") -> dict:
    """
    Industrial-grade JSON parser with 4 recovery layers.
    NEVER raises — always returns a dict.
    """
    if not raw or not raw.strip():
        log.error("safe_parse_json_v2[%s]: empty response from Claude", context)
        return _build_fallback_result("")

    # Layer 0: strip markdown fences + control chars
    cleaned = _strip_markdown_fences(raw)
    cleaned = _strip_control_chars(cleaned)

    # Layer 1: direct parse (happy path)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Layer 2: extract first complete {...} object
    try:
        candidate = _extract_json_object(cleaned)
        return json.loads(candidate)
    except (json.JSONDecodeError, Exception):
        pass

    # Layer 3: repair truncated JSON (max_tokens cut-off)
    try:
        candidate = _extract_json_object(cleaned)
        repaired  = _repair_truncated_json(candidate)
        result    = json.loads(repaired)
        log.warning("safe_parse_json_v2[%s]: repaired truncated JSON", context)
        result["_repaired"] = True
        return result
    except (json.JSONDecodeError, Exception):
        pass

    # Layer 4: regex field extraction fallback
    log.error(
        "safe_parse_json_v2[%s]: all parse layers failed — using fallback. "
        "Raw first 400 chars: %s",
        context, cleaned[:400]
    )
    return _build_fallback_result(cleaned)


# ══════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════

_PROMPT_SINGLE = """\
You are a senior HR advisory analyst at Direct Labour Consult (DLC), \
Botswana's premier HR consultancy (established 2018). \
Evaluate the CV below using DLC's five-layer diagnostic framework. \
Return ONLY a raw JSON object — no markdown, no code fences, no explanation text.

Required JSON structure (return ALL fields, be concise to stay within token limit):
{{
  "candidate_name":       "string — inferred from CV",
  "overall_score":        integer 0-100,
  "market_readiness":     "Excellent | Strong | Developing | Needs Improvement",
  "executive_summary":    "2-3 sentences. Specific, professional, honest.",
  "top_strengths":        ["string","string","string"],
  "critical_improvements":["string","string","string"],
  "sections": {{
    "first_impression":   {{"score":int,"feedback":"string max 80 words","strengths":["str"],"improvements":["str"]}},
    "value_signal":       {{"score":int,"feedback":"string max 80 words","strengths":["str"],"improvements":["str"]}},
    "evidence_of_impact": {{"score":int,"feedback":"string max 80 words","strengths":["str"],"improvements":["str"]}},
    "role_alignment":     {{"score":int,"feedback":"string max 80 words","strengths":["str"],"improvements":["str"]}},
    "ats_compatibility":  {{"score":int,"feedback":"string max 80 words","strengths":["str"],"improvements":["str"]}}
  }},
  "rewritten_section": {{
    "section_name":     "e.g. Professional Summary",
    "original_excerpt": "verbatim excerpt 40 words max",
    "rewritten":        "improved version 60 words max"
  }},
  "advisory_note": "1-2 sentences. Single most important action."
}}

Scoring: 85-100 Excellent | 70-84 Strong | 50-69 Developing | 0-49 Needs Improvement
Keep all feedback concise. Every observation must be traceable to actual CV content.

CV CONTENT:
---
{cv_text}
---"""


_PROMPT_BATCH_CANDIDATE = """\
You are a senior HR analyst at Direct Labour Consult (DLC). \
Evaluate this candidate CV for a recruiter making a hiring decision. \
Return ONLY a raw JSON object — no markdown, no code fences, no explanation.

Required JSON:
{{
  "candidate_name":    "inferred from CV or use filename if not found",
  "overall_score":     integer 0-100,
  "recommendation":    "Strong Hire | Consider | Risk",
  "recommendation_reason": "1 sentence explaining the recommendation",
  "top_strengths":     ["str","str","str"],
  "critical_gaps":     ["str","str"],
  "behavioural_risk":  "None identified | Low | Medium | High",
  "behavioural_risk_note": "1 sentence or empty string",
  "advisory_note":     "1 sentence for the hiring manager"
}}

Scoring: 80-100 = Strong Hire | 60-79 = Consider | 0-59 = Risk

CV CONTENT:
---
{cv_text}
---"""


# ══════════════════════════════════════════════════════════════════════════
#  CLAUDE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def analyse_with_claude(cv_text: str, name_hint: str = "") -> dict:
    """Single CV analysis for /analyze endpoint."""
    if _client is None:
        raise RuntimeError(
            "Analysis service is not configured (missing API key). "
            "Contact DLC support."
        )

    if len(cv_text) > MAX_TEXT_CHARS:
        log.info("CV text truncated %d → %d chars", len(cv_text), MAX_TEXT_CHARS)
        cv_text = cv_text[:MAX_TEXT_CHARS] + "\n\n[Content truncated for analysis]"

    log.info("Single CV: %d chars", len(cv_text))

    msg = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS_SINGLE,   # FIXED: was 2048, now 4096
        messages=[{
            "role":    "user",
            "content": _PROMPT_SINGLE.format(cv_text=cv_text),
        }],
    )

    raw = msg.content[0].text.strip()
    log.info("Claude response length: %d chars, stop_reason: %s",
             len(raw), msg.stop_reason)

    if msg.stop_reason == "max_tokens":
        log.warning("Claude hit max_tokens limit — response may be truncated")

    result = safe_parse_json_v2(raw, context="single_cv")

    if not result.get("candidate_name") and name_hint:
        result["candidate_name"] = name_hint

    return result


def analyse_batch_candidate(cv_text: str, filename: str) -> dict:
    """Single candidate analysis for /analyze-batch endpoint."""
    if _client is None:
        raise RuntimeError("Analysis service not configured.")

    if len(cv_text) > 10_000:
        cv_text = cv_text[:10_000] + "\n\n[Truncated]"

    msg = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS_BATCH,
        messages=[{
            "role":    "user",
            "content": _PROMPT_BATCH_CANDIDATE.format(cv_text=cv_text),
        }],
    )

    raw = msg.content[0].text.strip()
    result = safe_parse_json_v2(raw, context="batch_{}".format(filename))

    if not result.get("candidate_name"):
        result["candidate_name"] = Path(filename).stem.replace("_", " ").replace("-", " ")

    return result


# ══════════════════════════════════════════════════════════════════════════
#  UPLOAD HELPERS  (multi-field detection)
# ══════════════════════════════════════════════════════════════════════════

def _get_batch_files(req):
    """
    Detect uploaded files across ALL possible field names.
    Frontend may send: files, files[], cvs, cvs[], cv_files, cv_files[]
    """
    log.info("FILES RECEIVED: %s", list(req.files.keys()))
    log.info("FORM FIELDS: %s",    list(req.form.keys()))

    for field in ("files", "files[]", "cvs", "cvs[]", "cv_files", "cv_files[]"):
        batch = req.files.getlist(field)
        if batch and any(f.filename for f in batch):
            log.info("Batch files found under field=%r  count=%d", field, len(batch))
            return [f for f in batch if f.filename]

    # Last resort: grab everything from req.files
    all_files = []
    for key in req.files.keys():
        all_files.extend(req.files.getlist(key))
    all_files = [f for f in all_files if f.filename]

    if all_files:
        log.info("Batch files found via fallback scan  count=%d", len(all_files))

    return all_files


# ══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return redirect("https://directlabourconsult.com", code=302)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status":  "ok",
        "version": BACKEND_VERSION,
        "model":   ANTHROPIC_MODEL,
        "capabilities": {
            "pymupdf":  PYMUPDF_OK,
            "pdfminer": PDFMINER_OK,
            "ocr":      OCR_OK,
            "docx":     DOCX_OK,
        },
    }), 200


@app.route("/version", methods=["GET"])
def version():
    """Version verification endpoint — confirms correct code is deployed."""
    return jsonify({
        "version":          BACKEND_VERSION,
        "parser":           "safe_parse_json_v2",
        "upload_detection": "multi-field",
        "max_tokens_single": MAX_TOKENS_SINGLE,
        "max_tokens_batch":  MAX_TOKENS_BATCH,
        "status":           "active",
        "model":            ANTHROPIC_MODEL,
    }), 200


# ── /analyze  (single CV — CV Diagnostic product) ─────────────────────────

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return "", 204

    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def err(msg: str, status: int = 400):
        log.warning("ANALYZE ERROR [%s]: %s", ts, msg)
        return jsonify({"success": False, "error": msg}), status

    # 1. Form fields
    name  = (request.form.get("name")  or "").strip()
    email = (request.form.get("email") or "").strip()

    if not name:
        return err("Name is required.")
    if not email or "@" not in email:
        return err("A valid email address is required.")

    # 2. File
    cv_file = request.files.get("cv") or request.files.get("cv_original")
    if not cv_file or not cv_file.filename:
        return err("No CV file received. Please select and upload your CV.")

    filename = cv_file.filename
    ext      = Path(filename).suffix.lower().lstrip(".")

    if ext not in ALLOWED_EXTENSIONS:
        return err(
            "File type '.{}' is not supported. "
            "Please upload a PDF, DOCX, or DOC file.".format(ext)
        )

    try:
        data = cv_file.read()
    except Exception as exc:
        log.error("File read error: %s", exc)
        return err("Could not read the uploaded file. Please try again.", 500)

    if len(data) > MAX_FILE_BYTES:
        return err(
            "File size ({} MB) exceeds the 10 MB limit.".format(len(data) // 1024 // 1024)
        )
    if len(data) < 500:
        return err("The uploaded file appears to be empty or corrupt.")

    log.info("ANALYZE START  name=%r  file=%r  size=%d bytes", name, filename, len(data))

    # 3. Text extraction
    method  = "unknown"
    cv_text = ""

    client_text = (request.form.get("extracted_text") or "").strip()
    if client_text and _sufficient(client_text):
        cv_text = _clean(client_text)
        method  = "client-preextracted"
        log.info("Using client pre-extracted text  chars=%d", len(cv_text))
    else:
        try:
            cv_text, method = extract_cv_text(filename, data)
        except ValueError as exc:
            log.warning("Extraction failed: %s", exc)
            return err(str(exc), 422)
        except Exception as exc:
            log.error("Unexpected extraction error: %s", exc, exc_info=True)
            return err(
                "An unexpected error occurred while reading your CV. "
                "Please try again or contact DLC support.", 500
            )

    # 4. Claude analysis
    try:
        result = analyse_with_claude(cv_text, name_hint=name)
    except RuntimeError as exc:
        log.error("Claude analysis error: %s", exc)
        return err(str(exc), 503)
    except Exception as exc:
        log.error("Unexpected Claude error: %s", exc, exc_info=True)
        return err(
            "Analysis service encountered an unexpected error. "
            "Your payment is still valid — please try again in a moment.", 503
        )

    # 5. Metadata
    elapsed = round(time.time() - t0, 2)
    result["_meta"] = {
        "extraction_method": method,
        "chars_analysed":    len(cv_text),
        "ocr_used":          "ocr" in method,
        "processing_time_s": elapsed,
        "timestamp_utc":     ts,
        "backend_version":   BACKEND_VERSION,
    }

    log.info(
        "ANALYZE DONE  name=%r  score=%s  method=%s  time=%.2fs",
        name, result.get("overall_score", "?"), method, elapsed,
    )

    return jsonify({"success": True, "data": result}), 200


# ── /analyze-batch  (Recruiter Console product) ───────────────────────────

@app.route("/analyze-batch", methods=["POST", "OPTIONS"])
def analyze_batch():
    if request.method == "OPTIONS":
        return "", 204

    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def err(msg: str, status: int = 400):
        log.warning("BATCH ERROR [%s]: %s", ts, msg)
        return jsonify({"success": False, "error": msg}), status

    # 1. Detect uploaded files across all possible field names
    uploaded = _get_batch_files(request)

    if not uploaded:
        log.error(
            "BATCH: No files detected. "
            "Files dict keys: %s | Content-Type: %s",
            list(request.files.keys()),
            request.content_type,
        )
        return err(
            "No CV files received. Please ensure files are attached before submitting. "
            "Accepted formats: PDF, DOCX, DOC."
        )

    if len(uploaded) > MAX_BATCH_FILES:
        return err(
            "Maximum {} CVs per batch. You submitted {}.".format(
                MAX_BATCH_FILES, len(uploaded)
            )
        )

    # Filter to allowed extensions
    valid_files = []
    skipped     = []
    for f in uploaded:
        ext = Path(f.filename).suffix.lower().lstrip(".")
        if ext in ALLOWED_EXTENSIONS:
            valid_files.append(f)
        else:
            skipped.append(f.filename)

    if skipped:
        log.warning("Batch: skipping unsupported files: %s", skipped)

    if not valid_files:
        return err(
            "None of the uploaded files are in a supported format. "
            "Please upload PDF, DOCX, or DOC files."
        )

    log.info("BATCH START  files=%d  valid=%d  ts=%s", len(uploaded), len(valid_files), ts)

    # 2. Process each CV
    results = []
    errors  = []

    for i, cv_file in enumerate(valid_files):
        filename = cv_file.filename
        log.info("Batch processing [%d/%d]: %s", i + 1, len(valid_files), filename)

        try:
            data = cv_file.read()
        except Exception as exc:
            log.error("Batch file read error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": "Could not read file."})
            continue

        if len(data) > MAX_FILE_BYTES:
            errors.append({"filename": filename, "error": "File too large (max 10 MB)."})
            continue

        if len(data) < 200:
            errors.append({"filename": filename, "error": "File appears empty."})
            continue

        # Extract text
        try:
            cv_text, method = extract_cv_text(filename, data)
        except ValueError as exc:
            log.warning("Batch extraction failed [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": str(exc)})
            continue
        except Exception as exc:
            log.error("Batch extraction error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": "Text extraction failed."})
            continue

        # Claude analysis
        try:
            analysis = analyse_batch_candidate(cv_text, filename)
        except RuntimeError as exc:
            log.error("Batch Claude error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": str(exc)})
            continue
        except Exception as exc:
            log.error("Batch unexpected Claude error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": "Analysis failed."})
            continue

        analysis["filename"]         = filename
        analysis["extraction_method"] = method
        results.append(analysis)

    if not results:
        return jsonify({
            "success": False,
            "error":   "Analysis failed for all submitted CVs.",
            "errors":  errors,
        }), 500

    # 3. Rank by overall_score descending
    results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)

    # Assign rank numbers
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    elapsed = round(time.time() - t0, 2)

    log.info(
        "BATCH DONE  processed=%d  errors=%d  top_score=%s  time=%.2fs",
        len(results), len(errors),
        results[0].get("overall_score", "?") if results else "N/A",
        elapsed,
    )

    return jsonify({
        "success": True,
        "data": {
            "results":          results,
            "total_processed":  len(results),
            "total_submitted":  len(valid_files),
            "errors":           errors,
            "processing_time_s": elapsed,
            "timestamp_utc":    ts,
            "backend_version":  BACKEND_VERSION,
        },
    }), 200


# ══════════════════════════════════════════════════════════════════════════
#  GLOBAL ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(413)
def request_entity_too_large(_e):
    return jsonify({"success": False, "error": "File too large. Maximum size is 10 MB."}), 413


@app.errorhandler(500)
def internal_error(exc):
    log.error("Unhandled 500: %s", exc, exc_info=True)
    return jsonify({
        "success": False,
        "error": "An internal server error occurred. Please try again.",
    }), 500


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting DLC backend %s on port %d", BACKEND_VERSION, port)
    app.run(host="0.0.0.0", port=port, debug=False)
