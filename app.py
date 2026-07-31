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
import os as _os

# ── DEFENSIVE FIX: Clear proxy env vars before ANY import that touches httpx.
# Render may set HTTP_PROXY / HTTPS_PROXY on its infrastructure.
# If present, the anthropic SDK reads them and passes proxies= to httpx.Client()
# which was removed in httpx 0.28.0 → TypeError crash.
# We clear them here so the SDK never attempts proxy injection.
for _pv in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
    _os.environ.pop(_pv, None)
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
MAX_TOKENS_BATCH    = 3072               # upgraded prompt requires more output space

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
You are a Senior Assessment Consultant at Direct Labour Consult (DLC), \
Botswana's premier Industrial Psychology-Informed Executive Assessment practice. \
Conduct a rigorous, evidence-based candidate evaluation for the recruitment mandate below. \
Your assessment must reflect the depth and analytical standard of a Senior Industrial/Organisational \
Psychology practitioner — not a generic screener. \
Return ONLY a valid JSON object — no markdown, no code fences, no preamble.

RECRUITMENT MANDATE:
  Position:        {job_title}
  Department:      {job_dept}
  Min. Experience: {job_exp} years
  Education:       {job_edu}
  Key Skills:      {job_skills}
  Job Description: {job_desc}
  Disqualifiers:   {job_disq}

ASSESSMENT FRAMEWORK — apply all five dimensions:
  1. COMPETENCY EVIDENCE     Distinguish demonstrated competencies from claimed ones. \
Cite specific employers, roles, tenures, and achievements from the CV.
  2. CAREER TRAJECTORY       Analyse progression patterns: ascending, lateral, or fragmented. \
Note scope of accountability growth and any tenure concerns.
  3. PERSON-JOB FIT          Calibrate directly against the role demands above. \
Flag alignment factors and material gaps specific to this mandate.
  4. BEHAVIOURAL INDICATORS  Infer motivational orientation, resilience signals, and \
cultural fit from CV structure, tone, content patterns, and career choices.
  5. OCCUPATIONAL MATURITY   Assess functional readiness relative to the seniority level required \
for this position.

REQUIRED JSON (all fields mandatory):
{{
  "candidate_name":      "Full name inferred from CV. Use filename stem if not found.",
  "overall_score":       integer 0-100,
  "recommendation":      "Hire | Consider | Not Aligned",

  "executive_summary":   "3 sentences. Open with the single most significant finding about this candidate's suitability. Reference specific CV evidence — employer names, role titles, tenures, or quantified achievements. Close with a decisive and consultative assessment stance. Authoritative tone — no generic filler.",

  "job_fit_note":        "1-2 sentences. Specific alignment or gap between this candidate's demonstrated experience and the role requirements stated above. Name the role and evidence.",

  "key_strengths": [
    "Evidence-anchored: cite specific employer / role / achievement from CV, then state what this indicates about capability for this role",
    "Evidence-anchored: cite specific evidence — then state what this indicates",
    "Evidence-anchored: cite specific evidence — then state what this indicates"
  ],
  "key_concerns": [
    "Name the specific gap or risk — cite absence of evidence or a problematic pattern observed in the CV",
    "Name the specific gap or risk — cite evidence"
  ],

  "behavioural_risk":   "Low | Medium | High",
  "behavioural_notes":  "Analytical observation on behavioural or motivational risk indicators. Reference tenure patterns, unexplained gaps, career move motivations, or stated interests as signals. Empty string if risk is Low.",

  "sections": {{
    "first_impression":   {{"score": integer 0-100, "rationale": "What does CV structure, formatting, and professional framing signal about this candidate's professional identity and self-presentation standard?"}},
    "evidence_of_impact": {{"score": integer 0-100, "rationale": "Quality and specificity of achievement evidence — are outcomes quantified, attributed, and credible? Or are responsibilities listed without impact?"}},
    "role_alignment":     {{"score": integer 0-100, "rationale": "Degree of match between demonstrated experience and the specific functional demands of this role."}},
    "ats_compatibility":  {{"score": integer 0-100, "rationale": "Keyword density, role-relevant terminology, and structural scannability for this type of position."}}
  }},

  "occupational_profile": {{
    "leadership_readiness":  "Emerging | Developing | Established | Advanced",
    "operational_maturity":  "Graduate | Junior | Mid-level | Senior | Executive",
    "strategic_thinking":    "Absent | Limited | Present | Strong",
    "stakeholder_exposure":  "Internal only | Cross-functional | External / Board-level"
  }},

  "competency_assessment": {{
    "technical_competence":      "Below | Meets | Exceeds — 1-sentence evidence statement citing specific CV content",
    "analytical_reasoning":      "Below | Meets | Exceeds — 1-sentence evidence statement",
    "communication_proficiency": "Below | Meets | Exceeds — 1-sentence evidence statement",
    "leadership_and_influence":  "Below | Meets | Exceeds — 1-sentence evidence statement"
  }},

  "career_trajectory":   "1-2 sentences. Is the career pattern ascending, lateral, or fragmented? What does the scope of accountability growth — or absence of it — indicate about this candidate's ceiling and professional drive?",

  "recruiter_guidance":  "Recommended for Interview | Recommended for Shortlist | Development Candidate | Not Aligned Currently",
  "years_experience":    "estimated range e.g. 5-7 years",
  "education_alignment": "Exceeds requirements | Meets requirements | Below requirements | Unable to determine",
  "advisory_note":       "1-2 sentences. The single most important nuanced observation for the hiring manager — something that standard CV screening would miss. May relate to a hidden strength, a structural risk, or a context-specific consideration."
}}

SCORING FRAMEWORK:
  80-100  Strong, well-evidenced profile — recommend Hire
  60-79   Viable candidate with reservations — recommend Consider
  0-59    Significant gaps against mandate requirements — recommend Not Aligned

QUALITY STANDARDS:
- Every observation must cite specific CV content: employer name, role title, tenure, metric, or named achievement
- No generic filler (avoid: "demonstrates strong experience", "has a proven track record", "brings valuable skills")
- Tone: authoritative, consultative, professionally direct — not robotic, not harsh
- Clearly distinguish demonstrated evidence from reasonable inferences
- If job description is blank, calibrate against established professional standards for this role type
- If candidate name is unclear, derive it from the filename

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


def analyse_batch_candidate(cv_text: str, filename: str, job_params: dict = None) -> dict:
    """
    Single candidate analysis for /analyze-batch endpoint.
    Returns a dict with ALL keys the recruiter-upload.html frontend expects.
    job_params: dict with keys job_title, job_dept, job_exp, job_edu,
                job_skills, job_desc, job_disq — from request.form
    """
    if _client is None:
        raise RuntimeError("Analysis service not configured.")

    if len(cv_text) > 10_000:
        cv_text = cv_text[:10_000] + "\n\n[Truncated]"

    # Build prompt with job context (fall back to safe defaults if not provided)
    p = job_params or {}
    prompt = _PROMPT_BATCH_CANDIDATE.format(
        cv_text   = cv_text,
        job_title = p.get("job_title", "Not specified"),
        job_dept  = p.get("job_dept",  "Not specified"),
        job_exp   = p.get("job_exp",   "Not specified"),
        job_edu   = p.get("job_edu",   "Not specified"),
        job_skills= p.get("job_skills","Not specified"),
        job_desc  = p.get("job_desc",  ""),
        job_disq  = p.get("job_disq",  "None stated"),
    )

    msg = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS_BATCH,
        messages=[{
            "role":    "user",
            "content": prompt,
        }],
    )

    raw = msg.content[0].text.strip()
    result = safe_parse_json_v2(raw, context="batch_{}".format(filename))

    # ── Guarantee candidate_name is always set ────────────────────────────
    if not result.get("candidate_name"):
        result["candidate_name"] = (
            Path(filename).stem.replace("_", " ").replace("-", " ").title()
        )

    # ── Normalise recommendation to values recClass() recognises ──────────
    rec = (result.get("recommendation") or "").strip()
    if rec.lower() in ("hire", "strong hire"):
        result["recommendation"] = "Hire"
    elif rec.lower() in ("consider",):
        result["recommendation"] = "Consider"
    else:
        result["recommendation"] = "Not Aligned"

    # ── Normalise behavioural_risk to values riskClass() recognises ───────
    br = (result.get("behavioural_risk") or "Low").strip()
    if "high" in br.lower():
        result["behavioural_risk"] = "High"
    elif "med" in br.lower():
        result["behavioural_risk"] = "Medium"
    else:
        result["behavioural_risk"] = "Low"

    # ── Guarantee sections block exists with score keys ───────────────────
    sects = result.get("sections") or {}
    for sk in ("first_impression", "evidence_of_impact", "role_alignment", "ats_compatibility"):
        if sk not in sects or not isinstance(sects.get(sk), dict):
            sects[sk] = {"score": 0}
        elif "score" not in sects[sk]:
            sects[sk]["score"] = 0
    result["sections"] = sects

    # ── Guarantee frontend list keys ──────────────────────────────────────
    if not result.get("key_strengths"):
        result["key_strengths"] = result.pop("top_strengths", []) or []
    if not result.get("key_concerns"):
        result["key_concerns"] = result.pop("critical_gaps", []) or []
    if not result.get("behavioural_notes"):
        result["behavioural_notes"] = result.pop("behavioural_risk_note", "") or ""
    if not result.get("executive_summary"):
        result["executive_summary"] = result.pop("recommendation_reason", "") or ""

    # ── _filename — frontend uses c._filename for display ─────────────────
    result["_filename"] = filename

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

    # ── Read job parameters sent by the frontend ──────────────────────────────
    job_params = {
        "job_title":  (request.form.get("job_title")  or "").strip(),
        "job_dept":   (request.form.get("job_dept")   or "").strip(),
        "job_exp":    (request.form.get("job_exp")    or "").strip(),
        "job_edu":    (request.form.get("job_edu")    or "").strip(),
        "job_skills": (request.form.get("job_skills") or "").strip(),
        "job_desc":   (request.form.get("job_desc")   or "").strip(),
        "job_disq":   (request.form.get("job_disq")   or "").strip(),
    }
    log.info(
        "BATCH JOB PARAMS  title=%r  dept=%r  exp=%r  edu=%r",
        job_params["job_title"], job_params["job_dept"],
        job_params["job_exp"],   job_params["job_edu"],
    )

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
            analysis = analyse_batch_candidate(cv_text, filename, job_params)
        except RuntimeError as exc:
            log.error("Batch Claude error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": str(exc)})
            continue
        except Exception as exc:
            log.error("Batch unexpected Claude error [%s]: %s", filename, exc)
            errors.append({"filename": filename, "error": "Analysis failed."})
            continue

        analysis["_filename"]        = filename   # already set but ensure
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
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    # 4. Build summary object  (frontend reads data.summary.hire / .consider / .average_score)
    hire_count    = sum(1 for r in results if r.get("recommendation") == "Hire")
    consider_count = sum(1 for r in results if r.get("recommendation") == "Consider")
    not_aligned   = sum(1 for r in results if r.get("recommendation") == "Not Aligned")
    scores        = [r.get("overall_score", 0) for r in results if isinstance(r.get("overall_score"), (int, float))]
    avg_score     = round(sum(scores) / len(scores)) if scores else 0
    top_score     = max(scores) if scores else 0

    if avg_score >= 72:
        pool_quality = "Strong candidate pool"
    elif avg_score >= 55:
        pool_quality = "Mixed candidate pool"
    else:
        pool_quality = "Thin candidate pool — consider widening the search"

    summary = {
        "hire":          hire_count,
        "consider":      consider_count,
        "not_aligned":   not_aligned,
        "average_score": avg_score,
        "top_score":     top_score,
        "pool_quality":  pool_quality,
    }

    # 5. Pool-level executive report (computed — no extra Claude call)
    top = results[0] if results else {}
    executive_report = (
        "{} candidate{} assessed. {} recommended for hire, {} for consideration. "
        "Average pool score: {}/100. Top candidate: {} ({}). {}".format(
            len(results),
            "s" if len(results) != 1 else "",
            hire_count,
            consider_count,
            avg_score,
            top.get("candidate_name", "N/A"),
            top.get("overall_score", "N/A"),
            pool_quality + ".",
        )
    )

    elapsed = round(time.time() - t0, 2)

    log.info(
        "BATCH DONE  processed=%d  hire=%d  consider=%d  avg=%d  time=%.2fs",
        len(results), hire_count, consider_count, avg_score, elapsed,
    )

    # ── FLAT response — frontend reads data.ranked_candidates directly ──────
    # DO NOT nest under "data:" — recruiter-upload.html does:
    #   var data = await res.json();
    #   renderResults(data, params);
    # and renderResults reads data.ranked_candidates / data.total_candidates.
    return jsonify({
        "success":           True,
        "ranked_candidates": results,
        "total_candidates":  len(results),
        "summary":           summary,
        "executive_report":  executive_report,
        "job_title":         job_params.get("job_title", ""),
        "total_submitted":   len(valid_files),
        "errors":            errors,
        "processing_time_s": elapsed,
        "timestamp_utc":     ts,
        "backend_version":   BACKEND_VERSION,
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
@app.route('/optimize-application', methods=['POST'])
def optimize_application():
    try:
        file = request.files.get('file')
        job_description = request.form.get('job_description')

        if not file or not job_description:
            return jsonify({
                "success": False,
                "error": "Missing CV or job description"
            })

        # Simulated ATS processing (replace later with real logic)
        optimized_cv = {
            "summary": "Results-driven professional aligned to the job requirements, with strong experience in supplier management, cost control, and performance optimisation.",
            "experience": [
                "Managed supplier selection and evaluation based on cost, quality, and delivery timelines.",
                "Maintained strong supplier relationships and monitored performance metrics.",
                "Implemented procurement strategies that improved efficiency and reduced costs."
            ]
        }

        return jsonify({
            "success": True,
            "ats_score": 82,
            "job_match_score": 76,
            "optimized_cv": optimized_cv
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })
import os
import json
import logging
import tempfile
import threading
import time
import re
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import requests

# ══════════════════════════════════════════════════════
# DEPLOYMENT VERIFICATION — visible in Render logs
print("=" * 54)
print("=== DLC BACKEND V14C ACTIVE                      ===")
print("=== safe_parse_json_v2  |  multi-field upload     ===")
print("=== Build: 2026-05-21                             ===")
print("=" * 54, flush=True)
# ══════════════════════════════════════════════════════

# ── optional extraction libs ──────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── app setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

BACKEND_URL = os.environ.get("BACKEND_URL", "https://dlc-cv-backend.onrender.com")
ADMIN_KEY   = os.environ.get("ADMIN_KEY", "dlc-admin-2026")

# ── in-memory submission log (last 200 entries, thread-safe) ──────────────────
_log_lock    = threading.Lock()
_submissions = []

def log_submission(entry: dict):
    with _log_lock:
        _submissions.insert(0, entry)
        if len(_submissions) > 200:
            _submissions.pop()

# ── admin key check ───────────────────────────────────────────────────────────
def check_admin(req) -> bool:
    key = req.headers.get("X-Admin-Key") or req.args.get("key") or req.form.get("admin_key")
    return bool(key and key == ADMIN_KEY)

# ── keep-warm self-ping (prevents Render free-tier sleep) ─────────────────────
def _keep_warm():
    while True:
        try:
            time.sleep(13 * 60)
            r = requests.get(f"{BACKEND_URL}/health", timeout=10)
            logger.info(f"[keep-warm] ping → {r.status_code}")
        except Exception as e:
            logger.warning(f"[keep-warm] ping failed: {e}")

threading.Thread(target=_keep_warm, daemon=True).start()
logger.info("[keep-warm] self-ping thread started")

# ── file list helper — accepts any field name the frontend might send ──────────
def get_uploaded_files():
    """
    Accept files under any of: files, files[], cvs, cvs[]
    Returns the first non-empty list found, or [].
    """
    for field in ("files", "files[]", "cvs", "cvs[]"):
        found = request.files.getlist(field)
        if found and any(f.filename for f in found):
            logger.info(f"[upload] found {len(found)} file(s) under field='{field}'")
            return found
    # Debug: log what fields actually arrived
    logger.warning(f"[upload] no files found. form keys={list(request.form.keys())} "
                   f"files keys={list(request.files.keys())}")
    return []

# ── text extraction ───────────────────────────────────────────────────────────
def extract_text(file_storage) -> tuple[str, str]:
    filename = (file_storage.filename or "").lower()
    ext = os.path.splitext(filename)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    text, method = "", "unknown"
    try:
        if filename.endswith(".pdf") and HAS_PDF:
            with pdfplumber.open(tmp_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            method = "pdfplumber"
        elif filename.endswith((".docx", ".doc")) and HAS_DOCX:
            doc = DocxDocument(tmp_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            method = "python-docx"
        else:
            with open(tmp_path, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
            method = "raw-decode"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    logger.info(f"[extract] method={method} chars={len(text)} file={filename}")
    return text.strip(), method

# ── AI system prompts ─────────────────────────────────────────────────────────
SINGLE_SYSTEM = """You are a Chartered HR Practitioner and Occupational Psychologist at Direct Labour Consult, a professional HR advisory firm. You conduct structured CV assessments using industrial psychology principles and evidence-based career development frameworks.

Your assessment must be constructive, professional, and empowering. The tone reflects a senior career consultant providing developmental feedback. Never use language that demeans the candidate.

PROHIBITED WORDS: suffers, corrupted, failure, rejected, unreadable, terrible, poor, bad, wrong, weak, broken, disaster, useless.

REQUIRED TONE — use:
- "would benefit from" instead of "suffers from"
- "a key development area" instead of "critical failure"
- "may not progress past automated screening" instead of "will be rejected"
- "document accessibility opportunity" for technical issues (scanned PDFs, encoding issues)

Return ONLY a valid JSON object — no markdown, no preamble, no commentary.

{
  "score": <0-100 integer. Score holistically. If content exists, minimum 20.>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<actionable professional recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "rewrite_example": "<professionally rewritten summary paragraph — write this regardless of document quality, using available context>",
  "advisory_note": "<1-2 sentences of warm, forward-looking career guidance from a senior HR practitioner>"
}"""

BATCH_SYSTEM = """You are a Chartered HR Practitioner and Occupational Psychologist at Direct Labour Consult conducting a structured comparative talent assessment.

Assess and rank all candidates professionally using industrial psychology principles. Language must be constructive, evidence-based, and respectful of each candidate's professional journey.

PROHIBITED WORDS: suffers, corrupted, failure, rejected, unreadable, terrible, weak, broken.

Return ONLY a valid JSON array — no markdown, no extra text.

Each element:
{
  "rank": <1-based integer>,
  "name": "<candidate name from CV, or 'Candidate N' if not determinable>",
  "score": <0-100 integer>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<development area 1>", "<development area 2>"],
  "hire_recommendation": "<Strong Yes|Yes|Consider|Not Recommended>",
  "summary": "<2-sentence professional hiring manager summary using occupational psychology framing>"
}"""

# ── industrial-grade JSON extractor ──────────────────────────────────────────
def safe_parse_json(raw: str) -> dict | list:
    """
    Multi-stage JSON extraction. Never raises — always returns a parsed object
    or raises JSONDecodeError only after all recovery attempts exhausted.

    Stages:
    1. Strip markdown fences and control chars
    2. Direct parse
    3. Bracket-depth extraction (finds outermost { } or [ ])
    4. Regex fallback (last { ... } or [ ... ] in string)
    5. Truncated JSON repair (close open braces/brackets/strings)
    """
    if not raw:
        raise json.JSONDecodeError("Empty response", "", 0)

    # Stage 1 — cleanup
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    # Remove control characters except tab, newline, carriage return
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Stage 2 — direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Stage 3 — bracket-depth extraction
    for open_c, close_c in [('{', '}'), ('[', ']')]:
        start = cleaned.find(open_c)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc    = False
        end    = -1
        for i, ch in enumerate(cleaned[start:], start):
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    # Stage 4 — regex fallback (last complete-looking object/array)
    for pattern in (r'\{[^{}]*\}', r'\[[^\[\]]*\]'):
        matches = re.findall(pattern, cleaned, re.DOTALL)
        for m in reversed(matches):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue

    # Stage 5 — truncated repair
    attempt = cleaned
    if cleaned and cleaned[0] in ('{', '['):
        # Close open string
        if attempt.count('"') % 2 != 0:
            attempt += '"'
        # Close arrays then objects
        attempt += ']' * max(0, attempt.count('[') - attempt.count(']'))
        attempt += '}' * max(0, attempt.count('{') - attempt.count('}'))
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("All JSON extraction strategies exhausted", cleaned, 0)


def call_claude(system_prompt: str, user_content: str, image_blocks: list | None = None) -> dict | list:
    if image_blocks:
        content = [{"type": "text", "text": user_content}] + image_blocks
    else:
        content = user_content
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw = msg.content[0].text.strip()
    logger.debug(f"[claude] raw response length={len(raw)}")
    return safe_parse_json(raw)


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "keep_warm": "active",
        "pdf_support": HAS_PDF,
        "docx_support": HAS_DOCX,
    })

@app.route("/version", methods=["GET"])
def version():
    """Deployment verification — check this URL after every Render deploy."""
    return jsonify({
        "version":          "DLC_BACKEND_V14C",
        "build_date":       "2026-05-21",
        "parser":           "safe_parse_json_v2",
        "upload_detection": "multi-field",
        "fields_accepted":  ["files", "files[]", "cvs", "cvs[]"],
        "admin_endpoint":   "/admin/test",
        "status":           "production",
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "DLC CV Backend", "status": "running", "version": "14c"})


# ── /analyze — single CV ──────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    ts    = datetime.utcnow().isoformat()
    name  = request.form.get("name", "Unknown").strip()
    email = request.form.get("email", "").strip()
    file  = request.files.get("file")

    logger.info(f"[/analyze] name={name} email={email} admin={check_admin(request)} ts={ts}")

    if not file or not file.filename:
        return jsonify({"success": False, "error": "No CV file uploaded."}), 400

    fname = file.filename.lower()
    if not fname.endswith((".pdf", ".doc", ".docx")):
        return jsonify({"success": False, "error": "Only PDF, DOC, or DOCX files are accepted."}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"success": False, "error": "File too large. Maximum 10 MB."}), 400

    try:
        cv_text, method = extract_text(file)
        if len(cv_text) < 30:
            return jsonify({
                "success": False,
                "error": "The document content could not be extracted. Please ensure the file is not password-protected or image-only, and try again."
            }), 422

        result = call_claude(SINGLE_SYSTEM, f"Candidate Name: {name}\n\nCV CONTENT:\n{cv_text}")

        data = {
            "score":             result.get("score", result.get("overall_score", 0)),
            "market_readiness":  result.get("market_readiness", "Developing"),
            "strengths":         result.get("strengths", result.get("top_strengths", [])),
            "improvements":      result.get("improvements", result.get("key_improvements", [])),
            "rewrite_example":   result.get("rewrite_example", result.get("rewritten_summary", "")),
            "advisory_note":     result.get("advisory_note", ""),
            "name":              name,
            "email":             email,
            "extraction_method": method,
            "analysed_at":       ts,
        }

        logger.info(f"[/analyze] SUCCESS name={name} score={data['score']}")
        log_submission({
            "type": "individual", "name": name, "email": email,
            "score": data["score"], "market_readiness": data["market_readiness"],
            "file": file.filename, "status": "success", "timestamp": ts,
        })
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError as e:
        logger.error(f"[/analyze] JSON parse error: {e}")
        log_submission({"type": "individual", "name": name, "email": email,
                        "status": "error", "error": "JSON parse", "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Analysis temporarily unavailable. Please retry in a moment."
        }), 502
    except Exception as e:
        logger.error(f"[/analyze] ERROR: {e}", exc_info=True)
        log_submission({"type": "individual", "name": name, "email": email,
                        "status": "error", "error": str(e)[:120], "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Analysis could not be completed. Please try again."
        }), 500


# ── /analyze-batch — recruiter multi-CV ───────────────────────────────────────
@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    ts = datetime.utcnow().isoformat()

    job_title   = (request.form.get("job_title") or "Open Position").strip()
    job_dept    = request.form.get("job_dept", "")
    job_exp     = request.form.get("job_exp", "")
    job_edu     = request.form.get("job_edu", "")
    job_type    = request.form.get("job_type", "")
    job_skills  = request.form.get("job_skills", "")
    job_desc    = request.form.get("job_desc", "")
    job_disq    = request.form.get("job_disq", "")
    weights_raw = request.form.get("weights", "{}")
    try:
        weights = json.loads(weights_raw)
    except Exception:
        weights = {}

    # Accept files under any field name the frontend might use
    files = get_uploaded_files()

    logger.info(f"[/analyze-batch] job='{job_title}' files={len(files)} ts={ts}")

    if not files:
        return jsonify({"success": False, "error": "No CV files provided."}), 400
    if len(files) > 20:
        return jsonify({"success": False, "error": "Maximum 20 CVs per batch."}), 400

    extracted, errors = [], []
    for f in files:
        fname = (f.filename or "").lower()
        if not fname.endswith((".pdf", ".doc", ".docx")):
            errors.append(f"{f.filename}: format not supported (use PDF, DOC, DOCX)")
            continue
        f.seek(0, 2)
        fsize = f.tell()
        f.seek(0)
        if fsize > 10 * 1024 * 1024:
            errors.append(f"{f.filename}: file too large (max 10 MB), skipped")
            continue
        try:
            text, method = extract_text(f)
            if len(text) < 30:
                errors.append(f"{f.filename}: content could not be extracted, skipped")
                continue
            extracted.append({"filename": f.filename, "text": text, "method": method})
        except Exception as e:
            errors.append(f"{f.filename}: extraction error ({e}), skipped")

    if not extracted:
        return jsonify({
            "success": False,
            "error": "No readable CV content found in the uploaded files.",
            "file_errors": errors
        }), 422

    # Build role context for the AI
    job_ctx  = f"POSITION: {job_title}\n"
    if job_dept:   job_ctx += f"DEPARTMENT: {job_dept}\n"
    if job_exp:    job_ctx += f"MINIMUM EXPERIENCE: {job_exp}+ years\n"
    if job_edu:    job_ctx += f"EDUCATION REQUIREMENT: {job_edu}\n"
    if job_type:   job_ctx += f"EMPLOYMENT TYPE: {job_type}\n"
    if job_skills: job_ctx += f"KEY SKILLS REQUIRED: {job_skills}\n"
    if job_desc:   job_ctx += f"ROLE DESCRIPTION: {job_desc}\n"
    if job_disq:   job_ctx += f"DISQUALIFYING FACTORS: {job_disq}\n"
    if weights:    job_ctx += f"SCORING WEIGHTS: {json.dumps(weights)}\n"

    prompt = job_ctx + f"\nTotal Candidates: {len(extracted)}\n\n"
    for idx, cv in enumerate(extracted, 1):
        prompt += f"--- CANDIDATE {idx}: {cv['filename']} ---\n{cv['text'][:3000]}\n\n"

    try:
        results = call_claude(BATCH_SYSTEM, prompt)
        if not isinstance(results, list):
            results = [results]

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i

        logger.info(f"[/analyze-batch] SUCCESS job='{job_title}' candidates={len(results)}")
        log_submission({
            "type": "batch", "job_title": job_title,
            "total_candidates": len(results), "status": "success", "timestamp": ts,
        })
        return jsonify({
            "success": True,
            "job_title": job_title,
            "total_candidates": len(results),
            "analysed_at": ts,
            "file_errors": errors,
            "candidates": results,
        })

    except json.JSONDecodeError as e:
        logger.error(f"[/analyze-batch] JSON parse error: {e}")
        log_submission({"type": "batch", "job_title": job_title,
                        "status": "error", "error": "JSON parse", "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Analysis temporarily unavailable. Please retry in a moment."
        }), 502
    except Exception as e:
        logger.error(f"[/analyze-batch] ERROR: {e}", exc_info=True)
        log_submission({"type": "batch", "job_title": job_title,
                        "status": "error", "error": str(e)[:120], "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Batch analysis could not be completed. Please try again."
        }), 500


# ── admin endpoints ───────────────────────────────────────────────────────────
@app.route("/admin/submissions", methods=["GET"])
def admin_submissions():
    if not check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    with _log_lock:
        return jsonify({"submissions": list(_submissions), "total": len(_submissions)})

@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    if not check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    with _log_lock:
        total   = len(_submissions)
        success = sum(1 for s in _submissions if s.get("status") == "success")
        indiv   = sum(1 for s in _submissions if s.get("type") == "individual")
        batch   = sum(1 for s in _submissions if s.get("type") == "batch")
        scores  = [s["score"] for s in _submissions if s.get("score")]
        avg     = round(sum(scores) / len(scores), 1) if scores else 0
    return jsonify({
        "total_submissions": total, "successful": success,
        "failed": total - success, "individual_cvs": indiv,
        "batch_jobs": batch, "avg_score": avg,
        "server_time": datetime.utcnow().isoformat(),
    })

@app.route("/admin/test", methods=["POST"])
def admin_test():
    """Admin-only: full analysis without payment verification."""
    if not check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    return analyze()


# ══════════════════════════════════════════════════════════════════════════════
# ── ATS APPLICATION OPTIMISATION — additive feature, isolated from CV diagnostic
#    and recruiter routes above. Nothing above this line was modified.
# ══════════════════════════════════════════════════════════════════════════════

ATS_OPTIMIZE_SYSTEM = """You are the DLC ATS Application Engine™ — acting simultaneously as an Applicant Tracking System and a Chartered HR Practitioner / Occupational Psychologist at Direct Labour Consult. You evaluate one candidate against one specific job description, the way a real recruiter and a real ATS would, and then produce an optimised application.

Work through these steps using the CV (and, if provided, the target job description):

1. PARSE THE CV — extract the candidate's real name, most recent/likely professional title, location if stated, contact details if present, skills, full work history, education, and certifications. Never invent a name, employer, date, or qualification the candidate did not provide — use "[Not provided]" as a placeholder for header fields you cannot find, rather than guessing.
2. IF A JOB DESCRIPTION IS PROVIDED — identify required skills, preferred/nice-to-have skills, keywords, and the seniority/experience level implied, then align the rewritten CV to it.
3. MATCH (only if a job description was provided) — compare CV to role: skills overlap, experience alignment, keyword overlap, title alignment.
4. IDENTIFY GAPS (only if a job description was provided) — required skills missing or under-evidenced, weakly-written bullets, keywords the role uses that the CV doesn't.
5. IDENTIFY RISKS — flag realistic hiring-manager concerns ONLY when the evidence genuinely supports them (job hopping, unexplained gaps, over/underqualification, ATS-unfriendly formatting). Do not invent risks.
6. SCORE:
   - ats_compatibility_score (0-100): how well-structured and machine-parseable the CV itself is.
   - job_match_score (0-100 or null): only if a job description was provided — how well the candidate matches THIS role. Return null if no job description was given.
   - hiring_readiness_score (0-100): weighted composite. If a job description was provided: ATS Compatibility 25% / Job Match 40% / Content Strength 20% / Risk Factors 15%. If no job description was provided, reweight as: ATS Compatibility 40% / Content Strength 40% / Risk Factors 20% (job match removed from the formula since there is no target role to match against).
7. BUILD THE FULL CV — produce a complete, corporate, ATS-compliant CV using ONLY the candidate's real, provided experience and qualifications:
   - Professional summary: 4-5 lines, strong corporate language, naming years of experience (estimate conservatively from the CV's own dates if not explicit), industry focus, key strengths, and value to an employer. Do NOT use generic filler phrases such as "results-driven professional," "team player," "dynamic professional," "passionate about," or "synergy."
   - Core competencies: 8-12 concise ATS keyword phrases genuinely evidenced by the candidate's real background (e.g. "Supplier Management," "Cost Optimisation," "Contract Negotiation") — never invent expertise the CV doesn't support.
   - Professional experience: for EACH role found in the CV, rewrite 4-6 achievement-based bullets using strong action verbs (Led, Implemented, Managed, Improved, Delivered, Negotiated, Streamlined). Each bullet should reflect genuine impact and business value using only what the candidate actually described — you may reframe, quantify, and sharpen language, but never fabricate metrics, outcomes, or responsibilities not implied by the original content.
   - Education: cleanly rewritten from what the CV states.
   - Certifications: list only what the CV or supporting documents actually evidence. Return an empty list if none are found — never infer a certification the candidate never mentioned.

Use professional HR language throughout.

PROHIBITED WORDS: suffers, corrupted, failure, rejected, unreadable, terrible, poor, bad, wrong, weak, broken, results-driven, synergy.

Return ONLY a valid JSON object — no markdown, no preamble, no commentary:

{
  "ats_score": <0-100 integer>,
  "job_match_score": <0-100 integer, or null if no job description was provided>,
  "hiring_readiness_score": <0-100 integer, weighted per step 6>,
  "gaps": ["<gap 1>", "<gap 2>"],
  "risks": ["<risk, only if genuinely evidenced — empty array if none>"],
  "cv_document": {
    "header": {
      "full_name": "<candidate's real name, or '[Not provided]'>",
      "professional_title": "<derived from most recent/dominant experience>",
      "location": "<if stated, else '[Not provided]'>",
      "email": "<if stated, else '[Not provided]'>",
      "phone": "<if stated, else '[Not provided]'>"
    },
    "professional_summary": "<4-5 line corporate summary>",
    "core_competencies": ["<competency 1>", "<competency 2>", "... 8-12 total"],
    "professional_experience": [
      {
        "title": "<role title>",
        "employer": "<employer name>",
        "dates": "<date range as stated>",
        "bullets": ["<achievement bullet 1>", "<bullet 2>", "... 4-6 total"]
      }
    ],
    "education": [
      {"qualification": "<degree/qualification>", "institution": "<institution>", "dates": "<if stated>"}
    ],
    "certifications": ["<certification 1>", "... or empty array if none found"]
  },
  "keywords_added": ["<keyword 1>", "<keyword 2>"]
}"""


def build_cv_docx(cv_document: dict) -> bytes:
    """
    Builds a real, ATS-readable .docx CV from the structured cv_document
    returned by the model. No tables, no graphics, no text boxes — plain
    headings and bullet lists only, so ATS parsers can read it cleanly.
    Returns raw docx bytes.
    """
    from docx import Document as DocxWriter
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io

    doc = DocxWriter()

    header = cv_document.get("header", {}) or {}

    name_p = doc.add_paragraph()
    name_run = name_p.add_run(header.get("full_name") or "[Not provided]")
    name_run.bold = True
    name_run.font.size = Pt(20)
    name_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    title_p = doc.add_paragraph()
    title_run = title_p.add_run(header.get("professional_title") or "")
    title_run.font.size = Pt(13)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    contact_bits = [b for b in [header.get("location"), header.get("email"), header.get("phone")]
                    if b and b != "[Not provided]"]
    if contact_bits:
        contact_p = doc.add_paragraph()
        contact_p.add_run(" | ".join(contact_bits)).font.size = Pt(10)
        contact_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_heading(text):
        return doc.add_heading(text, level=2)

    if cv_document.get("professional_summary"):
        add_heading("Professional Summary")
        doc.add_paragraph(cv_document["professional_summary"])

    comps = cv_document.get("core_competencies") or []
    if comps:
        add_heading("Core Competencies")
        for i in range(0, len(comps), 2):
            pair = comps[i:i+2]
            doc.add_paragraph(" • ".join(pair))

    experience = cv_document.get("professional_experience") or []
    if experience:
        add_heading("Professional Experience")
        for role in experience:
            role_p = doc.add_paragraph()
            role_run = role_p.add_run(f"{role.get('title','')} — {role.get('employer','')}")
            role_run.bold = True
            if role.get("dates"):
                dates_p = doc.add_paragraph()
                dates_run = dates_p.add_run(role["dates"])
                dates_run.italic = True
                dates_run.font.size = Pt(10)
            for bullet in role.get("bullets", []):
                doc.add_paragraph(bullet, style="List Bullet")

    education = cv_document.get("education") or []
    if education:
        add_heading("Education")
        for edu in education:
            line = edu.get("qualification", "")
            if edu.get("institution"):
                line += f" — {edu['institution']}"
            if edu.get("dates"):
                line += f" ({edu['dates']})"
            doc.add_paragraph(line, style="List Bullet")

    certs = cv_document.get("certifications") or []
    if certs:
        add_heading("Certifications")
        for c in certs:
            doc.add_paragraph(c, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


SUPPORTING_DOC_MAX_COUNT = 5
SUPPORTING_DOC_MAX_SIZE  = 10 * 1024 * 1024  # 10 MB each, same cap as the main CV
SUPPORTING_DOC_ALLOWED   = (".pdf", ".doc", ".docx", ".jpg", ".jpeg", ".png")

def process_supporting_docs(files):
    """
    Lightweight, additive processing for optional supporting documents
    (certifications, qualifications, etc). Does not use OCR or any new
    dependency: PDFs/DOCs reuse the existing extract_text() function;
    images are passed directly to Claude's vision capability as part of
    the same model call. Returns (supporting_text, image_blocks, used_count).
    Silently skips files beyond the count/size caps or unsupported types
    rather than failing the whole request.
    """
    supporting_text_parts = []
    image_blocks = []
    used_count = 0

    for f in files[:SUPPORTING_DOC_MAX_COUNT]:
        if not f or not f.filename:
            continue
        fname = f.filename.lower()
        if not fname.endswith(SUPPORTING_DOC_ALLOWED):
            continue

        f.seek(0, 2)
        fsize = f.tell()
        f.seek(0)
        if fsize > SUPPORTING_DOC_MAX_SIZE:
            continue

        if fname.endswith((".pdf", ".doc", ".docx")):
            try:
                text, _ = extract_text(f)
                if len(text.strip()) >= 20:
                    supporting_text_parts.append(f"--- {f.filename} ---\n{text.strip()}")
                    used_count += 1
            except Exception as e:
                logger.warning(f"[supporting_docs] extract failed for {f.filename}: {e}")

        elif fname.endswith((".jpg", ".jpeg", ".png")):
            try:
                raw = f.read()
                media_type = "image/png" if fname.endswith(".png") else "image/jpeg"
                image_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(raw).decode("utf-8"),
                    },
                })
                used_count += 1
            except Exception as e:
                logger.warning(f"[supporting_docs] image read failed for {f.filename}: {e}")

    supporting_text = "\n\n".join(supporting_text_parts) if supporting_text_parts else ""
    return supporting_text, image_blocks, used_count


@app.route("/optimize-application", methods=["POST"])
def optimize_application():
    """
    DLC ATS CV Generator — additive endpoint.
    Accepts a CV file (required) and job description (optional — if provided,
    the CV is tailored to that specific role and scored against it; if not,
    a strong general ATS-optimised CV is still produced). Generates a full
    structured CV and a real downloadable .docx file.
    Does not touch /analyze, /analyze-batch, or any recruiter logic above.
    """
    ts   = datetime.utcnow().isoformat()
    file = request.files.get("file")
    job_description = request.form.get("job_description", "").strip()

    logger.info(f"[/optimize-application] admin={check_admin(request)} ts={ts} has_job_desc={bool(job_description)}")

    if not file or not file.filename:
        return jsonify({"success": False, "error": "No CV file uploaded."}), 400

    fname = file.filename.lower()
    if not fname.endswith((".pdf", ".doc", ".docx")):
        return jsonify({"success": False, "error": "Only PDF, DOC, or DOCX files are accepted."}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"success": False, "error": "File too large. Maximum 10 MB."}), 400

    # job_description is now optional. If provided, it must be substantive.
    if job_description and len(job_description) < 40:
        return jsonify({"success": False, "error": "Please provide the full job description, or leave it blank for a general optimisation."}), 400

    try:
        cv_text, method = extract_text(file)
        if len(cv_text) < 30:
            return jsonify({
                "success": False,
                "error": "The document content could not be extracted. Please ensure the file is not password-protected or image-only, and try again."
            }), 422

        # Optional supporting documents — additional certifications, qualifications, etc.
        supporting_docs = request.files.getlist("supporting_docs")
        supporting_text, image_blocks, docs_used = process_supporting_docs(supporting_docs)

        user_content = f"CANDIDATE CV:\n{cv_text}"
        if job_description:
            user_content += f"\n\nTARGET JOB DESCRIPTION:\n{job_description}"
        else:
            user_content += "\n\n(No target job description was provided — produce a strong general ATS-optimised CV. Set job_match_score to null and reweight hiring_readiness_score as instructed.)"
        if supporting_text:
            user_content += f"\n\nSUPPORTING DOCUMENTS (certifications/qualifications provided by the candidate):\n{supporting_text}"
        if image_blocks:
            user_content += "\n\n(Additional supporting document images are attached — review them for relevant certifications or qualifications.)"

        result = call_claude(ATS_OPTIMIZE_SYSTEM, user_content, image_blocks=image_blocks or None)

        cv_document = result.get("cv_document", {}) or {}

        # Generate the real .docx file from the structured CV
        docx_bytes = build_cv_docx(cv_document)
        docx_b64 = base64.b64encode(docx_bytes).decode("utf-8")

        data = {
            "ats_score":              result.get("ats_score", 0),
            "job_match_score":        result.get("job_match_score"),  # null if no job description
            "hiring_readiness_score": result.get("hiring_readiness_score", 0),
            "gaps":                   result.get("gaps", []),
            "risks":                  result.get("risks", []),
            "cv_preview":             cv_document,
            "keywords_added":         result.get("keywords_added", []),
            "extraction_method":      method,
            "optimised_at":           ts,
            "download_filename":      "DLC_Optimised_CV.docx",
            "cv_file_base64":         docx_b64,
        }
        if docs_used > 0:
            data["supporting_docs_note"] = "Additional certifications and qualifications have been integrated"

        logger.info(
            f"[/optimize-application] SUCCESS "
            f"hiring_readiness={data['hiring_readiness_score']} "
            f"job_match={data['job_match_score']} ats={data['ats_score']}"
        )
        log_submission({
            "type": "ats_optimisation", "score": data["hiring_readiness_score"],
            "file": file.filename, "status": "success", "timestamp": ts,
        })
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError as e:
        logger.error(f"[/optimize-application] JSON parse error: {e}")
        log_submission({"type": "ats_optimisation", "status": "error",
                        "error": "JSON parse", "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Optimisation temporarily unavailable. Please retry in a moment."
        }), 502
    except Exception as e:
        logger.error(f"[/optimize-application] ERROR: {e}", exc_info=True)
        log_submission({"type": "ats_optimisation", "status": "error",
                        "error": str(e)[:120], "timestamp": ts})
        return jsonify({
            "success": False,
            "error": "Optimisation could not be completed. Please try again."
        }), 500


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
