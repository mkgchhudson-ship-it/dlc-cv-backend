import os
import json
import logging
import tempfile
import threading
import time
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import requests

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


def call_claude(system_prompt: str, user_content: str) -> dict | list:
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
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


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
