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

# ── keep-warm ping ─────────────────────────────────────────────────────────────
# Render free tier sleeps after 15 min inactivity. This self-pings every 13 min
# so the service NEVER goes cold for end users.
def keep_warm():
    while True:
        try:
            time.sleep(13 * 60)  # 13 minutes
            r = requests.get(f"{BACKEND_URL}/health", timeout=10)
            logger.info(f"[keep-warm] ping → {r.status_code}")
        except Exception as e:
            logger.warning(f"[keep-warm] ping failed: {e}")

threading.Thread(target=keep_warm, daemon=True).start()
logger.info("[keep-warm] self-ping thread started — backend will never sleep")

# ── admin key ─────────────────────────────────────────────────────────────────
ADMIN_KEY = os.environ.get("ADMIN_KEY", "dlc-admin-2026")

# ── in-memory submission log (last 200 entries) ───────────────────────────────
_log_lock = threading.Lock()
_submissions = []   # list of dicts, newest first

def log_submission(entry: dict):
    with _log_lock:
        _submissions.insert(0, entry)
        if len(_submissions) > 200:
            _submissions.pop()


# ── text extraction ───────────────────────────────────────────────────────────
def extract_text(file_storage):
    filename = (file_storage.filename or "").lower()
    ext = os.path.splitext(filename)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    text = ""
    method = "unknown"
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

    logger.info(f"[extract] method={method} chars={len(text)}")
    return text.strip(), method


# ── Claude helpers ────────────────────────────────────────────────────────────
SINGLE_SYSTEM = """You are a Chartered HR Practitioner and Occupational Psychologist at Direct Labour Consult, a professional HR advisory firm. You conduct structured CV assessments using industrial psychology principles and evidence-based career development frameworks.

Your assessment must be constructive, professional, and empowering — the tone should reflect a senior career consultant providing developmental feedback, not a critic. Never use words like: suffers, corrupted, failure, rejected, unreadable, terrible, poor, bad, wrong, or any language that demeans the candidate. Where a CV has technical issues (e.g. scanned image, unextractable text), frame this as a document accessibility opportunity, not a personal failing.

Use the language of occupational psychology and HR best practice:
- Instead of "your CV is bad" → "there is an opportunity to strengthen the document's market positioning"
- Instead of "suffers from" → "would benefit from"  
- Instead of "critical failure" → "a key development area"
- Instead of "immediately rejected" → "may not progress past automated screening"

Analyse the CV and return ONLY a valid JSON object — no markdown, no extra text, no commentary before or after.

{
  "score": <0-100 integer — assess holistically; even a weak CV should score above 20 if some content exists>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["<genuine strength 1 — frame positively>", "<strength 2>", "<strength 3>"],
  "improvements": ["<specific, actionable, professionally worded recommendation 1>", "<recommendation 2>", "<recommendation 3>"],
  "rewrite_example": "<a professionally rewritten summary paragraph demonstrating best practice — write this even if the original is weak, using what context is available>",
  "advisory_note": "<1-2 sentences of warm, forward-looking career guidance from a senior HR practitioner perspective>"
}"""

BATCH_SYSTEM = """You are a Chartered HR Practitioner and Occupational Psychologist at Direct Labour Consult. You are conducting a structured comparative talent assessment using industrial psychology principles.

Assess and rank all candidates professionally and objectively. Your language must reflect senior HR practice — constructive, evidence-based, and respectful of each candidate's professional journey. Avoid demeaning language.

Return ONLY a valid JSON array — no markdown, no extra text.

Each element:
{
  "rank": <1-based integer>,
  "name": "<candidate name or 'Candidate N' if unknown>",
  "score": <0-100 integer>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "improvements": ["<development area 1>", "<development area 2>"],
  "hire_recommendation": "<Strong Yes|Yes|Consider|Not Recommended>",
  "summary": "<2-sentence professional hiring manager assessment using occupational psychology framing>"
}"""


def call_claude(system_prompt: str, user_content: str) -> dict | list:
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = msg.content[0].text.strip()

    # Strip markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    # Remove control characters that break JSON (except \n \r \t)
    raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', raw)

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract the outermost JSON object or array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = raw.find(start_char)
        if start == -1:
            continue
        # Find matching close by tracking depth
        depth = 0
        in_string = False
        escape_next = False
        end_pos = -1
        for i, ch in enumerate(raw[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        if end_pos != -1:
            candidate = raw[start:end_pos + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Last resort: truncated JSON — try to close it
                pass

    # Last resort: close truncated JSON by appending missing structure
    # Try to complete a truncated object
    attempt = raw
    if raw.startswith('{') or raw.startswith('['):
        open_braces = raw.count('{') - raw.count('}')
        open_brackets = raw.count('[') - raw.count(']')
        # Close any open string first (heuristic)
        if attempt.count('"') % 2 != 0:
            attempt += '"'
        attempt += ']' * max(0, open_brackets)
        attempt += '}' * max(0, open_braces)
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract valid JSON from AI response", raw, 0)


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "DLC CV Backend", "status": "running"})


# ── /analyze — single CV ──────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    ts = datetime.utcnow().isoformat()
    is_admin = check_admin(request)
    name = request.form.get("name", "Unknown")
    email = request.form.get("email", "")
    file = request.files.get("file")

    logger.info(f"[/analyze] name={name} email={email} ts={ts}")

    if not file or not file.filename:
        return jsonify({"success": False, "error": "No CV file uploaded."}), 400

    fname = file.filename.lower()
    if not fname.endswith((".pdf", ".doc", ".docx")):
        return jsonify({"success": False, "error": "Only PDF, DOC, or DOCX files accepted."}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:
        return jsonify({"success": False, "error": "File too large. Maximum 10 MB."}), 400

    try:
        cv_text, method = extract_text(file)
        if len(cv_text) < 50:
            return jsonify({"success": False, "error": "Could not read CV content. Please try a different format."}), 422

        result = call_claude(SINGLE_SYSTEM, f"Name: {name}\n\nCV TEXT:\n{cv_text}")

        # normalise field names for legacy frontend compatibility
        data = {
            "score":            result.get("score", result.get("overall_score", 0)),
            "market_readiness": result.get("market_readiness", ""),
            "strengths":        result.get("strengths", result.get("top_strengths", [])),
            "improvements":     result.get("improvements", result.get("key_improvements", [])),
            "rewrite_example":  result.get("rewrite_example", result.get("rewritten_summary", "")),
            "advisory_note":    result.get("advisory_note", ""),
            "name":             name,
            "email":            email,
            "extraction_method": method,
            "analysed_at":      ts,
        }

        logger.info(f"[/analyze] SUCCESS name={name} score={data['score']}")
        log_submission({
            "type": "individual",
            "name": name,
            "email": email,
            "score": data["score"],
            "market_readiness": data["market_readiness"],
            "file": file.filename,
            "status": "success",
            "timestamp": ts,
        })
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError as e:
        logger.error(f"[/analyze] JSON parse error: {e}")
        log_submission({"type":"individual","name":name,"email":email,"status":"error","error":"JSON parse","timestamp":ts})
        return jsonify({"success": False, "error": "Analysis engine returned an unexpected response. Please try again."}), 502
    except Exception as e:
        logger.error(f"[/analyze] ERROR: {e}", exc_info=True)
        log_submission({"type":"individual","name":name,"email":email,"status":"error","error":str(e)[:120],"timestamp":ts})
        return jsonify({"success": False, "error": "Analysis failed. Please try again in a moment."}), 500


# ── /analyze-batch — recruiter multi-CV ───────────────────────────────────────
@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    ts = datetime.utcnow().isoformat()
    job_title   = request.form.get("job_title", "Open Position")
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

    files = request.files.getlist("files") or request.files.getlist("cvs[]")

    logger.info(f"[/analyze-batch] job={job_title} count={len(files)} ts={ts}")

    if not files or len(files) == 0:
        return jsonify({"success": False, "error": "No CV files provided."}), 400
    if len(files) > 20:
        return jsonify({"success": False, "error": "Maximum 20 CVs per batch."}), 400

    extracted = []
    errors = []
    for i, f in enumerate(files):
        fname = f.filename.lower()
        if not fname.endswith((".pdf", ".doc", ".docx")):
            errors.append(f"{f.filename}: unsupported format, skipped")
            continue
        try:
            text, method = extract_text(f)
            if len(text) < 50:
                errors.append(f"{f.filename}: could not read content, skipped")
                continue
            extracted.append({"filename": f.filename, "text": text, "method": method})
        except Exception as e:
            errors.append(f"{f.filename}: extraction failed ({e}), skipped")

    if not extracted:
        return jsonify({"success": False, "error": "No readable CVs found in upload.", "file_errors": errors}), 422

    # Build rich job context for AI
    job_context = f"POSITION: {job_title}\n"
    if job_dept:    job_context += f"DEPARTMENT: {job_dept}\n"
    if job_exp:     job_context += f"MINIMUM EXPERIENCE: {job_exp}+ years\n"
    if job_edu:     job_context += f"EDUCATION REQUIREMENT: {job_edu}\n"
    if job_type:    job_context += f"EMPLOYMENT TYPE: {job_type}\n"
    if job_skills:  job_context += f"KEY SKILLS REQUIRED: {job_skills}\n"
    if job_desc:    job_context += f"ROLE DESCRIPTION: {job_desc}\n"
    if job_disq:    job_context += f"DISQUALIFYING FACTORS: {job_disq}\n"
    if weights:     job_context += f"SCORING WEIGHTS: {json.dumps(weights)}\n"

    combined = job_context + f"\nTotal Candidates: {len(extracted)}\n\n"
    for idx, cv in enumerate(extracted, 1):
        combined += f"--- CANDIDATE {idx}: {cv['filename']} ---\n{cv['text'][:3000]}\n\n"

    try:
        results = call_claude(BATCH_SYSTEM, combined)
        if not isinstance(results, list):
            results = [results]

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i

        logger.info(f"[/analyze-batch] SUCCESS job={job_title} candidates={len(results)}")
        log_submission({
            "type": "batch",
            "job_title": job_title,
            "total_candidates": len(results),
            "status": "success",
            "timestamp": ts,
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
        log_submission({"type":"batch","job_title":job_title,"status":"error","error":"JSON parse","timestamp":ts})
        return jsonify({"success": False, "error": "Analysis engine returned unexpected response. Please try again."}), 502
    except Exception as e:
        logger.error(f"[/analyze-batch] ERROR: {e}", exc_info=True)
        log_submission({"type":"batch","job_title":job_title,"status":"error","error":str(e)[:120],"timestamp":ts})
        return jsonify({"success": False, "error": "Batch analysis failed. Please try again."}), 500


# ── /admin/submissions — protected log endpoint ───────────────────────────────
def check_admin(req):
    key = req.headers.get("X-Admin-Key") or req.args.get("key")
    return key == ADMIN_KEY

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
        total = len(_submissions)
        success = sum(1 for s in _submissions if s.get("status") == "success")
        individual = sum(1 for s in _submissions if s.get("type") == "individual")
        batch = sum(1 for s in _submissions if s.get("type") == "batch")
        scores = [s["score"] for s in _submissions if s.get("score")]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    return jsonify({
        "total_submissions": total,
        "successful": success,
        "failed": total - success,
        "individual_cvs": individual,
        "batch_jobs": batch,
        "avg_score": avg_score,
        "uptime_since": datetime.utcnow().isoformat(),
    })

@app.route("/admin/test", methods=["POST"])
def admin_test():
    """Admin-only test endpoint — bypasses payment, runs full analysis."""
    if not check_admin(request):
        return jsonify({"error": "Unauthorized"}), 401
    # Delegate to the normal analyze logic
    return analyze()


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
