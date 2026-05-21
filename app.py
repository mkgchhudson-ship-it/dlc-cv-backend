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
SINGLE_SYSTEM = """You are a senior HR consultant at Direct Labour Consult, Gaborone, Botswana.
Analyse the CV and return ONLY a JSON object — no markdown, no extra text.

{
  "score": <0-100 integer>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<action 1>", "<action 2>", "<action 3>"],
  "rewrite_example": "<rewritten summary paragraph>",
  "advisory_note": "<1-2 sentence personalised closing advice>"
}"""

BATCH_SYSTEM = """You are a senior HR consultant at Direct Labour Consult.
You will receive multiple CVs. Score and rank ALL of them. Return ONLY a JSON array — no markdown.

Each element:
{
  "rank": <1-based integer>,
  "name": "<candidate name or 'Candidate N' if unknown>",
  "score": <0-100 integer>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Work>",
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "hire_recommendation": "<Strong Yes|Yes|Maybe|No>",
  "summary": "<2-sentence hiring manager summary>"
}"""


def call_claude(system_prompt: str, user_content: str) -> dict | list:
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = msg.content[0].text.strip()
    # strip possible markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


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
        return jsonify({"success": True, "data": data})

    except json.JSONDecodeError as e:
        logger.error(f"[/analyze] JSON parse error: {e}")
        return jsonify({"success": False, "error": "Analysis engine returned an unexpected response. Please try again."}), 502
    except Exception as e:
        logger.error(f"[/analyze] ERROR: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Analysis failed. Please try again in a moment."}), 500


# ── /analyze-batch — recruiter multi-CV ───────────────────────────────────────
@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    ts = datetime.utcnow().isoformat()
    job_title = request.form.get("job_title", "Open Position")
    files = request.files.getlist("files")

    logger.info(f"[/analyze-batch] job={job_title} count={len(files)} ts={ts}")

    if not files or len(files) == 0:
        return jsonify({"success": False, "error": "No CV files uploaded."}), 400
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

    # Build combined prompt
    combined = f"JOB TITLE: {job_title}\nTotal CVs: {len(extracted)}\n\n"
    for idx, cv in enumerate(extracted, 1):
        combined += f"--- CV {idx}: {cv['filename']} ---\n{cv['text'][:3000]}\n\n"

    try:
        results = call_claude(BATCH_SYSTEM, combined)
        if not isinstance(results, list):
            results = [results]

        # Sort by score descending, re-rank
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i

        logger.info(f"[/analyze-batch] SUCCESS job={job_title} candidates={len(results)}")
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
        return jsonify({"success": False, "error": "Analysis engine returned unexpected response. Please try again."}), 502
    except Exception as e:
        logger.error(f"[/analyze-batch] ERROR: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Batch analysis failed. Please try again."}), 500


# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
