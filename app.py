"""
Direct Labour Consult — CV Analysis Backend
Uses raw HTTP requests — no Anthropic SDK, no httpx, no proxies issue.
"""

import json
import logging
import os
import tempfile
import traceback
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("dlc_cv")

app = Flask(__name__)
CORS(app, origins=["*"])

UPLOAD_DIR     = tempfile.gettempdir()
ALLOWED_EXTS   = {"pdf", "doc", "docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 12000


def _allowed_ext(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def _get_ext(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf(path):
    try:
        import fitz
        parts = []
        with fitz.open(path) as doc:
            for page in doc:
                parts.append(page.get_text("text"))
        text = "\n".join(parts).strip()
        if text and len(text) >= 80:
            return text, "pymupdf"
    except Exception as e:
        log.warning("PyMuPDF failed: %s", e)

    try:
        from pdfminer.high_level import extract_text as pm_extract
        text = (pm_extract(path) or "").strip()
        if text and len(text) >= 80:
            return text, "pdfminer"
    except Exception as e:
        log.warning("pdfminer failed: %s", e)

    return "", "none"

def _extract_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)
        text = "\n".join(parts).strip()
        if text:
            return text, "python-docx"
    except Exception as e:
        log.warning("docx extraction failed: %s", e)
    return "", "none"

def extract_text(path, ext):
    if ext in ("doc", "docx"):
        return _extract_docx(path)
    return _extract_pdf(path)


# ── Claude via raw HTTP — NO SDK, NO HTTPX, NO PROXIES ───────────────────────

def analyse_cv(cv_text, name):
    if len(cv_text) > MAX_TEXT_CHARS:
        cv_text = cv_text[:MAX_TEXT_CHARS] + "\n[... document continues ...]"

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    prompt = f"""You are a senior HR consultant at Direct Labour Consult, Gaborone, Botswana.
Review this CV with the rigour a hiring manager applies when screening candidates.

Candidate name: {name}

CV TEXT:
---
{cv_text}
---

Return ONLY a raw JSON object — no markdown, no code fences, no preamble.

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
      "feedback": "<feedback on ATS compatibility>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }}
  }},
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "critical_improvements": ["<action 1>", "<action 2>", "<action 3>"],
  "rewritten_section": {{
    "section_name": "<section name>",
    "original_excerpt": "<original text max 80 words>",
    "rewritten": "<professionally rewritten version>"
  }},
  "advisory_note": "<1-2 sentences of personalised DLC advisory guidance>"
}}

Rules: Be direct and specific. Every improvement must be actionable.
Score: 75+ = strong, 50-74 = developing, below 50 = needs work."""

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": "claude-opus-4-5",
        "max_tokens": 2500,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=90,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic API {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    raw = data["content"][0]["text"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "DLC CV Analyser",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    req_id   = str(uuid.uuid4())[:8]
    tmp_path = None

    name  = (request.form.get("name")  or "").strip()
    email = (request.form.get("email") or "").strip()

    if not name:
        return jsonify({"success": False, "error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"success": False, "error": "A valid email address is required."}), 400
    if "cv" not in request.files:
        return jsonify({"success": False, "error": "No CV file uploaded."}), 400

    file = request.files["cv"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not _allowed_ext(file.filename):
        return jsonify({"success": False, "error": "Only PDF, DOC, DOCX accepted."}), 400

    ext = _get_ext(file.filename)
    log.info("[%s] received | name=%s | email=%s | file=%s", req_id, name, email, file.filename)

    try:
        tmp_path = os.path.join(UPLOAD_DIR, f"{req_id}_{uuid.uuid4()}.{ext}")
        file.save(tmp_path)

        if os.path.getsize(tmp_path) > MAX_FILE_BYTES:
            return jsonify({"success": False, "error": "File exceeds 10MB limit."}), 400

        cv_text, method = extract_text(tmp_path, ext)
        log.info("[%s] extraction=%s | chars=%d", req_id, method, len(cv_text))

        if not cv_text or len(cv_text) < 80:
            return jsonify({"success": False,
                "error": "Could not extract text from this file. Please ensure the CV contains selectable text and is not a scanned image."}), 400

        log.info("[%s] sending to Claude", req_id)
        result = analyse_cv(cv_text, name)
        result["email"] = email

        log.info("[%s] SUCCESS | score=%s | readiness=%s",
                 req_id, result.get("overall_score"), result.get("market_readiness"))

        return jsonify({"success": True, "data": result})

    except json.JSONDecodeError as e:
        log.error("[%s] JSON parse error: %s", req_id, e)
        return jsonify({"success": False, "error": "Analysis returned invalid format. Please retry."}), 500
    except Exception as e:
        log.error("[%s] Error: %s\n%s", req_id, str(e), traceback.format_exc())
        return jsonify({"success": False, "error": "An error occurred. Please try again."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                log.info("[%s] temp file deleted", req_id)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
