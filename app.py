"""
Direct Labour Consult — CV Analysis Backend
============================================
Flask API that accepts CV uploads, extracts text,
sends to Claude for structured analysis, and returns JSON.

Files are NEVER stored — deleted immediately after processing.
"""

import os
import uuid
import json
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import pdfplumber
from docx import Document

app = Flask(__name__)

# Allow requests from your Netlify domain + localhost for dev
# STEP 3: Replace with your actual Netlify URL
ALLOWED_ORIGINS = [
    "https://cv.directlabourconsult.com",
    "https://directlabourconsult.com",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5500",
    "*",  # Remove this in production — keep only your domain
]
CORS(app, origins=ALLOWED_ORIGINS)

UPLOAD_DIR        = tempfile.gettempdir()
ALLOWED_EXTS      = {"pdf", "doc", "docx"}
MAX_FILE_BYTES    = 10 * 1024 * 1024   # 10 MB
MAX_TEXT_CHARS    = 12000              # Trim very long CVs


# ── HELPERS ─────────────────────────────────────────────────────────

def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def extract_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def extract_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(path: str, ext: str) -> str:
    """Route extraction by file type."""
    if ext == "pdf":
        return extract_pdf(path)
    if ext in ("doc", "docx"):
        return extract_docx(path)
    return ""


def analyse(cv_text: str, name: str) -> dict:
    """
    Send CV text to Claude and receive a structured JSON analysis.
    Returns a Python dict ready to be serialised.
    """
    # Trim to avoid token bloat
    if len(cv_text) > MAX_TEXT_CHARS:
        cv_text = cv_text[:MAX_TEXT_CHARS] + "\n[... document continues ...]"

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""You are a senior HR consultant at Direct Labour Consult, an HR advisory firm
based in Gaborone, Botswana. You are reviewing a submitted CV with the same rigour
a hiring manager would apply when screening candidates for competitive roles.

Candidate name: {name}

CV TEXT:
---
{cv_text}
---

Return ONLY a raw JSON object — no markdown, no code fences, no preamble.
Use exactly this structure:

{{
  "candidate_name": "{name}",
  "overall_score": <integer 0–100>,
  "market_readiness": "<Excellent | Strong | Developing | Needs Improvement>",
  "executive_summary": "<2–3 honest sentences on the CV's current market position>",
  "sections": {{
    "professional_summary": {{
      "score": <integer 0–100>,
      "feedback": "<specific feedback on clarity, strength, positioning>",
      "strengths": ["<strength>", "<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "work_experience": {{
      "score": <integer 0–100>,
      "feedback": "<feedback on impact, achievements, structure>",
      "strengths": ["<strength>", "<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "skills": {{
      "score": <integer 0–100>,
      "feedback": "<feedback on relevance, keyword alignment>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>", "<improvement>"]
    }},
    "formatting_readability": {{
      "score": <integer 0–100>,
      "feedback": "<feedback on layout, hierarchy, readability>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement>"]
    }},
    "ats_compatibility": {{
      "score": <integer 0–100>,
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
  "advisory_note": "<1–2 sentences of personalised DLC advisory guidance for this candidate>"
}}

Rules:
- Be direct, specific, honest. Avoid vague praise.
- Every improvement must be concrete and actionable.
- The rewrite must be meaningfully better — not just rephrased.
- Score 0–100 where 75+ = strong, 50–74 = developing, <50 = needs work."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── ROUTES ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — used by Render to verify the service is up."""
    return jsonify({"status": "ok", "service": "DLC CV Analyser"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Form fields: name, email
    File field:  cv (PDF, DOC, DOCX)
    Returns:     JSON analysis result
    """
    # ── Validate form fields
    name  = (request.form.get("name")  or "").strip()
    email = (request.form.get("email") or "").strip()

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required."}), 400
    if "cv" not in request.files:
        return jsonify({"error": "No CV file was uploaded."}), 400

    file = request.files["cv"]
    if not file.filename:
        return jsonify({"error": "No file was selected."}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Only PDF and Word documents (.pdf, .doc, .docx) are accepted."}), 400

    # ── Save to temp
    ext       = file.filename.rsplit(".", 1)[1].lower()
    tmp_name  = f"{uuid.uuid4()}.{ext}"
    tmp_path  = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        file.save(tmp_path)

        if os.path.getsize(tmp_path) > MAX_FILE_BYTES:
            return jsonify({"error": "File exceeds the 10 MB size limit."}), 400

        # ── Extract text
        cv_text = extract_text(tmp_path, ext)

        if not cv_text or len(cv_text) < 80:
            return jsonify({
                "error": (
                    "Could not extract readable text from this file. "
                    "Please ensure the CV is not a scanned image and contains selectable text."
                )
            }), 400

        # ── Analyse
        result        = analyse(cv_text, name)
        result["email"] = email
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Analysis produced an unexpected response. Please try again."}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"AI service error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        # ── ALWAYS delete the temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ── ENTRY POINT ─────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
