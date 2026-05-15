import os
import json
import logging
import tempfile
import re
import gc
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
CORS(app, origins=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MAX_TEXT_CHARS = 15000
MAX_PAGES = 30

def extract_text_from_path(tmp_path, filename):
    text = ""
    fname = filename.lower()

    if fname.endswith(".pdf"):
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    chunks = []
                    for page in pdf.pages[:MAX_PAGES]:
                        t = page.extract_text()
                        if t:
                            chunks.append(t)
                    text = "\n".join(chunks)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")

        if not text or len(text) < 100:
            if HAS_PYPDF:
                try:
                    reader = pypdf.PdfReader(tmp_path)
                    chunks = []
                    for page in reader.pages[:MAX_PAGES]:
                        t = page.extract_text()
                        if t:
                            chunks.append(t)
                    text = "\n".join(chunks)
                except Exception as e:
                    logger.warning(f"pypdf failed: {e}")

        if not text or len(text) < 100:
            try:
                with open(tmp_path, "rb") as f:
                    raw = f.read(5 * 1024 * 1024)
                strings = re.findall(b'[\\x20-\\x7E]{5,}', raw)
                text = " ".join(s.decode("ascii", errors="ignore") for s in strings)
            except Exception as e:
                logger.warning(f"PDF raw extraction failed: {e}")

    elif fname.endswith(".docx"):
        if HAS_DOCX:
            try:
                doc = DocxDocument(tmp_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                paragraphs.append(cell.text)
                text = "\n".join(paragraphs)
            except Exception as e:
                logger.warning(f"docx extraction failed: {e}")

    elif fname.endswith(".doc"):
        try:
            import subprocess
            result = subprocess.run(["antiword", tmp_path], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                text = result.stdout
        except Exception:
            pass

        if not text or len(text) < 80:
            try:
                if HAS_DOCX:
                    doc = DocxDocument(tmp_path)
                    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception:
                pass

        if not text or len(text) < 80:
            try:
                with open(tmp_path, "rb") as f:
                    raw = f.read(3 * 1024 * 1024)
                for encoding in ("utf-16-le", "latin-1"):
                    try:
                        decoded = raw.decode(encoding, errors="ignore")
                        clean = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', decoded)
                        words = [w for w in clean.split() if len(w) > 1]
                        candidate = " ".join(words)
                        if len(candidate) > len(text):
                            text = candidate
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f".doc raw extraction failed: {e}")
    else:
        try:
            with open(tmp_path, "rb") as f:
                raw = f.read(2 * 1024 * 1024)
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            pass

    return text.strip()[:MAX_TEXT_CHARS]


def extract_text(file_storage):
    filename = file_storage.filename or "upload.pdf"
    ext = os.path.splitext(filename.lower())[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        return extract_text_from_path(tmp_path, filename)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


SINGLE_SYSTEM = """You are a senior HR consultant and career strategist at Direct Labour Consult.
Analyse the provided CV and return ONLY a valid JSON object — no markdown, no preamble.
JSON structure:
{
  "overall_score": <integer 0-100>,
  "market_readiness": "<Excellent|Strong|Developing|Needs Improvement>",
  "candidate_name": "<name from CV or 'Candidate'>",
  "executive_summary": "<2-3 sentence overall assessment>",
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "critical_improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "sections": {
    "first_impression":     {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]},
    "value_signal":         {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]},
    "evidence_of_impact":   {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]},
    "role_alignment":       {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]},
    "ats_compatibility":    {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]},
    "market_positioning":   {"score": <0-100>, "feedback": "<text>", "strengths": ["..."], "improvements": ["..."]}
  },
  "rewritten_section": {
    "section_name": "<e.g. Professional Summary>",
    "original":     "<original text from CV>",
    "rewritten":    "<professionally rewritten version>"
  }
}"""

BATCH_SYSTEM = """You are a senior HR consultant at Direct Labour Consult performing evidence-based candidate shortlisting.
You will receive a full job specification AND a candidate CV. Score the candidate STRICTLY against the job requirements.
Return ONLY a valid JSON object — no markdown, no preamble.

JSON structure:
{
  "candidate_name":    "<full name from CV or Candidate X>",
  "overall_score":     <integer 0-100 reflecting match to THIS specific job>,
  "market_readiness":  "<Excellent|Strong|Developing|Needs Improvement>",
  "recommendation":    "<Hire|Consider|Do Not Hire>",
  "executive_summary": "<2-3 sentence summary focused on fit for this specific role>",
  "job_fit_note":      "<1 sentence: how well this candidate matches the job advert>",
  "key_strengths":     ["<job-relevant strength 1>", "<strength 2>", "<strength 3>"],
  "key_concerns":      ["<concern vs job requirements 1>", "<concern 2>"],
  "skills_matched":    ["<required skill found in CV>"],
  "skills_missing":    ["<required skill NOT in CV>"],
  "experience_match":  "<Exceeds|Meets|Below|Unknown> requirements",
  "education_match":   "<Exceeds|Meets|Below|Unknown> requirements",
  "behavioural_risk":  "<Low|Medium|High>",
  "behavioural_notes": "<1-2 sentences on behavioural fit signals>",
  "sections": {
    "first_impression":   {"score": <0-100>, "feedback": "<text>"},
    "evidence_of_impact": {"score": <0-100>, "feedback": "<text>"},
    "role_alignment":     {"score": <0-100>, "feedback": "<text based on job spec>"},
    "ats_compatibility":  {"score": <0-100>, "feedback": "<text>"}
  }
}

RULES: Score against the SPECIFIC job. If disqualifying criteria match, set recommendation to Do Not Hire and score below 30. Be honest — recruiters need accuracy."""


def parse_json_response(raw):
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def call_claude(system, prompt, max_tokens=2000):
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "DLC CV Analyser",
        "status": "ok",
        "version": "v5",
        "timestamp": datetime.utcnow().isoformat() + "+00:00"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "cv" not in request.files:
            return jsonify({"success": False, "error": "No CV file provided"}), 400
        cv_file = request.files["cv"]
        job_title = request.form.get("job_title", "")
        cv_text = extract_text(cv_file)
        logger.info(f"Single CV: {len(cv_text)} chars")
        if not cv_text or len(cv_text) < 80:
            return jsonify({"success": False, "error": "Could not extract text. Please use a readable PDF or Word document."}), 400
        raw = call_claude(SINGLE_SYSTEM, f"JOB TITLE: {job_title}\n\nCV TEXT:\n{cv_text}", max_tokens=2000)
        result = parse_json_response(raw)
        gc.collect()
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    try:
        files = request.files.getlist("cvs[]")
        if not files:
            return jsonify({"success": False, "error": "No CV files provided"}), 400
        if len(files) > 25:
            return jsonify({"success": False, "error": "Maximum 25 CVs per batch"}), 400

        job_title = request.form.get("job_title", "")
        logger.info(f"Batch: {len(files)} CVs, role: '{job_title}'")

        # Extract all text first
        candidates = []
        for i, cv_file in enumerate(files):
            fname = cv_file.filename or f"cv_{i+1}.pdf"
            try:
                text = extract_text(cv_file)
                logger.info(f"  [{i+1}] {fname}: {len(text)} chars")
                candidates.append((i, text, fname))
            except Exception as e:
                logger.error(f"  [{i+1}] {fname}: {e}")
                candidates.append((i, "", fname))

        # Analyse each
        results = []
        for i, text, fname in candidates:
            clean_name = re.sub(r'\.(pdf|docx?|txt)$', '', fname, flags=re.IGNORECASE)
            if not text or len(text) < 80:
                results.append({
                    "candidate_name": clean_name,
                    "overall_score": 0,
                    "market_readiness": "Needs Improvement",
                    "recommendation": "Do Not Hire",
                    "executive_summary": "Could not extract text from this file.",
                    "key_strengths": [],
                    "key_concerns": ["File could not be read — may be scanned/image PDF or corrupted"],
                    "behavioural_risk": "Unknown",
                    "behavioural_notes": "Unable to assess.",
                    "sections": {},
                    "_file_index": i,
                    "_filename": fname
                })
            else:
                # Build rich job context prompt
                job_context = f"""JOB TITLE: {job_title or 'Not specified'}
DEPARTMENT: {request.form.get('job_dept', 'Not specified')}
MINIMUM EXPERIENCE: {request.form.get('job_exp', 'Any')} years
EDUCATION REQUIRED: {request.form.get('job_edu', 'Any')}
EMPLOYMENT TYPE: {request.form.get('job_type', 'Full-time')}
REQUIRED SKILLS: {request.form.get('job_skills', 'Not specified')}
DISQUALIFYING CRITERIA: {request.form.get('job_disq', 'None specified')}

JOB ADVERT / DESCRIPTION:
{request.form.get('job_desc', 'No detailed description provided.')}

SCORING WEIGHTS (1-10, higher = more important):
{request.form.get('weights', '{}')}
"""
                prompt = f"""{job_context}

CANDIDATE CV:
{text}"""
                try:
                    raw = call_claude(BATCH_SYSTEM, prompt, max_tokens=1500)
                    result = parse_json_response(raw)
                    result["_file_index"] = i
                    result["_filename"] = fname
                    results.append(result)
                except Exception as e:
                    logger.error(f"Claude failed for {fname}: {e}")
                    results.append({
                        "candidate_name": clean_name,
                        "overall_score": 0,
                        "market_readiness": "Needs Improvement",
                        "recommendation": "Do Not Hire",
                        "executive_summary": "Analysis failed for this candidate.",
                        "key_strengths": [],
                        "key_concerns": [f"AI analysis error: {str(e)[:80]}"],
                        "behavioural_risk": "Unknown",
                        "behavioural_notes": "Unable to assess.",
                        "sections": {},
                        "_file_index": i,
                        "_filename": fname
                    })
            gc.collect()

        results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank

        scored = [r.get("overall_score", 0) for r in results if r.get("overall_score", 0) > 0]
        hire_count = sum(1 for r in results if r.get("recommendation") == "Hire")
        consider_count = sum(1 for r in results if r.get("recommendation") == "Consider")

        return jsonify({
            "success": True,
            "job_title": job_title,
            "total_candidates": len(results),
            "summary": {
                "hire": hire_count,
                "consider": consider_count,
                "do_not_hire": len(results) - hire_count - consider_count,
                "average_score": round(sum(scored)/len(scored)) if scored else 0,
                "top_candidate": results[0].get("candidate_name") if results else None
            },
            "ranked_candidates": results
        })

    except Exception as e:
        logger.error(f"Batch error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "error": "File too large. Maximum 50MB per file."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
