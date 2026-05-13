import os
import json
import logging
import tempfile
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

# PDF / DOCX extraction
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

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
CORS(app, origins=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text(file_storage):
    """Extract plain text from PDF or DOCX with multiple fallback methods."""
    filename = (file_storage.filename or "").lower()
    ext = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    text = ""
    try:
        if filename.endswith(".pdf"):
            # Method 1: pdfplumber (best for text PDFs)
            if HAS_PDF:
                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        pages_text = []
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                pages_text.append(t)
                        text = "\n".join(pages_text)
                except Exception as e:
                    logger.warning(f"pdfplumber failed: {e}")

            # Method 2: pypdf fallback
            if (not text or len(text) < 100) and HAS_PYPDF:
                try:
                    reader = pypdf.PdfReader(tmp_path)
                    pages_text = []
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            pages_text.append(t)
                    text = "\n".join(pages_text)
                except Exception as e:
                    logger.warning(f"pypdf failed: {e}")

            # Method 3: raw byte extraction (last resort for any PDF)
            if not text or len(text) < 100:
                try:
                    with open(tmp_path, "rb") as f:
                        raw = f.read()
                    # Extract readable ASCII strings from binary
                    import re as _re
                    strings = _re.findall(b'[\\x20-\\x7E]{4,}', raw)
                    text = " ".join(s.decode("ascii", errors="ignore") for s in strings)
                except Exception as e:
                    logger.warning(f"Raw extraction failed: {e}")

        elif filename.endswith((".docx", ".doc")) and HAS_DOCX:
            try:
                doc = DocxDocument(tmp_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                # Also extract from tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                paragraphs.append(cell.text)
                text = "\n".join(paragraphs)
            except Exception as e:
                logger.warning(f"docx extraction failed: {e}")

        else:
            with open(tmp_path, "rb") as f:
                raw = f.read()
            text = raw.decode("utf-8", errors="ignore")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return text.strip()


# ─────────────────────────────────────────────
# SINGLE CV ANALYSIS  (/analyze)
# ─────────────────────────────────────────────

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


def analyse_single(cv_text, job_title=""):
    prompt = f"JOB TITLE (if provided): {job_title}\n\nCV TEXT:\n{cv_text[:12000]}"
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SINGLE_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ─────────────────────────────────────────────
# BATCH ANALYSIS  (/analyze-batch)
# ─────────────────────────────────────────────

BATCH_SYSTEM = """You are a senior HR consultant at Direct Labour Consult performing candidate shortlisting.
Analyse the provided CV and return ONLY a valid JSON object — no markdown, no preamble.

JSON structure:
{
  "candidate_name":    "<full name from CV or 'Candidate X'>",
  "overall_score":     <integer 0-100>,
  "market_readiness":  "<Excellent|Strong|Developing|Needs Improvement>",
  "recommendation":    "<Hire|Consider|Do Not Hire>",
  "executive_summary": "<2-3 sentence summary of this candidate>",
  "key_strengths":     ["<strength 1>", "<strength 2>", "<strength 3>"],
  "key_concerns":      ["<concern 1>", "<concern 2>"],
  "behavioural_risk":  "<Low|Medium|High>",
  "behavioural_notes": "<1-2 sentences on any behavioural/cultural fit risk signals>",
  "sections": {
    "first_impression":   {"score": <0-100>, "feedback": "<text>"},
    "evidence_of_impact": {"score": <0-100>, "feedback": "<text>"},
    "role_alignment":     {"score": <0-100>, "feedback": "<text>"},
    "ats_compatibility":  {"score": <0-100>, "feedback": "<text>"}
  }
}"""


def analyse_candidate(cv_text, job_title, index):
    prompt = f"ROLE BEING RECRUITED FOR: {job_title or 'Not specified'}\n\nCANDIDATE CV:\n{cv_text[:10000]}"
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=BATCH_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        result["_file_index"] = index
        return result
    except Exception as e:
        logger.error(f"Candidate {index} analysis failed: {e}")
        return {
            "candidate_name": f"Candidate {index + 1}",
            "overall_score": 0,
            "market_readiness": "Needs Improvement",
            "recommendation": "Do Not Hire",
            "executive_summary": "Analysis failed for this candidate.",
            "key_strengths": [],
            "key_concerns": ["Could not process this CV file."],
            "behavioural_risk": "Unknown",
            "behavioural_notes": "Unable to assess.",
            "sections": {},
            "_file_index": index,
            "_error": str(e)
        }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "service": "DLC CV Analyser",
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "+00:00"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Single CV analysis — existing endpoint, unchanged behaviour."""
    try:
        if "cv" not in request.files:
            return jsonify({"success": False, "error": "No CV file provided"}), 400

        cv_file = request.files["cv"]
        job_title = request.form.get("job_title", "")

        cv_text = extract_text(cv_file)
        if not cv_text or len(cv_text) < 100:
            return jsonify({"success": False, "error": "Could not extract text from CV. Please ensure it is a readable PDF or Word document."}), 400

        result = analyse_single(cv_text, job_title)
        return jsonify({"success": True, "data": result})

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({"success": False, "error": "AI returned an unexpected format. Please try again."}), 500
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({"success": False, "error": "Analysis service error. Please try again."}), 500


@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    """
    Batch CV analysis for recruiter flow.
    Accepts: multiple CV files (field name 'cvs[]') + optional 'job_title'
    Returns: ranked list of candidates with scores, recommendations, behavioural risk.
    """
    try:
        files = request.files.getlist("cvs[]")
        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No CV files provided"}), 400
        if len(files) > 25:
            return jsonify({"success": False, "error": "Maximum 25 CVs per batch"}), 400

        job_title = request.form.get("job_title", "")
        logger.info(f"Batch analysis: {len(files)} CVs, role: '{job_title}'")

        results = []
        for i, cv_file in enumerate(files):
            try:
                cv_text = extract_text(cv_file)
                if not cv_text or len(cv_text) < 80:
                    result = {
                        "candidate_name": cv_file.filename or f"Candidate {i+1}",
                        "overall_score": 0,
                        "market_readiness": "Needs Improvement",
                        "recommendation": "Do Not Hire",
                        "executive_summary": "Could not extract text from this file.",
                        "key_strengths": [],
                        "key_concerns": ["File could not be read — may be scanned/image PDF"],
                        "behavioural_risk": "Unknown",
                        "behavioural_notes": "Unable to assess.",
                        "sections": {},
                        "_file_index": i,
                        "_filename": cv_file.filename
                    }
                else:
                    result = analyse_candidate(cv_text, job_title, i)
                    result["_filename"] = cv_file.filename
            except Exception as e:
                logger.error(f"File {i} ({cv_file.filename}) failed: {e}")
                result = {
                    "candidate_name": cv_file.filename or f"Candidate {i+1}",
                    "overall_score": 0,
                    "market_readiness": "Needs Improvement",
                    "recommendation": "Do Not Hire",
                    "executive_summary": "Processing error for this candidate.",
                    "key_strengths": [],
                    "key_concerns": ["Processing error"],
                    "behavioural_risk": "Unknown",
                    "behavioural_notes": "Unable to assess.",
                    "sections": {},
                    "_file_index": i,
                    "_filename": cv_file.filename,
                    "_error": str(e)
                }
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)

        # Add rank
        for rank, r in enumerate(results, 1):
            r["rank"] = rank

        # Summary stats
        hire_count = sum(1 for r in results if r.get("recommendation") == "Hire")
        consider_count = sum(1 for r in results if r.get("recommendation") == "Consider")
        avg_score = round(sum(r.get("overall_score", 0) for r in results) / len(results)) if results else 0

        return jsonify({
            "success": True,
            "job_title": job_title,
            "total_candidates": len(results),
            "summary": {
                "hire": hire_count,
                "consider": consider_count,
                "do_not_hire": len(results) - hire_count - consider_count,
                "average_score": avg_score,
                "top_candidate": results[0].get("candidate_name") if results else None
            },
            "ranked_candidates": results
        })

    except Exception as e:
        logger.error(f"Batch analyze error: {e}")
        return jsonify({"success": False, "error": f"Batch analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
