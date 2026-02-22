import os
import re
import requests
import pdfplumber
import docx
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HUGGING FACE CONFIGURATION
# ===============================

HF_API_URL = "https://api-inference.huggingface.co/models/lawal-Dare/legal-bert-nigeria"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set.")

HF_HEADERS = {}
if HF_TOKEN:
    HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ===============================
# FILE EXTRACTION
# ===============================


def extract_text_from_docx(file_path):
    document = docx.Document(file_path)
    return "\n".join([p.text for p in document.paragraphs])


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ===============================
# TEXT PROCESSING
# ===============================


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


# ===============================
# BATCH SCORING
# ===============================


def score_sentences(sentences):

    payload = {"inputs": sentences}

    try:
        response = requests.post(
            HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60
        )

        response.raise_for_status()
        result = response.json()

        scores = []

        for item in result:
            for label in item:
                if "1" in label["label"] or "important" in label["label"].lower():
                    scores.append(label["score"])
                    break
            else:
                scores.append(item[0]["score"])

        return scores

    except Exception as e:
        print("HF API Error:", e)
        return [0.5] * len(sentences)


# ===============================
# STRUCTURED SUMMARY GENERATOR
# ===============================


def generate_structured_summary(text):

    # Normalize text
    text = text.replace("\r", "")

    # ---------------------------
    # 1. Detect Section Headings
    # ---------------------------

    sections = {"facts": "", "issues": "", "analysis": "", "decision": ""}

    lower_text = text.lower()

    # Try splitting by known headings
    headings = {
        "facts": ["facts", "background", "case background"],
        "issues": ["issues for determination", "issues"],
        "analysis": ["analysis", "court's analysis", "consideration"],
        "decision": ["decision", "holding", "orders", "conclusion"],
    }

    for key, patterns in headings.items():
        for pattern in patterns:
            if pattern in lower_text:
                start = lower_text.find(pattern)
                sections[key] = text[start : start + 3000]
                break

    # ---------------------------
    # 2. If No Headings, Use Positional Logic
    # ---------------------------

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]

    total = len(paragraphs)

    if not sections["facts"]:
        sections["facts"] = " ".join(paragraphs[: max(1, int(total * 0.25))])

    if not sections["analysis"]:
        sections["analysis"] = " ".join(
            paragraphs[int(total * 0.25) : int(total * 0.75)]
        )

    if not sections["decision"]:
        sections["decision"] = " ".join(paragraphs[int(total * 0.75) :])

    # ---------------------------
    # 3. Clean Decision Section
    # ---------------------------

    trigger_phrases = [
        "i hereby order",
        "i accordingly order",
        "it is hereby ordered",
        "the defendant shall",
        "the claimant is entitled",
        "judgment is entered",
    ]

    decision_text = sections["decision"]
    lower_decision = decision_text.lower()

    for phrase in trigger_phrases:
        if phrase in lower_decision:
            start = lower_decision.find(phrase)
            sections["decision"] = decision_text[start:]
            break

    sections["decision"] = sections["decision"].replace("\n", "<br>")

    return {
        "facts": sections["facts"][:1500],
        "reasoning": sections["analysis"][:2000],
        "decision": sections["decision"],
    }


# ===============================
# MAIN ROUTE
# ===============================


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        text = request.form.get("text", "")

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)

            elif file.filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)

        if not text.strip():
            return jsonify(
                {"facts": "No input provided.", "reasoning": "", "decision": ""}
            )

        summary = generate_structured_summary(text)

        return jsonify(summary)

    return render_template("index.html")


# ===============================
# LOCAL RUN
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
