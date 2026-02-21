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

    # ---------------------------
    # 1. Extract Final Decision
    # ---------------------------

    # ---------------------------
# Improved Decision Extraction
# ---------------------------

decision_section = ""

# Work only on the last 30% of the document
cut_index = int(len(text) * 0.7)
end_part = text[cut_index:]

trigger_phrases = [
    "i hereby order",
    "i accordingly order",
    "it is hereby ordered",
    "judgment is hereby entered",
    "the defendant shall pay",
    "the claimant is entitled",
    "the court awards"
]

lower_end = end_part.lower()

for phrase in trigger_phrases:
    if phrase in lower_end:
        start = lower_end.find(phrase)
        decision_section = end_part[start:]
        break

# Clean formatting
if decision_section:
    decision_section = decision_section.replace("\n", "<br>")
else:
    decision_section = "Final decision section could not be clearly isolated."
    # ---------------------------
    # 2. Split Into Paragraph Blocks
    # ---------------------------

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]

    if not paragraphs:
        return {"facts": "", "reasoning": "", "decision": decision_section}

    # ---------------------------
    # 3. Positional Logic
    # ---------------------------

    total = len(paragraphs)

    # First 30% → Background / Facts
    facts_block = paragraphs[: max(1, int(total * 0.3))]

    # Middle 50% → Court Analysis
    reasoning_block = paragraphs[int(total * 0.3) : int(total * 0.8)]

    # ---------------------------
    # 4. Importance Filtering Within Sections
    # ---------------------------

    facts_sentences = split_into_sentences(" ".join(facts_block))
    reasoning_sentences = split_into_sentences(" ".join(reasoning_block))

    # Score separately
    facts_scores = score_sentences(facts_sentences)
    reasoning_scores = score_sentences(reasoning_sentences)

    facts_scored = list(zip(facts_sentences, facts_scores))
    reasoning_scored = list(zip(reasoning_sentences, reasoning_scores))

    facts_scored.sort(key=lambda x: x[1], reverse=True)
    reasoning_scored.sort(key=lambda x: x[1], reverse=True)

    top_facts = [s for s, score in facts_scored[:5]]
    top_reasoning = [s for s, score in reasoning_scored[:8]]

    # Restore order
    top_facts.sort(key=lambda s: text.find(s))
    top_reasoning.sort(key=lambda s: text.find(s))

    facts_text = " ".join(top_facts)
    reasoning_text = " ".join(top_reasoning)

    if decision_section:
        decision_section = decision_section.replace("\n", "<br>")
    else:
        decision_section = "Final decision section could not be automatically isolated."

    return {
        "facts": facts_text,
        "reasoning": reasoning_text,
        "decision": decision_section,
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
