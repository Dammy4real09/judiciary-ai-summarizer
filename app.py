import os
import re
import requests
import pdfplumber
import docx
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set.")

HEADERS = {}
if HF_TOKEN:
    HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

LEGALBERT_API = (
    "https://api-inference.huggingface.co/models/lawal-Dare/legal-bert-nigeria"
)
LLM_API = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"

# ===============================
# FILE EXTRACTION
# ===============================


def extract_docx(path):
    document = docx.Document(path)
    return "\n".join([p.text for p in document.paragraphs])


def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# ===============================
# TEXT UTILITIES
# ===============================


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


# ===============================
# LEGALBERT SCORING
# ===============================


def score_sentences(sentences):
    payload = {"inputs": sentences}

    try:
        response = requests.post(
            LEGALBERT_API, headers=HEADERS, json=payload, timeout=60
        )
        response.raise_for_status()
        results = response.json()

        scores = []

        for item in results:
            for label in item:
                if "1" in label["label"] or "important" in label["label"].lower():
                    scores.append(label["score"])
                    break
            else:
                scores.append(item[0]["score"])

        return scores

    except Exception as e:
        print("LegalBERT Error:", e)
        return [0.5] * len(sentences)


# ===============================
# LLM REFORMATTING
# ===============================


def reformat_with_llm(text):

    prompt = f"""
You are a Nigerian judicial assistant.

Restructure the following extracted judgment content into:

1. Facts of the Case
2. Issues for Determination
3. Court's Reasoning
4. Final Decision / Orders

Maintain formal tone.
Do not invent facts.

Extracted Content:
{text}
"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.2},
    }

    try:
        response = requests.post(LLM_API, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"]

    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ LLM restructuring failed."


# ===============================
# HYBRID PIPELINE
# ===============================


def generate_summary(text):

    sentences = split_sentences(text)

    if not sentences:
        return "Insufficient text."

    scores = score_sentences(sentences)

    ranked = list(zip(sentences, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_sentences = [s for s, _ in ranked[:20]]
    top_sentences.sort(key=lambda s: text.find(s))

    extracted = " ".join(top_sentences)

    return reformat_with_llm(extracted)


# ===============================
# ROUTES
# ===============================


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        text = request.form.get("text", "")

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            if file.filename.endswith(".pdf"):
                text = extract_pdf(path)
            elif file.filename.endswith(".docx"):
                text = extract_docx(path)

        if not text.strip():
            return jsonify({"decision": "No input provided."})

        summary = generate_summary(text)

        return jsonify({"decision": summary})

    return render_template("index.html")


# ===============================
# LOCAL RUN
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
