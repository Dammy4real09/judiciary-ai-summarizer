import os
import re
import requests
import pdfplumber
import docx
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set.")

HF_HEADERS = {}
if HF_TOKEN:
    HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

LEGALBERT_API_URL = (
    "https://api-inference.huggingface.co/models/lawal-Dare/legal-bert-nigeria"
)
LLM_API_URL = (
    "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
)


def extract_text_from_docx(file_path):
    document = docx.Document(file_path)
    return "\n".join([p.text for p in document.paragraphs])


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def score_sentences(sentences):
    payload = {"inputs": sentences}

    try:
        response = requests.post(
            LEGALBERT_API_URL, headers=HF_HEADERS, json=payload, timeout=60
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
        print("LegalBERT Error:", e)
        return [0.5] * len(sentences)


def reformat_with_llm(extracted_text):

    prompt = f"""
You are a Nigerian judicial legal assistant.

Rewrite the extracted content clearly into:

1. Facts of the Case
2. Issues for Determination
3. Court's Reasoning
4. Final Decision / Orders

Maintain formal tone.
Do not invent facts.

Extracted Content:
{extracted_text}
"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.2},
    }

    try:
        response = requests.post(
            LLM_API_URL, headers=HF_HEADERS, json=payload, timeout=120
        )

        response.raise_for_status()
        result = response.json()

        return result[0]["generated_text"]

    except Exception as e:
        print("LLM Error:", e)
        return "LLM restructuring failed."


def generate_hybrid_summary(text):

    sentences = split_into_sentences(text)

    if not sentences:
        return "Insufficient text."

    scores = score_sentences(sentences)

    scored = list(zip(sentences, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    top_sentences = [s for s, score in scored[:20]]
    top_sentences.sort(key=lambda s: text.find(s))

    extracted_text = " ".join(top_sentences)

    return reformat_with_llm(extracted_text)


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
            return jsonify({"decision": "No input provided."})

        summary = generate_hybrid_summary(text)

        return jsonify({"decision": summary})

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
