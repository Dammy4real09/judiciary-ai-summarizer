import os
import pdfplumber
import docx
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set.")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

HF_MODEL = (
    "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta"
)


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
# SUMMARY GENERATION
# ===============================


def generate_summary(text):

    prompt = f"""
You are a Nigerian judicial legal assistant.

Restructure the following court judgment into clearly labeled sections:

Facts of the Case:
Issues for Determination:
Court's Reasoning:
Final Decision / Orders:

Maintain formal legal tone.
Do not invent facts.
Clearly number the final orders.
If monetary awards are present, state them clearly.

Judgment:
{text[:4000]}
"""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a precise judicial summarization assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "parameters": {"max_new_tokens": 500, "temperature": 0.2},
        "options": {"wait_for_model": True},
    }

    try:
        response = requests.post(HF_MODEL, headers=HEADERS, json=payload, timeout=180)

        response.raise_for_status()
        result = response.json()

        # Zephyr router response format
        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        return "Unexpected response format from model."

    except Exception as e:
        print("HF Router Error:", e)
        return "LLM restructuring failed."


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
