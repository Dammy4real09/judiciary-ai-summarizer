import os
import pdfplumber
import docx
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def extract_docx(path):
    document = docx.Document(path)
    return "\n".join([p.text for p in document.paragraphs])

def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def generate_summary(text):

    prompt = f"""
You are a Nigerian judicial legal assistant.

Restructure the following judgment into:

1. Facts of the Case
2. Issues for Determination
3. Court's Reasoning
4. Final Decision / Orders

Maintain formal judicial tone.
Do not invent facts.
If monetary awards exist, clearly state them.
Number final orders where appropriate.

Judgment Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise judicial summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)