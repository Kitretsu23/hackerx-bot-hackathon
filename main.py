import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import uuid
import pypdf #remove this later
import fitz

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# --- IMPORTANT ---
# Set your Google API key as an environment variable named 'GOOGLE_API_KEY'
# For example, in your terminal:
# export GOOGLE_API_KEY="YOUR_API_KEY"
# Or, for development, you can uncomment the line below and paste your key.
# IMPORTANT: Do not commit your API key to version control.
# os.environ['GOOGLE_API_KEY'] = "YOUR_GEMINI_API_KEY_HERE"

model = None
try:
    # It's recommended to use environment variables for API keys.
    api_key = "AIzaSyBjNedPpiPy-D1_W699PrX5bd-n8BVerc8"
    if not api_key:
        logging.warning(
            "GOOGLE_API_KEY environment variable not set. The API will not work without it."
        )
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")  # Using a faster model with low temperature for consistent, factual responses
        logging.info("Google Generative AI configured.")

except Exception as e:
    logging.error(f"Could not configure Google Generative AI: {e}")


def download_file(url, save_path):
    """Downloads a file from a URL and saves it locally."""
    try:
        logging.info(f"Downloading document from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("this is edited")
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Document saved to {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file: {e}")
        return False


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF (fast and reliable)."""
    try:
        logging.info(f"Extracting text from {pdf_path} using PyMuPDF")
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        logging.info("Successfully extracted text using PyMuPDF.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text with PyMuPDF: {e}")
        return None


@app.route("/api/v1/hackrx/run", methods=["POST"])
def process_document():
    if not model:
        return jsonify(
            {"error": "Generative model not configured. Please check API key."}
        ), 500

    data = request.get_json()
    if not data or "documents" not in data or "questions" not in data:
        return jsonify(
            {"error": "Invalid request body. 'documents' and 'questions' are required."}
        ), 400

    doc_url = data["documents"]
    questions = data["questions"]

    # Generate a unique filename to avoid conflicts
    pdf_path = f"temp_document_{uuid.uuid4()}.pdf"

    if not download_file(doc_url, pdf_path):
        return jsonify({"error": "Failed to download the document."}), 500

    try:
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            return jsonify({"error": "Failed to extract text from the document."}), 500

        question_list_str = "\n".join([f"- {q}" for q in questions])
        logging.info(question_list_str)
        prompt = f"""You are a legal-aware AI assistant specializing in insurance document analysis. Your task is to carefully analyze the provided insurance document and answer each question with the highest level of accuracy and precision.

CRITICAL GUIDELINES:
1.  **Information Source**: Use ONLY information explicitly stated in the provided document. Do not rely on external knowledge or assumptions.
2.  **Answer Quality**:
    * Provide exact details and keep each answer under 100 words while maintaining completeness.
    * If the question is a Yes or No question, your answer should start with "Yes" or "No" and then provide an explanation for why the answer is Yes or No.
3.  **Handling Missing Information**:
    * If a direct answer is not found, try to understand the meaning of the question and return the closest answer while also informing the user about the absence of the information.
    * If the answer is NOT found in the document, clearly state: "This information is not explicitly stated in the provided document."
    * Suggest what specific sections or topics might contain the missing information.
4. **Citation Requirements**:
    * ALWAYS cite the specific section, article number, clause number, or page reference for every piece of information provided.

5.  **Document Content**:
---
{document_text}
---

5.  **Questions to Answer**:
{question_list_str}

6.  **Return the result strictly in this JSON format (no extra commentary or formatting):
```json
{{
  "answers": [
    "Answer to question 1",
    "Answer to question 2"
  ]
}}```
"""



        response = model.generate_content(prompt)

        # Clean the response to ensure it is a valid JSON string
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        answers_json = json.loads(response_text)

        if "answers" not in answers_json or not isinstance(
            answers_json["answers"], list
        ):
            logging.error(
                f"Model response did not have the expected format: {response_text}"
            )
            raise ValueError("Model response is not in the expected format.")

        logging.info("Successfully generated answers from the document.")
        logging.info(answers_json)
        return jsonify(answers_json)

    except Exception as e:
        logging.error(f"An error occurred with the generative model: {e}")
        return jsonify(
            {"error": f"An error occurred with the generative model: {str(e)}"}
        ), 500
    finally:
        # Clean up the local file
        try:
            os.remove(pdf_path)
            logging.info(f"Removed temporary local file: {pdf_path}")
        except OSError as e:
            logging.error(f"Error removing temporary file {pdf_path}: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
