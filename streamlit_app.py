import streamlit as st
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient

# Set your Hugging Face API token
hf_token = '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'

# Initialize the Hugging Face Inference Client
client = InferenceClient(
    "google/gemma-1.1-7b-it",
    token=hf_token,
)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_response(prompt):
    """Generate response from Hugging Face's Inference Client."""
    messages = [{"role": "user", "content": prompt}]
    response = ""
    for message in client.chat_completion(messages=messages, max_tokens=500, stream=True):
        response += message.choices[0].delta.content
    return response.strip()

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Extracting text from the PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", pdf_text, height=300)

    if pdf_text:
        st.write("You can now chat with the PDF content.")
        user_query = st.text_input("Ask a question about the PDF content:")

        if user_query:
            st.write("Generating response...")
            chat_prompt = f"Based on the following PDF content, {pdf_text[:2000]}... Answer the following question: {user_query}"
            response = generate_response(chat_prompt)
            st.write(response)
