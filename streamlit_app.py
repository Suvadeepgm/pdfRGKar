import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64


load_dotenv()

# Configure LLM settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token="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",
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Directories
PERSIST_DIR = "./db"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Static PDF configuration
STATIC_PDF_FILE = "WBSP070037082024_36_2025-01-20.pdf"  # Replace with your actual PDF file name
STATIC_PDF_PATH = STATIC_PDF_FILE 

# Ensure the static PDF is in place
def ensure_static_pdf():
    if not os.path.exists(STATIC_PDF_PATH):
        raise FileNotFoundError(
            f"Static PDF file '{STATIC_PDF_FILE}' is missing in the 'data' directory. "
            "Please place the file there before running the app."
        )

# PDF Display Function
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Data Ingestion for Static PDF
def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# Query Handler
def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named PdfMadeEasy, created by Suvadeep. Your main goal is to give answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document. Always give a reference from the document. 
            Context:
            {context_str}
            Question:
            {query_str}
            """
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)

    if hasattr(answer, "response"):
        return answer.response
    elif isinstance(answer, dict) and "response" in answer:
        return answer["response"]
    else:
        return "Sorry, I couldn't find an answer."

# Streamlit App Initialization
st.title("RG Kar ChatBot")

# Ensure the static PDF is present and load it
ensure_static_pdf()

# Display the static PDF
st.sidebar.title("Static Document:")
st.sidebar.write("Currently loaded document:")
st.sidebar.write(STATIC_PDF_FILE)
display_pdf(STATIC_PDF_PATH)

# Ingest static PDF data if not already processed
if not os.listdir(PERSIST_DIR):  # Check if persistence directory is empty
    with st.spinner("Processing static document..."):
        data_ingestion()
        st.success("Static document processed successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about the static PDF document."}]

user_prompt = st.chat_input("Ask me anything about the document:")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
