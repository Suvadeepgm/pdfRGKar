import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64

# Load environment variables
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

# Directories and file configuration
PERSIST_DIR = "./db"  # For storing index files
STATIC_PDF_FILE = "WBSP070037082024_36_2025-01-20.pdf"  # Static PDF file in the root directory
STATIC_PDF_PATH = STATIC_PDF_FILE  # Path is just the file name since it's in the root directory

# Ensure static PDF exists
def ensure_static_pdf():
    if not os.path.exists(STATIC_PDF_PATH):
        raise FileNotFoundError(
            f"Static PDF file '{STATIC_PDF_FILE}' is missing in the root directory. "
            "Please place the file in the root before running the app."
        )

# PDF Display Function
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Data Ingestion for Static PDF
def data_ingestion():
    documents = SimpleDirectoryReader(".").load_data()  # Load files from the current directory
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
            """You are a Q&A assistant named PdfMadeEasy, created by Suvadeep. Your main goal is to give answers as accurately as possible, based on the instructions and context you have been given. If asked about the accused, the accused is same as the person given punishment. Just explain things from the pdf. You don't have to go before or after the timeline. Always give a reference from the document. 
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
st.sidebar.write("Judgement Copy Loaded.")
#st.sidebar.write(STATIC_PDF_FILE)
#display_pdf(STATIC_PDF_PATH)

# Ingest static PDF data if not already processed
if not os.listdir(PERSIST_DIR):  # Check if persistence directory is empty
    with st.spinner("Processing document..."):
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
