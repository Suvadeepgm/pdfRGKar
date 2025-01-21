import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import os
import base64

# Load environment variables
load_dotenv()

# Qdrant configuration
QDRANT_URL = "https://2a47a3e8-5374-4b45-a52e-9d25f20dadef.us-west-1-0.aws.cloud.qdrant.io:6333"  # Replace with your Qdrant URL
QDRANT_API_KEY = "KmwYx5NSfpGxQeHj6bBBb-SfTvJdH7MEJISnoWEROuoh599MoA7ZCw"  # Replace with your Qdrant API key
QDRANT_COLLECTION_NAME = "RGKar"  # Collection name

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Configure LLM settings
llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=10000,
    token="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",
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Ensure Qdrant collection exists
if not qdrant_client.get_collection(QDRANT_COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=embed_model.embedding_dim, distance=Distance.COSINE),
    )

# PDF Display Function
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Data Ingestion for Selected PDF
def data_ingestion(selected_pdf):
    filepath = os.path.join(".", selected_pdf)
    documents = SimpleDirectoryReader(filepath).load_data()
    storage_context = StorageContext.from_qdrant(
        qdrant_client=qdrant_client, collection_name=QDRANT_COLLECTION_NAME
    )
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index

# Query Handler
def handle_query(query, index):
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant named Rg Kar Bot, created by Suvadeep. Your main goal is to give answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document. Always give a reference from the document. 
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

# Static PDFs
STATIC_PDFS = {
    "Document 1": "WBSP070037082024_36_2025-01-20.pdf",
    "Document 2": "WBSP070037082024_35_2025-01-20.pdf",
}

# Sidebar for Document Selection
st.sidebar.title("Select Document:")
selected_doc_name = st.sidebar.selectbox("Choose a document:", list(STATIC_PDFS.keys()))
selected_pdf = STATIC_PDFS[selected_doc_name]

# Display the selected PDF
st.sidebar.write("Currently loaded document:")
st.sidebar.write(selected_doc_name)
display_pdf(os.path.join(".", selected_pdf))

# Ingest selected PDF data
with st.spinner("Processing the selected document..."):
    index = data_ingestion(selected_pdf)
    st.success(f"{selected_doc_name} processed successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Hello! Ask me anything about {selected_doc_name}."}]

user_prompt = st.chat_input("Ask me anything about the document:")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    response = handle_query(user_prompt, index)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
