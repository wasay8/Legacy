
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import csv
import sys
import time

# Setup
csv.field_size_limit(sys.maxsize)
load_dotenv()

st.set_page_config(page_title="Mental Health LLM Assistant", layout="wide")
st.title("🧠 Mental Health Counseling Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"rag_chat_history": [], "llm_chat_history": []}
else:
    if "rag_chat_history" not in st.session_state.chat_history:
        st.session_state.chat_history["rag_chat_history"] = []
    if "llm_chat_history" not in st.session_state.chat_history:
        st.session_state.chat_history["llm_chat_history"] = []

# === Embedding function ===
def create_vector_embedding_with_metadata(counsel_data_path, kaggle_data_path, save_path):
    counsel_loader = CSVLoader(file_path=counsel_data_path)
    counsel_data = counsel_loader.load()

    kaggle_loader = CSVLoader(file_path=kaggle_data_path)
    kaggle_data = kaggle_loader.load()

    processed_counsel_docs = []
    for document in counsel_data:
        content = document.page_content
        lines = content.split("\n")

        question_text, answer_text = "", ""
        for line in lines:
            if line.startswith("questionText:"):
                question_text = line.replace("questionText:", "").strip()
            elif line.startswith("answerText:"):
                answer_text = line.replace("answerText:", "").strip()

        full_content = f"Question: {question_text}\nAnswer: {answer_text}"

        meta = {
            "questionTitle": "",
            "therapistName": "",
            "topics": ""
        }

        for line in lines:
            if line.startswith("questionTitle:"):
                meta["questionTitle"] = line.replace("questionTitle:", "").strip()
            elif line.startswith("therapistName:"):
                meta["therapistName"] = line.replace("therapistName:", "").strip()
            elif line.startswith("topics:"):
                meta["topics"] = line.replace("topics:", "").strip()

        processed_counsel_docs.append(Document(page_content=full_content, metadata=meta))

    all_docs = processed_counsel_docs + kaggle_data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY_2"])
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

# === Sidebar for dataset upload ===
with st.sidebar:
    st.header("📁 Create Vector Database")

    # CSV and vector paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(project_root, '..', 'data')
    vectorstore_folder = os.path.join(project_root, '..', 'vectorstore')

    counsel_path = os.path.join(data_folder, 'counselchat-data.csv')
    kaggle_path = os.path.join(data_folder, 'train.csv')

    if st.button("🔍 Load & Embed Data"):
        with st.spinner("Processing and embedding... (1-2 minutes)"):
            if os.path.exists(os.path.join(vectorstore_folder, "index.faiss")):
                st.info("🔁 Loading existing vector database...")
                embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY_2"])
                vectors = FAISS.load_local(vectorstore_folder, embeddings, allow_dangerous_deserialization=True)
            else:
                st.info("🛠️ Creating new vector database...")
                os.makedirs(vectorstore_folder, exist_ok=True)
                vectors = create_vector_embedding_with_metadata(counsel_path, kaggle_path, vectorstore_folder)

            st.session_state["vectors"] = vectors
        st.success("✅ Vector database is ready!")

# === Query LLM with RAG ===
def query_llm(user_prompt, vectors):
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY_2"])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a mental health counselor AI assistant. Use the following context to answer the user's question as helpfully and empathetically as possible.

        Context:
        {context}

        Question: {question}
        """
    )

    retriever = vectors.as_retriever()
    retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    start = time.process_time()
    response = retriever_chain({"query": user_prompt})
    duration = time.process_time() - start

    return response["result"], response["source_documents"], duration

# === Direct Suggestion (No RAG) ===
def direct_llm_suggestion(user_input):
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY_2"])
    prompt = f"""
    You are a mental health counseling assistant. A user is seeking guidance on how to best support a patient. 

    Situation:
    {user_input}

    Provide a helpful, ethical, and empathetic suggestion for the counselor.
    """
    response = llm.invoke(prompt)
    return response.content

# === Interaction mode ===
st.subheader("Choose Interaction Mode")
interaction_mode = st.radio(
    "Select how you'd like to interact with the assistant:",
    ["🔍 Search Database", "💡 Ask LLM for a Suggestion"],
    horizontal=True
)

# === Clear chat and display histories based on mode ===
if interaction_mode == "🔍 Search Database":
    st.markdown("### 🗂️  Chat with Database")
    for entry in st.session_state.chat_history["rag_chat_history"]:
        with st.chat_message("user"):
            st.markdown(entry["user"])
        with st.chat_message("assistant"):
            st.markdown(entry["assistant"])

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history["rag_chat_history"] = []
            st.rerun()

    if "vectors" not in st.session_state:
        st.warning("📥 Please load and embed your datasets from the sidebar to use database search.")
    else:
        user_prompt = st.chat_input("Ask a question based on counseling data:")
        if user_prompt:
            with st.spinner("Querying..."):
                answer, docs, duration = query_llm(user_prompt, st.session_state["vectors"])
            st.session_state.chat_history["rag_chat_history"].append({
                "user": user_prompt,
                "assistant": answer
            })
            st.rerun()

elif interaction_mode == "💡 Ask LLM for a Suggestion":
    st.markdown("### 💬 Chat with LLM")
    for entry in st.session_state.chat_history["llm_chat_history"]:
        with st.chat_message("user"):
            st.markdown(entry["user"])
        with st.chat_message("assistant"):
            st.markdown(entry["assistant"])

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history["llm_chat_history"] = []
            st.rerun()

    user_input = st.chat_input("Describe the situation for a suggestion:")
    if user_input:
        with st.spinner("Thinking..."):
            suggestion = direct_llm_suggestion(user_input)
        st.session_state.chat_history["llm_chat_history"].append({
            "user": user_input,
            "assistant": suggestion
        })
        st.rerun()
