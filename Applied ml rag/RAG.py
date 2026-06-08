import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

#config

st.set_page_config(
    page_title="Research Copilot",
    page_icon="🤖",
    layout="centered"
)

load_dotenv()

#gemini credentials
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

#embedding model

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

#chromadb

@st.cache_resource
def load_collection():

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    return chroma_client.get_collection(
        "research_papers"
    )

collection = load_collection()

#rag

def ask_rag(question):

    query_embedding = embedding_model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    context = "\n\n".join(
        results["documents"][0]
    )

    sources = list(
        {
            meta["source"]
            for meta in results["metadatas"][0]
        }
    )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Use only the provided context.

Context:
{context}

Question:
{question}

If the context is insufficient,
say so clearly.
"""
    )

    return response.text, sources

#stream effect

def stream_text(text):

    for word in text.split():
        yield word + " "

#ui

st.title("🤖 Research Paper Assistant")

st.markdown(
    "Ask questions about the uploaded research papers."
)

#chat history

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything about the research papers."
        }
    ]

# Display previous messages

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#input

prompt = st.chat_input(
    "Ask about the papers..."
)

if prompt:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message

    with st.chat_message("assistant"):

        with st.spinner("Searching papers..."):

            answer, sources = ask_rag(prompt)

        streamed_answer = st.write_stream(
            stream_text(answer)
        )

        st.markdown("---")

        st.caption(
            "📚 Sources: " +
            ", ".join(sources)
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": streamed_answer
        }
    )
