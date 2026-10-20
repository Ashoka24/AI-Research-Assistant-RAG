from pathlib import Path

import streamlit as st

from src.rag_assistant import ResearchAssistant

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "documents"

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
st.title("Research Assistant")
st.write("Search a local collection of research notes and inspect the passages that support each result.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of passages", 1, 8, 4)
    st.caption("Add permitted .txt, .md, or text-based .pdf files to data/documents/.")
    if st.button("Rebuild index"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource
def load_assistant():
    return ResearchAssistant(DOCS_DIR)

assistant = load_assistant()
st.metric("Indexed passages", len(assistant.chunks))
question = st.text_area("Question", placeholder="What does the document collection say about retrieval evaluation?")
if st.button("Search", type="primary", disabled=not question.strip()):
    results = assistant.retrieve(question, top_k=top_k)
    if not results:
        st.info("No matching passages found. Try different wording or add relevant documents.")
    for item in results:
        with st.expander(f"[{item.source_id}] {item.source} · chunk {item.chunk_id}"):
            st.caption(f"Lexical relevance score: {item.score:.3f}; this is not a confidence score.")
            st.write(item.text)
st.caption("This is a local lexical retrieval demo, not a fact-checking system. Verify claims against original sources.")
