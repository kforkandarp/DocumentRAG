import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()
# loads your .env file so GROQ_API_KEY is available via os.environ.get()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DocChat — Chat with your PDFs",
    page_icon="📄",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# Same dark theme as the YouTube RAG app
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Overall background */
.stApp { background-color: #0f0f0f; color: #f0f0f0; font-family: 'DM Sans', sans-serif; }

/* Title */
h1 { font-family: 'Space Mono', monospace !important; color: #f0f0f0 !important; letter-spacing: -1px; }

/* Chat message bubbles — user */
.user-bubble {
    background: #1e3a5f;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
}

/* Chat message bubbles — assistant */
.bot-bubble {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.95rem;
}

/* Source chunk box */
.source-box {
    background: #111;
    border-left: 3px solid #e63946;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    font-size: 0.8rem;
    color: #aaa;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

/* Status badges */
.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; font-family: 'Space Mono', monospace; }
.badge-green { background: #1a3a2a; color: #4ade80; border: 1px solid #166534; }
.badge-red   { background: #3a1a1a; color: #f87171; border: 1px solid #991b1b; }
.badge-blue  { background: #1a2a3a; color: #60a5fa; border: 1px solid #1d4ed8; }

/* File uploader */
.stFileUploader { background: #1a1a1a !important; border: 1px dashed #333 !important; border-radius: 8px !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED: load the LLM once and reuse across reruns
# same as YouTube RAG — prevents reloading on every interaction
# ============================================================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", "paste_your_key_here"),
        temperature=0.2
        # low temperature = more factual, less random — important for document Q&A
    )


# ============================================================
# PROCESS PDF
# This replaces the entire YouTube transcript fetching section
# from the previous app — everything else is identical
# ============================================================
@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes: bytes, filename: str):
    """
    Loads a PDF, chunks it, embeds it, builds FAISS retriever.
    file_bytes is the raw PDF content.
    filename is used as the cache key — same file won't be reprocessed.
    Returns (retriever, page_count, preview_text)
    """

    # ----------- SAVE PDF TO TEMP FILE -----------
    # PyPDFLoader needs a file path, not raw bytes
    # so we save the uploaded file to a temporary location first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    # tmp_path is now the path to the saved PDF on disk

    # ----------- LOAD PDF -----------
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    # pages is a list of LangChain Document objects, one per PDF page
    # each Document has .page_content (text) and .metadata (page number etc.)

    # clean up the temp file after loading
    os.unlink(tmp_path)
    # os.unlink deletes a file — we don't need it anymore after loading

    if not pages:
        raise ValueError("Could not extract any text from this PDF. It may be scanned or image-based.")
    # some PDFs are just scanned images with no actual text — catch that early

    # ----------- CHUNKING -----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    # same as YouTube RAG — overlapping chunks preserve context at boundaries

    documents = splitter.split_documents(pages)
    # split_documents preserves the metadata (page numbers) from each page
    # this is better than create_documents because we keep page number info
    # which lets us show "this answer came from page 3" in the UI

    # ----------- EMBEDDINGS -----------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # converts each chunk into a semantic vector — same model as YouTube RAG

    # ----------- VECTOR STORE -----------
    vector_store = FAISS.from_documents(documents, embeddings)
    # stores all vectors in FAISS for fast similarity search — same as before

    # ----------- RETRIEVER -----------
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
        # return top 4 most relevant chunks for each question
    )

    preview = pages[0].page_content[:400]
    # grab first 400 characters of page 1 as a preview for the UI

    return retriever, len(pages), preview


# ============================================================
# PROMPT TEMPLATE
# Same clean prompt as the Groq version of YouTube RAG
# ============================================================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions about documents.
Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know based on the provided document."
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""
)
# "Always mention which part" encourages the model to cite sources naturally


# ============================================================
# RAG PIPELINE
# Identical to YouTube RAG — retriever + prompt + LLM
# ============================================================
def rag_pipeline(query: str, retriever, llm) -> tuple[str, list]:
    docs = retriever.invoke(query)
    # retrieves top-4 most semantically similar chunks

    context = "\n\n".join(doc.page_content for doc in docs)
    # merges chunks into one context string for the LLM

    if not context.strip():
        return "I don't know based on the provided document.", []
    # graceful fallback if retrieval returns nothing useful

    final_prompt = prompt.format(context=context, question=query)
    # fills in {context} and {question} placeholders

    response = llm.invoke(final_prompt)
    # sends to LLaMA 3.3 70B via Groq and gets answer back

    return response.content, docs
    # .content extracts just the text from the ChatGroq response object


# ============================================================
# SESSION STATE INIT
# Persists values across Streamlit reruns
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    # stores all messages so chat history survives reruns

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    # stores the FAISS retriever once a PDF is processed

if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None
    # tracks which PDF is currently loaded


# ============================================================
# UI — HEADER
# ============================================================
st.markdown("# 📄 DocChat")
st.markdown("<p style='color:#666; margin-top:-12px; font-family:Space Mono,monospace; font-size:0.8rem;'>chat with any PDF document</p>", unsafe_allow_html=True)
st.divider()


# ============================================================
# UI — FILE UPLOADER
# st.file_uploader renders a drag-and-drop file upload widget
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    # only allow PDF files — blocks other file types automatically
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # check if this is a new file or the same one already loaded
    if uploaded_file.name != st.session_state.loaded_filename:
        # new file uploaded — process it
        with st.spinner("Reading and indexing your PDF..."):
            try:
                file_bytes = uploaded_file.read()
                # .read() gets the raw bytes of the uploaded file

                retriever, page_count, preview = process_pdf(file_bytes, uploaded_file.name)
                # process_pdf is cached by filename — same file won't re-embed

                st.session_state.retriever = retriever
                st.session_state.loaded_filename = uploaded_file.name
                st.session_state.chat_history = []
                # clear chat history when a new document is loaded

                st.markdown(f'<span class="badge badge-green">✅ {uploaded_file.name} loaded — {page_count} pages</span>', unsafe_allow_html=True)
                # show success badge with filename and page count

                with st.expander("📄 Document preview"):
                    st.caption(preview + "...")
                # collapsible preview of the first 400 chars so user knows it loaded correctly

            except ValueError as e:
                st.markdown(f'<span class="badge badge-red">❌ {e}</span>', unsafe_allow_html=True)
    else:
        # same file already loaded — no need to reprocess
        st.markdown(f'<span class="badge badge-blue">ℹ️ {uploaded_file.name} already loaded</span>', unsafe_allow_html=True)


# ============================================================
# UI — CHAT AREA
# Only shown once a PDF is successfully loaded
# ============================================================
if st.session_state.retriever:

    st.divider()

    # render all past messages from history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            # show source chunks the answer was built from
            if msg.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for i, doc in enumerate(msg["sources"]):
                        page_num = doc.metadata.get("page", "?") + 1
                        # +1 because PDF pages are 0-indexed internally but humans count from 1
                        st.markdown(f'<div class="source-box"><b>Page {page_num}</b><br>{doc.page_content[:300]}...</div>', unsafe_allow_html=True)
                        # shows page number instead of just "Chunk N" — more useful for documents

    st.divider()

    # chat input pinned to bottom of screen
    user_input = st.chat_input("Ask something about the document...")

    if user_input:
        # immediately render user message before model responds
        st.markdown(f'<div class="user-bubble">🧑 {user_input}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            llm = load_llm()
            answer, sources = rag_pipeline(user_input, st.session_state.retriever, llm)
        # runs RAG pipeline and gets answer + source chunks back

        st.markdown(f'<div class="bot-bubble">🤖 {answer}</div>', unsafe_allow_html=True)

        if sources:
            with st.expander("📎 Sources used", expanded=False):
                for i, doc in enumerate(sources):
                    page_num = doc.metadata.get("page", "?") + 1
                    st.markdown(f'<div class="source-box"><b>Page {page_num}</b><br>{doc.page_content[:300]}...</div>', unsafe_allow_html=True)

        # save both messages to history so they persist across reruns
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

else:
    # placeholder shown before any PDF is uploaded
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color: #444;'>
        <div style='font-size: 3rem;'>📄</div>
        <p style='font-family: Space Mono, monospace; font-size: 0.85rem; margin-top: 12px;'>
            upload a pdf above<br>then ask anything about it
        </p>
    </div>
    """, unsafe_allow_html=True)