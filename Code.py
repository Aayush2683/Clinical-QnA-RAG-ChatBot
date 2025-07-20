import os
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1"
)

import re
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- CONFIG -----------------------------
MODEL_PATH = Path("./Llama-3.2-1B-Instruct-f16.gguf")  # put your model file here
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 40
LLAMA_CTX = 4096
DEFAULT_TOP_K = 5

SYSTEM_MSG = (
    "You are a concise clinical QA assistant. Use ONLY the provided context chunks. "
    "If the answer is not present, reply exactly: 'I cannot find that in the provided documents.' "
    "Cite chunk numbers like [1][3]. Do not hallucinate codes or criteria."
)

# ----------------------------- UTILITIES -----------------------------
def rough_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)   # coarse approximation for MiniLM

def pdf_to_text(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
    text = "\n".join(parts)
    text = re.sub(r'\u00ad', '', text)          # soft hyphens
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def chunk_document(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    buf, buf_tok = [], 0
    for p in paragraphs:
        ptok = rough_tokens(p)
        sents = re.split(r'(?<=[.!?])\s+', p) if ptok > CHUNK_TARGET_TOKENS * 1.4 else [p]
        for s in sents:
            stok = rough_tokens(s)
            if buf_tok + stok > CHUNK_TARGET_TOKENS and buf:
                chunks.append(" ".join(buf))
                if CHUNK_OVERLAP_TOKENS > 0:
                    overlap, ot = [], 0
                    for prev in reversed(buf):
                        vt = rough_tokens(prev)
                        if ot + vt <= CHUNK_OVERLAP_TOKENS:
                            overlap.insert(0, prev)
                            ot += vt
                        else:
                            break
                    buf = overlap
                    buf_tok = sum(rough_tokens(x) for x in buf)
                else:
                    buf, buf_tok = [], 0
            buf.append(s)
            buf_tok += stok
    if buf:
        chunks.append(" ".join(buf))
    return chunks

# ----------------------------- EMBEDDINGS / INDEX -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

def build_index_from_pdfs(pdf_paths: List[Path]):
    texts = []
    meta = []
    for pdf in pdf_paths:
        raw = pdf_to_text(pdf)
        chunks = chunk_document(raw)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            meta.append({
                "source": pdf.name,
                "chunk_id": f"{pdf.name}::chunk_{i}",
                "text": ch
            })
    if not texts:
        raise ValueError("No text extracted from uploaded PDFs.")
    model = get_embedder()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype("float32"))
    return index, meta, emb

def retrieve(query: str,
             index: faiss.Index,
             meta: List[Dict],
             top_k: int,
             embeddings: np.ndarray,
             mmr: bool,
             mmr_lambda: float = 0.65):
    model = get_embedder()
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    if not mmr:
        D, I = index.search(qv.astype('float32'), top_k)
        results = []
        for score, ix in zip(D[0], I[0]):
            if ix < 0: continue
            item = dict(meta[ix])
            item["score"] = float(score)
            results.append(item)
        return results, qv

    # MMR reranking
    sim_to_query = (embeddings @ qv.T).ravel()
    selected = []
    candidate_indices = list(range(len(meta)))
    while len(selected) < min(top_k, len(candidate_indices)):
        if not selected:
            best = int(np.argmax(sim_to_query[candidate_indices]))
            real = candidate_indices[best]
            selected.append(real)
            candidate_indices.remove(real)
            continue
        mmr_scores = []
        for c in candidate_indices:
            diversity = max(
                cosine_similarity(embeddings[c].reshape(1, -1),
                                  embeddings[s].reshape(1, -1))[0][0]
                for s in selected
            )
            score = mmr_lambda * sim_to_query[c] - (1 - mmr_lambda) * diversity
            mmr_scores.append((score, c))
        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        chosen = mmr_scores[0][1]
        selected.append(chosen)
        candidate_indices.remove(chosen)
    results = []
    for ix in selected:
        item = dict(meta[ix])
        item["score"] = float(sim_to_query[ix])
        results.append(item)
    return results, qv

# ----------------------------- LLM -----------------------------
@st.cache_resource(show_spinner=True)
def load_llm(model_path: str, ctx: int):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Place your GGUF file there."
        )
    return Llama(
        model_path=model_path,
        n_ctx=ctx,
        n_threads=int(os.getenv("LLAMA_THREADS", "8")),
        n_batch=128,
        n_gpu_layers=int(os.getenv("LLAMA_GPU_LAYERS", "0")),
        verbose=False
    )

def build_prompt(question: str, retrieved: List[Dict]) -> str:
    ctx_blocks = []
    for i, r in enumerate(retrieved, 1):
        ctx_blocks.append(f"[{i}] Source: {r['source']}\n{r['text']}")
    context = "\n\n".join(ctx_blocks)
    return f"""System: {SYSTEM_MSG}

Context:
{context}

User Question: {question}

Answer:"""

def generate_answer(llm: Llama, prompt: str, temperature: float = 0.2, max_tokens: int = 512):
    out = llm(prompt, temperature=temperature, max_tokens=max_tokens,
              stop=["</s>", "User:", "System:"])
    return out["choices"][0]["text"].strip()

# ----------------------------- STREAMLIT UI -----------------------------
st.set_page_config(page_title="Clinical RAG (Upload First)", layout="wide")
st.title("🩺 Aayush's Clinical RAG – Upload PDFs First (No Preloaded Data)")
st.caption("Upload your clinical guideline PDFs, build an in-memory vector index, then ask questions. Nothing is stored persistently.")

# State containers
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "emb" not in st.session_state:
    st.session_state.emb = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# Sidebar Controls
with st.sidebar:
    st.subheader("Settings")
    top_k = st.number_input("Top-K Retrieval", 1, 20, DEFAULT_TOP_K)
    use_mmr = st.checkbox("Use MMR (diversity reranking)", value=False)
    mmr_lambda = st.slider("MMR λ (relevance↔diversity)", 0.1, 0.9, 0.65, 0.05)
    temp = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max Answer Tokens", 64, 2048, 512, 32)
    st.markdown("---")
    clear_btn = st.button("🗑 Clear Index / Reset Session")
    if clear_btn:
        for key in ["index", "meta", "emb", "llm"]:
            st.session_state[key] = None
        st.success("Session cleared. Upload new PDFs.")

uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

build_col, sample_col = st.columns([1,1])

with build_col:
    build_clicked = st.button("📚 Build / Rebuild Index", type="primary")

with sample_col:
    run_samples = st.button("▶ Run Sample Assignment Queries", disabled=st.session_state.index is None)

if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF before building.")
    else:
        pdf_save_dir = Path("uploaded_pdfs")
        pdf_save_dir.mkdir(exist_ok=True)
        pdf_paths = []
        for uf in uploaded_files:
            dest = pdf_save_dir / uf.name
            dest.write_bytes(uf.read())
            pdf_paths.append(dest)
        with st.spinner("Extracting, chunking, embedding..."):
            index, meta, emb = build_index_from_pdfs(pdf_paths)
            st.session_state.index = index
            st.session_state.meta = meta
            st.session_state.emb = emb
            st.session_state.llm = load_llm(str(MODEL_PATH), LLAMA_CTX)
        st.success(f"Indexed {len(meta)} chunks from {len(pdf_paths)} PDF file(s).")

if run_samples:
    if st.session_state.index is None:
        st.error("Index not built yet.")
    else:
        sample_queries = [
            'Give me the correct coded classification for the following diagnosis: "Recurrent depressive disorder, currently in remission"'
        ]
        for q in sample_queries:
            retrieved, _ = retrieve(q, st.session_state.index, st.session_state.meta,
                                    top_k, st.session_state.emb, use_mmr, mmr_lambda)
            prompt = build_prompt(q, retrieved)
            ans = generate_answer(st.session_state.llm, prompt,
                                  temperature=temp, max_tokens=max_tokens)
            with st.expander(f"Q: {q}"):
                st.write(ans)
                st.caption("Retrieved Chunks:")
                for i, r in enumerate(retrieved, 1):
                    st.markdown(f"**[{i}] score={r['score']:.4f}** `{r['source']}`\n\n{r['text'][:1200]}")

st.markdown("---")
query = st.text_area("Ask a clinical question (after building index):", height=160)

answer_btn = st.button("🔎 Get Answer")

if answer_btn:
    if st.session_state.index is None:
        st.error("Index not built yet. Upload PDFs and click build first.")
    elif not query.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Retrieving & generating..."):
            retrieved, _ = retrieve(query, st.session_state.index, st.session_state.meta,
                                    top_k, st.session_state.emb, use_mmr, mmr_lambda)
            prompt = build_prompt(query, retrieved)
            ans = generate_answer(st.session_state.llm, prompt,
                                  temperature=temp, max_tokens=max_tokens)
        st.subheader("Answer")
        st.write(ans)
        with st.expander("Retrieved Chunks"):
            for i, r in enumerate(retrieved, 1):
                st.markdown(f"**[{i}] score={r['score']:.4f}** `{r['source']}`\n\n{r['text'][:1500]}")
        if not retrieved:
            st.info("No chunks retrieved — index might be empty or PDFs lacked extractable text.")

st.markdown("---")
st.caption("Prototype only – not a diagnostic or treatment tool.")