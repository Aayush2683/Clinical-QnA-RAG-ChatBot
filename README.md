# Clinical-QnA-RAG-ChatBot – Explanatory Note

## 1. Tools & Models Used

| Layer | Choice | Rationale (Time-boxed 4h scope) |
|-------|--------|---------------------------------|
| **Language Model** | `Llama-3.2-1B-Instruct-f16.gguf` (local via `llama-cpp-python`) | Small footprint → fast cold start & inference on CPU; no external API latency or PHI exposure. |
| **Inference Backend** | `llama-cpp-python` | Simple Python API, supports quantized GGUF, adjustable context window & threads. |
| **Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers) | 384-dim vectors; strong speed/quality trade-off; quick download; low RAM. |
| **Vector Store / ANN** | FAISS `IndexFlatIP` | With normalized embeddings, inner product = cosine; trivial, robust, no tuning. |
| **PDF Parsing** | `pdfplumber` | Retains most clinical text structure; handles multi-line & hyphenation. |
| **Web UI** | Streamlit | Rapid upload + interaction layer; minimal boilerplate. |
| **Misc** | `numpy`, `sklearn` (MMR cosine), `tqdm`, `faiss-cpu` | Standard performant Python ML tooling. |

> *Excluded deliberately:* Ontology linking, rerank cross-encoders, hybrid BM25, heavy guardrail frameworks (time & complexity trade-off).

---

## 2. Use of AI Tools (ChatGPT / Copilot) – Disclosure

| Component / Activity | AI Assistance? | Nature of Assistance | Manual Validation |
|----------------------|---------------|----------------------|-------------------|
| Project skeleton & config fields | Yes (ChatGPT) | Brainstormed minimal clean layout | Adjusted & pruned |
| Chunking strategy (paragraph + sentence fallback) | Partial | Prompted for ideas; adapted logic | Inspected chunk lengths |
| MMR pseudo-code reminder | Yes | Retrieved standard MMR formula | Rewrote, simplified |
| System prompt wording & safety fallback | Yes | Drafted phrasing variants | Final wording tightened |
| README polish | Yes | Style suggestions | Trimmed & edited |
| llama-cpp parameter sanity | Yes | Confirmed essential args | Cross-checked locally |

**Not AI-generated:** Core retrieval/index build code, error checks, guardrails for “no index built”.

**Reason for disclosure:** Transparency (evaluation criterion: responsible AI tool usage).

---

## 3. Design Decisions & Assumptions

### 3.1 Retrieval
- Dense semantic retrieval only (MiniLM + FAISS).  
- Normalized embeddings + `IndexFlatIP` → cosine similarity without extra distance transforms.  
- Default `top_k = 5` (adjustable) balances breadth vs context length.

### 3.2 Chunking
- Paragraph-first to preserve clinical criteria blocks.
- Oversized paragraphs split by sentence if > ~1.4× target tokens.
- Target ≈350 tokens (rough: words * 1.3) + 40-token overlap to reduce boundary loss.

### 3.3 Prompt & Guardrails
- System message forbids fabrication & mandates exact fallback:  
  **`I cannot find that in the provided documents.`**
- Requires citing chunk numbers `[n]` to encourage grounding.

### 3.4 LLM Choice
- 1B instruct GGUF → quick local inference; stays within 4h scope.
- 4096 context window sufficient for prompt + up to ~8 chunks.

### 3.5 MMR (Optional)
- Toggleable; default off for simplicity.
- λ = 0.65 (mild diversity without sacrificing primary relevance).

### 3.6 Session Model
- **Empty start:** No preloaded corpus; user must upload PDFs → explicit provenance.
- In-memory only (not persisted) keeps state clean & reproducible.

### 3.7 Performance Assumptions
- Expected corpus small (few PDFs) → Flat index acceptable (O(N) scan still fast).
- Embedding latency dominates first run; subsequent queries sub-second (depends on hardware).

### 3.8 Privacy & Security
- Entire pipeline local; no external network calls post-install.
- Assumes uploaded PDFs are de-identified; still not a diagnostic system.

### 3.9 Limitations

| Area | Limitation | Impact |
|------|------------|--------|
| Code/criteria specificity | No ontology (ICD/DSM) mapping layer | May paraphrase; won’t auto-supply missing codes |
| Hallucination mitigation | Prompt-only (no verifier) | Subtle hallucinations still possible |
| Evaluation metrics | No automated QA scoring | Manual review required |
| Long/tabular PDFs | Simple chunker (no table reconstruction) | Possible context fragmentation |
| Multilingual support | English-tuned embeddings | Non-English accuracy may drop |

### 3.10 Time Management Trade-offs

| Implemented | Deferred |
|-------------|---------|
| Streamlit upload UI | Ontology code normalizer |
| Dense retrieval + MMR | Hybrid BM25 fusion |
| Chunk overlap | Cross-encoder reranker |
| Citation pattern | Automatic grounding score |
| Local llama-cpp | Persistent disk index |

### 3.11 Future Extensions
- ICD/DSM code regex + mapping table.
- Automatic answer grounding / faithfulness score.
- Adaptive chunk size trimming to fit context precisely.
- Source text highlighting in UI.
- Hybrid sparse+dense retrieval.

---

## 4. Interview / Oral Summary (One-Liner)

> *“I built a lean local RAG: PDF upload → paragraph/sentence chunking with overlap → MiniLM embeddings + FAISS cosine → optional MMR rerank → small 1B LLaMA instruct model with a strict no-hallucination prompt and explicit fallback. Everything starts empty to guarantee data provenance.”*

---

## 5. Disclaimer

Prototype for educational / evaluation use only. **Not** a clinical decision support or diagnostic tool.
