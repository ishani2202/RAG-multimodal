# 🔬 RAG-Powered Research Paper Retrieval & Summarization

> *Semantic search meets LLM-powered workflows. Stop drowning in PDFs, start building.*

A two-stage RAG system that finds relevant research papers and generates actionable research workflows from natural language queries.

## 🎯 What It Does

Query: *"I want to build a biomedical NER model using BERT"*

**Stage 1**: Finds top 10 most semantically relevant papers from your collection  
**Stage 2**: Extracts key sections and generates a step-by-step implementation workflow

## 🏗️ How It Works

```
Stage 1: Query → LLM (synthetic abstract) → FAISS search → Top papers
Stage 2: PDFs → Text extraction → Chunking → Retrieval → LLM workflow
```

**Key Innovation**: LLM-generated synthetic abstracts align your casual query with academic paper language, dramatically improving retrieval quality.

## 📦 Installation

```bash
# Clone and install
git clone https://github.com/yourusername/rag-research-assistant.git
cd rag-research-assistant
pip install -r requirements.txt

# Pull LLM (requires Ollama)
ollama pull deepseek-r1:7b
```

**Dependencies**: `numpy`, `faiss-cpu`, `sentence-transformers`, `langchain-ollama`, `pymupdf`

## 🚀 Usage

### Stage 1: Find Papers

1. Create `metadata.json`:
```json
[
  {
    "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf",
    "summary": "BERT: Pre-training of Deep Bidirectional Transformers..."
  }
]
```

2. Run: `python Stage1.py`

**Output**: Top 10 papers ranked by semantic similarity

### Stage 2: Generate Workflow

1. Add PDFs to `PDF_data/` folder
2. Edit query in `stage2_text.py`
3. Run: `python stage2_text.py`

**Output**: 
- `extracted_pdf_data.json`: Full PDF text
- `generated_workflow.txt`: Step-by-step methodology

## 🔧 Configuration

**Change LLM**:
```python
ollama_llm = OllamaLLM(model="llama3")  # Any Ollama model
```

**Adjust retrieval**:
```python
index.search(query_embedding, k=10)  # Change k for more/fewer results
```

**Modify chunking**:
```python
chunk_text(text, chunk_size=525, overlap=250)  # Tune for your domain
```

## 🧠 Technical Details

- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
- **Search**: FAISS IndexFlatIP (Stage 1) / IndexFlatL2 (Stage 2)
- **LLM**: DeepSeek-R1 7B via Ollama (local inference)
- **Chunking**: 525 words with 250-word overlap



## 🗺️ Future Improvements

- [ ] Web interface (Gradio)
- [ ] Incremental indexing
- [ ] Hybrid BM25 + dense retrieval
- [ ] Citation graph integration
- [ ] Automated metadata extraction


## 📜 License

MIT License

---

*Built with FAISS, Sentence-Transformers, LangChain, and Ollama. Fueled by coffee and deadline panic.*
