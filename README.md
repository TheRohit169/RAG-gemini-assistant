# GenAI Support Assistant with RAG

A production-style AI-powered chat assistant that answers company support questions using Retrieval-Augmented Generation (RAG). Built with FastAPI, FAISS, sentence-transformers, and Google Gemini 1.5 Flash.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/CSS/JS)                │
│   Chat UI  →  fetch("/api/chat")  ←  JSON Response          │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP POST
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Backend                            │
│                                                              │
│  /api/chat  →  RAGService                                    │
│                   │                                          │
│                   ├── 1. RetrievalService                    │
│                   │      ├── EmbeddingService (MiniLM)       │
│                   │      └── FAISSStore (cosine similarity)  │
│                   │                                          │
│                   ├── 2. MemoryService (session history)     │
│                   │                                          │
│                   └── 3. LLMService (Gemini 1.5 Flash)      │
└─────────────────────────────────────────────────────────────┘
```

---

## RAG Workflow

```
User Question
     │
     ▼
Generate Query Embedding (all-MiniLM-L6-v2)
     │
     ▼
FAISS Cosine Similarity Search (top 3 chunks)
     │
     ▼
Similarity Threshold Check (≥ 0.70)
     │
  ┌──┴──────────────────┐
  │ Below threshold      │ Above threshold
  ▼                      ▼
Fallback message    Build RAG Prompt
                         │
                         ▼
                  Inject: Context + History + Question
                         │
                         ▼
                  Gemini 1.5 Flash (temp=0.2)
                         │
                         ▼
                  Grounded Response → User
```

---

## Project Structure

```
project/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, startup indexing
│   │   ├── routes/
│   │   │   └── chat.py              # /api/chat and /health endpoints
│   │   ├── services/
│   │   │   ├── embedding_service.py # sentence-transformers wrapper
│   │   │   ├── retrieval_service.py # FAISS query + threshold filter
│   │   │   ├── llm_service.py       # Gemini API wrapper
│   │   │   ├── rag_service.py       # Pipeline orchestrator
│   │   │   └── memory_service.py    # Session-based conversation memory
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   ├── vectorstore/
│   │   │   └── faiss_store.py       # FAISS IndexFlatIP wrapper
│   │   └── utils/
│   │       ├── chunking.py          # Word-based overlapping chunker
│   │       └── logger.py            # Centralised logging
│   ├── docs.json                    # 10-document knowledge base
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## Embedding Strategy

- **Model**: `all-MiniLM-L6-v2` from sentence-transformers
- **Dimension**: 384
- **Why MiniLM**: Fast, lightweight (80 MB), strong semantic quality, runs on CPU
- **Process**: Every document is chunked → each chunk is embedded independently → stored in FAISS

---

## Similarity Search

- **Algorithm**: Cosine similarity via FAISS `IndexFlatIP`
- **Normalisation**: Vectors are L2-normalised before indexing; inner product then equals cosine similarity
- **Top-K**: Returns top 3 most similar chunks
- **Threshold**: 0.70 — queries scoring below this return a safe fallback response
- **Why cosine?**: Scale-invariant, measures directional similarity in semantic space

---

## Prompt Design

The RAG prompt is intentionally constrained:

```
You are a helpful AI assistant...

Use ONLY the provided context to answer.
If the answer is not present in the context,
say you do not have enough information.

Context:
{retrieved_context}    ← real document chunks with source labels

Conversation History:
{history}              ← last 5 user/assistant pairs

User Question:
{question}
```

**Why low temperature (0.2)?** Produces deterministic, factual answers rather than creative hallucinations.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Clone & install

```bash
git clone https://github.com/yourusername/genai-rag-assistant.git
cd genai-rag-assistant/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The startup process will:
1. Load `docs.json`
2. Chunk all documents
3. Generate embeddings (MiniLM model downloads on first run ~80 MB)
4. Build FAISS index
5. Start serving on `http://localhost:8000`

### 4. Open the frontend

Open `frontend/index.html` in your browser — or serve it:

```bash
cd ../frontend
python -m http.server 3000
# Visit http://localhost:3000
```

---

## API Usage

### POST /api/chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"test123","message":"How do I reset my password?"}'
```

Response:
```json
{
  "reply": "To reset your password, navigate to Settings > Security > Change Password...",
  "tokensUsed": 312,
  "retrievedChunks": 2
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "healthy"}
```

---

## Deployment

### Backend → Render

1. Push code to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable: `GEMINI_API_KEY=your_key`

### Frontend → Netlify

1. Drag and drop the `frontend/` folder to [netlify.com/drop](https://app.netlify.com/drop)
2. Update `API_BASE` in `script.js` to your Render backend URL

---

## Future Improvements

- [ ] Persistent FAISS index (save/load to disk)
- [ ] JWT authentication
- [ ] Support uploading new documents via API
- [ ] Markdown rendering in chat UI
- [ ] Conversation history persistence (SQLite)
- [ ] Multi-document cross-referencing
- [ ] Streaming responses (SSE)
- [ ] Admin panel to manage knowledge base
