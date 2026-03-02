# Swiggy Annual Report — RAG-Based Question Answering System

## About the Project

Every year, companies like Swiggy publish detailed annual reports — hundreds of pages packed with financial data, strategic priorities, governance details, and operational metrics. Sifting through all of that to find one specific number or insight is time-consuming and tedious.

This project solves that problem. It is a **Retrieval-Augmented Generation (RAG)** application that lets you ask plain English questions about the **Swiggy Annual Report (FY 2023-24)** and receive accurate, context-grounded answers in seconds — with full source traceability so you can verify every claim.

---

## Document Source

| Detail    | Info |
|-----------|------|
| **Report** | Swiggy Limited — Annual Report FY 2023-24 |
| **Format** | PDF |
| **Public Link** | [Swiggy Annual Report (C:\Users\udayw\NEWEL\data\swiggy_annual_report.pdf) |

> The report was sourced from publicly available regulatory filings on BSE India.

---

## Architecture

The system follows a classic RAG pipeline — load, chunk, embed, store, retrieve, generate:

```
                        ┌─────────────────────┐
                        │   Swiggy Annual      │
                        │   Report (PDF)       │
                        └──────────┬──────────┘
                                   │
                          1. Load & Parse
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Text Preprocessing  │
                        │  & Chunking          │
                        │  (LangChain)         │
                        └──────────┬──────────┘
                                   │
                       2. Generate Embeddings
                          (BAAI/bge-small-en-v1.5)
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   FAISS Vector       │
                        │   Store (Persisted)  │
                        └──────────┬──────────┘
                                   │
            User Question ────► 3. Semantic Search (Top-K)
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   LLM Generation     │
                        │   (Google Gemini)     │
                        │   Context-Only Mode  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Gradio Web UI      │
                        │   Answer + Sources   │
                        └─────────────────────┘
```

### How the pipeline works, step by step

1. **Document Loading** — The Swiggy Annual Report PDF is loaded using LangChain's `PyPDFLoader`, which extracts text page-by-page while retaining page number metadata.
2. **Cleaning & Chunking** — Raw text is cleaned (whitespace normalization, header/footer removal) and split into overlapping chunks of ~1,000 characters using `RecursiveCharacterTextSplitter`. Overlap ensures that no information is lost at chunk boundaries.
3. **Embedding** — Each chunk is converted into a dense 384-dimensional vector using the **BAAI/bge-small-en-v1.5** sentence embedding model from HuggingFace. These embeddings capture the semantic meaning of each chunk.
4. **Indexing** — All vectors are stored in a **FAISS** (Facebook AI Similarity Search) index on disk, enabling millisecond-level similarity lookups even across thousands of chunks.
5. **Retrieval** — When a user asks a question, it's embedded with the same model and the top-5 most semantically similar chunks are retrieved from FAISS.
6. **Generation** — The retrieved chunks are injected into a strict prompt template and passed to **Google Gemini 2.5 Flash**, which generates an answer using *only* the provided context. If the answer isn't in the context, the model explicitly says so.

---

## Tech Stack

| Layer              | Technology                                | Role |
|--------------------|-------------------------------------------|------|
| **Orchestration**  | LangChain                                 | Ties document loading, chunking, retrieval, and LLM calls together |
| **Embeddings**     | BAAI/bge-small-en-v1.5 (HuggingFace)     | Converts text into dense vectors for semantic search |
| **Vector Store**   | FAISS (Facebook AI Similarity Search)     | Stores and searches embeddings locally on disk |
| **LLM**           | Google Gemini 2.5 Flash                   | Generates final answers from retrieved context |
| **PDF Parsing**    | PyPDF                                     | Extracts text from the annual report PDF |
| **Web Interface**  | Gradio                                    | Provides a clean, interactive chat-style UI |
| **Language**       | Python 3.10+                              | Everything runs in Python |

---

## Project Structure

```
NEWEL/
│
├── data/
│   └── swiggy_annual_report.pdf        # The Swiggy Annual Report PDF
│
├── vectorstore/                         # Auto-generated FAISS index (persisted to disk)
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # Central configuration (model names, chunk size, API keys)
│   ├── document_processor.py            # PDF loading → text cleaning → chunking
│   ├── vector_store.py                  # Embedding generation, FAISS indexing, similarity search
│   └── rag_pipeline.py                  # Prompt construction + LLM call (Gemini)
│
├── app.py                               # Gradio web app entry point
├── requirements.txt                     # All Python dependencies
├── .env.example                         # Template for environment variables
├── .gitignore
└── README.md
```

---

## Local Setup — Step by Step

### Prerequisites

- **Python 3.10 or higher** installed on your machine
- A **Google Gemini API key** (free tier works)
- The Swiggy Annual Report PDF (link above, or use the one in `data/`)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/NEWEL.git
cd NEWEL
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

> This will install LangChain, FAISS, sentence-transformers, Gradio, and all other required libraries. The first run will also download the BAAI embedding model (~90 MB) automatically.

### 4. Get a Google Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **"Create API Key"** → select **"Create in new project"**
4. Copy the key

### 5. Configure your environment

```bash
copy .env.example .env          # Windows
cp .env.example .env            # macOS / Linux
```

Open the `.env` file and paste your key:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 6. Place the PDF

If you haven't already, download the Swiggy Annual Report and place it at:

```
data/swiggy_annual_report.pdf
```

### 7. Run the application

```bash
python app.py
```

The Gradio interface will launch and open in your browser at **http://localhost:7860**.

- Click **"Initialize System"** — this processes the PDF, generates embeddings, and builds the FAISS index. Takes about 1–2 minutes on the first run; subsequent starts load the cached index instantly.
- Type any question about the Swiggy Annual Report and hit **Ask**.

---

## Sample Questions You Can Try

| Question | What it tests |
|----------|---------------|
| What is Swiggy's total revenue for FY 2023-24? | Financial data retrieval |
| Who are the members of Swiggy's Board of Directors? | Governance information |
| What are Swiggy's key business segments? | Strategic overview |
| What is Swiggy's employee count? | Operational data |
| What were the major risk factors mentioned? | Risk analysis |
| What is Swiggy's CSR expenditure? | Compliance data |
| Who is the CEO of Swiggy? | Leadership details |

---

## Configuration

All tunable parameters live in `src/config.py`:

| Parameter         | Default                  | What it controls |
|-------------------|--------------------------|------------------|
| `CHUNK_SIZE`      | 1,000 characters         | Size of each text chunk |
| `CHUNK_OVERLAP`   | 200 characters           | Overlap between consecutive chunks |
| `TOP_K`           | 5                        | Number of chunks retrieved per query |
| `TEMPERATURE`     | 0.1                      | LLM creativity — lower is more factual |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`      | HuggingFace embedding model |
| `GEMINI_MODEL`    | `gemini-2.5-flash`       | Google Gemini model variant |

---

## Anti-Hallucination Design

This system takes hallucination prevention seriously:

- **Strict system prompt** — The LLM is explicitly instructed to answer *only* from the provided document context, never from its training data.
- **Graceful "I don't know"** — If the retrieved context doesn't contain the answer, the model responds with *"I don't have enough information in the Swiggy Annual Report to answer this question"* instead of guessing.
- **Source transparency** — Every answer comes with the exact supporting chunks from the report, including page numbers, so you can verify everything yourself.

---

## License

This project was built for educational and assignment purposes only. The Swiggy Annual Report is a publicly available document sourced from regulatory filings.
