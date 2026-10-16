# Retrieval-Augmented Generation (RAG) Chatbot

A Python project that indexes local documents, retrieves text relevant to a question, and asks a Groq-hosted language model to produce a concise answer from that text. The repository contains a FastAPI service, a FAISS-backed retrieval pipeline, example documents, and notebooks that explore document loading and a separate ChromaDB prototype.

## How It Works

The API request path is:

1. `src.data_loader.load_documents("data")` loads supported files beneath `data/` into LangChain documents.
2. `src.embedding.EmbeddingPipeline` splits documents into chunks (1,000 characters with 200 characters of overlap by default) and embeds each chunk with Sentence Transformers' `all-MiniLM-L6-v2` model.
3. `src.vector_store.FaissVectorStore` stores embeddings in a FAISS `IndexFlatL2` index and stores each chunk's text in a companion pickle file.
4. `src.search.RAGSearch` embeds a question, retrieves its nearest chunks, assembles them as context, and sends a prompt to Groq using `qwen/qwen3.8-27b` by default.
5. The API returns the generated text in a JSON object.

The service uses retrieved text as model context; it does not fine-tune the language model.

## Requirements

- Python 3.14, as specified by `.python-version` and `pyproject.toml`.
- A Groq API key, loaded from the `GROQ_API_KEY` environment variable.
- Packages listed in `requirements.txt` for the current application and notebook dependencies.

The dependency declarations are not identical: `pyproject.toml` does not list every package used by the service, including FastAPI, Uvicorn, FAISS, Sentence Transformers, and Groq integration. For this repository's current app, use `requirements.txt` to install dependencies. Some optional document loaders may need their own extra packages depending on the file types you add.

## Installation

From the project root, create and activate a virtual environment, then install the requirements. For example, in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the Groq key in the process environment, or put it in a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

`.env` is ignored by Git. Keep the real key private and do not commit it.

## Run the API

Start the application from the project root:

```powershell
uvicorn app:app --reload
```

The application is available at `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health-style message indicating that the chatbot is running. |
| `GET` | `/index` | Loads supported documents under `data/`, builds and saves the FAISS index and metadata, then loads them into the API's store. |
| `POST` | `/query?query=...` | Retrieves up to three chunks and asks Groq to summarize relevant context for the query. |

For example, after starting the service, request an answer with:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/query?query=What%20is%20machine%20learning%3F"
```

The response follows the `QueryResposnse` schema in `api/schemas.py`:

```json
{
	"response": "..."
}
```

`query` is currently a query-string parameter, not a JSON request body. The API loads or builds its retrieval components during application startup, so configure `GROQ_API_KEY` before launching Uvicorn.

## Documents and Indexing

Place source files anywhere beneath `data/`. The loader recursively searches for:

- PDF: `.pdf`
- Plain text: `.txt`
- CSV: `.csv`
- Excel: `.xlsx`
- Word: `.docx`
- JSON: `.json`

The repository includes `data/pdf/Turbines_and_Compressors.pdf` and two text examples in `data/text_files/`: `machine_learning.txt` and `python_intro.txt`. The text examples cover introductory machine learning and Python topics. Run `GET /index` after adding or changing source files to rebuild the API's FAISS index.

The active API index is stored under `faiss_store/`:

- `faiss.index` contains the FAISS vector index.
- `metadata.pkl` contains the text associated with the vectors.

Both files are required for `RAGSearch` to load an existing index. The first run downloads the sentence-transformer model if it is not already cached. Indexing and model loading can take time, especially on CPU.

## Project Layout

```text
.
├── app.py                    # FastAPI application and HTTP endpoints
├── api/
│   └── schemas.py             # Query response schema
├── src/
│   ├── data_loader.py         # Recursive file loading
│   ├── embedding.py           # Text chunking and embedding generation
│   ├── search.py              # Retrieval plus Groq summarization
│   └── vector_store.py        # FAISS indexing, persistence, and search
├── data/
│   ├── pdf/                   # Example PDF source
│   ├── text_files/            # Example TXT sources
│   └── vector_store/          # ChromaDB data used by notebook prototype
├── faiss_store/               # FAISS index used by the API
├── notebooks/
│   ├── document.ipynb         # LangChain document and loader experiments
│   └── pdf_loader.ipynb       # PDF-to-Chroma retrieval and Groq prototype
├── requirements.txt           # Application and notebook dependencies
├── pyproject.toml             # uv project metadata and declared dependencies
└── .python-version            # Python version selection (3.14)
```

The project also contains `uv.lock`, which records a uv dependency resolution, and `.gitignore`, which excludes the local `.env`, virtual environment, and generated Python/build files.

## Source Modules

- **`app.py`** creates the FastAPI app and a FAISS store. `/index` loads all files in `data/`, builds the index, and persists it. `/query` uses `RAGSearch` and returns the result using the schema in `api/schemas.py`.
- **`src/data_loader.py`** selects a LangChain community loader based on the supported file extension. A failure to load one file is logged and does not stop attempts to load other files.
- **`src/embedding.py`** uses `RecursiveCharacterTextSplitter` and `SentenceTransformer`. Its defaults are a 1,000-character chunk size and 200-character overlap.
- **`src/vector_store.py`** builds a FAISS L2 index, stores chunk text as metadata, persists the index and metadata separately, and embeds query text before nearest-neighbor search.
- **`src/search.py`** retrieves matching chunk text and invokes Groq to summarize it for the question. It returns `No relevant documents found.` when no context was retrieved.
- **`api/schemas.py`** defines the API response as one string field named `response`.

## Notebooks and Vector Stores

The notebooks are experiments rather than required steps to run the API:

- `notebooks/document.ipynb` demonstrates LangChain `Document` metadata, creates the sample text files, and tries loading TXT, PDF, and CSV documents.
- `notebooks/pdf_loader.ipynb` walks through loading and chunking PDFs, generating embeddings, persisting and querying a ChromaDB collection, and passing retrieved context to Groq. Its ChromaDB data is in `data/vector_store/`.

The notebook's ChromaDB vector store is separate from the API's FAISS index. Updating one does not update the other. The API uses FAISS; ChromaDB is used in the notebook prototype.

## Current Implementation Notes

- The `RAGSearch` constructor expects `faiss_store/faiss.index` and `faiss_store/metadata.pkl` together. If either is missing, it attempts to build an index. In that fallback path, `src/search.py` imports `load_documents` as `from data_loader import load_documents`, but the module in this repository is `src/data_loader.py`. Since `RAGSearch` is constructed while `app.py` is imported, this can prevent the server from starting when either file is missing, so `/index` is not available to repair that state. Correct the import or provide both index files before starting the app.
- `requirements.txt` is the more complete install list for the current app. The `[project.scripts]` entry in `pyproject.toml` points to `rag:main`, while the function is currently in `src/__init__.py`; the documented way to launch the API is `uvicorn app:app --reload`.
- `src/vector_store.py` returns the requested number of FAISS result slots even when some indices are missing, so retrieval quality depends on having a populated index and enough chunks for the requested `top_k`.

## License and Contributions

No license or contribution policy is currently included in the repository. Add one before redistributing the project or accepting external contributions if those are needed.
