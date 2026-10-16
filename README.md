# Document Chatbot (RAG) — beginner guide

Ask questions about your own PDF and text files. The bot finds the most
relevant passages and asks Claude to answer **using only those passages**.
If the answer isn't in your files, it says so instead of guessing.

```
your files ──► chunks ──► embeddings ──► Chroma DB (on your disk)
                                              │
your question ──► embedding ──► find closest chunks
                                              │
                         Claude ◄── question + those chunks only
                            │
                       answer + sources
```

## What this demonstrates

- **Chunking:** documents are split into overlapping ~1,000-character pieces so each piece is small enough to match a question precisely.
- **Embeddings:** each chunk becomes a vector (a list of numbers capturing its meaning) using a free local model, so similar meanings land close together.
- **Vector search:** questions are embedded the same way and matched against a local Chroma database to find the closest chunks.
- **Grounded answers with citations:** Claude only sees the retrieved chunks, must cite them like `[1]`, and says so when the answer isn't there.
- **Change detection:** the index rebuilds itself automatically when files are added, edited or deleted.

## What's in this folder

| File | What it does |
|---|---|
| `rag_core.py` | The engine: load → chunk → embed → store → retrieve → ask Claude. Read this to learn how RAG works; every step is commented. |
| `ingest.py` | Builds the database from the `docs/` folder. |
| `chat_terminal.py` | Chat in the terminal. |
| `app_streamlit.py` | The same chatbot as a web page. |
| `requirements.txt` | The Python packages to install. |
| `.env.example` | Template for your API key. |
| `docs/` | **Put your documents here** (.pdf, .txt, .md, .pine). A sample handbook is included. |
| `chroma_db/` | Created automatically. The vector database. Safe to delete; it will be rebuilt. |

## Setup (one time, about 5 minutes)

You need **Python 3.10 or newer**. Check with `python --version`.

**1. Download the code and open a terminal in its folder:**

```bash
git clone https://github.com/Dcryptoman/rag-document-chatbot.git
cd rag-document-chatbot
```

**2. Create a virtual environment** (a private box for this project's packages):

```bash
python -m venv .venv
```

**3. Activate it.** On Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

On Mac/Linux it's `source .venv/bin/activate`. Your prompt now starts with `(.venv)`.
If PowerShell says scripts are disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**4. Install the packages:**

```bash
pip install -r requirements.txt
```

**5. Add your Anthropic API key.** Get one at https://console.anthropic.com/ → API Keys.
Copy `.env.example` to a new file called `.env` and paste your key in:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Keep `.env` private. Never share or upload it.

**6. Add your documents** to the `docs/` folder.

## Run it

Terminal version:

```bash
python chat_terminal.py
```

The first run downloads a small free embedding model (~80 MB) and indexes
your files. Then type questions. Commands: `reindex` (after changing files),
`sources` (show the exact passages used), `quit`.

Web version:

```bash
streamlit run app_streamlit.py
```

Your browser opens at http://localhost:8501. You can upload files from the
sidebar, and each answer has a "Sources" box showing the passages it used.
Press `Ctrl+C` in the terminal to stop it.

To also read folders elsewhere on your computer, list them in `EXTRA_FOLDERS`
at the top of `rag_core.py` (it's empty by default). Sub-folders named
`archive` are skipped. The bot only reads files; it never changes them.

Re-indexing is automatic: before every question the bot checks whether any file
was added, edited or deleted, and rebuilds the index if so. `reindex` / the
"Re-index docs folder" button / `python ingest.py` force a rebuild by hand.

## How it avoids made-up answers

1. Claude only receives the 5 passages that best match your question, not the internet or your whole drive.
2. The system prompt (in `rag_core.py`) tells Claude to use only those passages, cite them like `[1]`, and reply "I couldn't find the answer to that in your documents" when they don't cover the question.
3. Every answer shows its sources, so you can check them yourself.

No method makes errors impossible, so check the sources on anything important.

## Settings you can change (top of `rag_core.py`)

| Setting | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk. Smaller chunks are more precise but carry less context. |
| `CHUNK_OVERLAP` | 200 | Characters repeated between neighbouring chunks. |
| `TOP_K` | 5 | How many chunks Claude gets per question. Raise it if answers miss details. |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Which Claude model answers. `claude-sonnet-5-5` and `claude-haiku-5-5` are cheaper. |

After changing `CHUNK_SIZE` or `CHUNK_OVERLAP`, re-index.

## Cost

Embeddings run on your computer for free. Each question makes one Claude API
call, with about 1,500–2,500 input tokens (5 chunks + instructions). With
Opus 5.5 ($4 per million input tokens, $20 per million output) that's
roughly 1–2 cents per question.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No API key found" / "Your API key was rejected" | Check `.env` exists in this folder (not `.env.txt`) and the key is correct, with no quotes or spaces. |
| "No text found … scanned image?" | The PDF is a photo of text. It needs OCR first (e.g. export it as "searchable PDF" from a scanner app). |
| Answers ignore a new file | Re-index. |
| `streamlit` not recognised | Activate the virtual environment first (step 3). |
| Anything weird with the database | Delete the `chroma_db` folder and re-index. |

## Limits

- The bot doesn't remember earlier questions. Each question is answered on its own. This keeps every answer tied to your documents.
- Each re-index rebuilds everything from scratch. That's fine for hundreds of files, but slow for tens of thousands.
- Tables and images inside PDFs come through poorly or not at all, because only the text is read.
