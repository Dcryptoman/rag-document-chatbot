"""
rag_core.py - the "engine" of the chatbot.

Both the terminal app (chat_terminal.py) and the web app (app_streamlit.py)
import their logic from this file, so the RAG pipeline lives in one place.

RAG = Retrieval-Augmented Generation. In plain English:
  1. LOAD     - read your PDF and text files.
  2. CHUNK    - cut the text into small overlapping pieces ("chunks").
  3. EMBED    - turn each chunk into a list of numbers (an "embedding") that
                captures its meaning. Similar meanings -> similar numbers.
  4. STORE    - save chunks + embeddings in a local Chroma database on disk.
  5. RETRIEVE - when you ask a question, embed the question and find the
                chunks whose numbers are closest (= most similar meaning).
  6. GENERATE - send ONLY those chunks + your question to Claude, with strict
                instructions to answer from that content and nothing else.
"""

import json
import os
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

# Reads ANTHROPIC_API_KEY (and anything else) from a ".env" file next to
# this script, so you don't have to set environment variables by hand.
load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Settings - the knobs you are most likely to want to change.
# ---------------------------------------------------------------------------
BASE_FOLDER = Path(__file__).parent
DOCS_FOLDER = BASE_FOLDER / "docs"        # put your .pdf / .txt / .md / .pine files here
DB_FOLDER = BASE_FOLDER / "chroma_db"     # Chroma saves its database here
COLLECTION_NAME = "my_documents"          # a "table" inside the database

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".pine"}  # .pine = TradingView Pine Script

# Optional extra folders read IN PLACE - no copies, so the bot always sees the
# latest version. The bot only ever reads files; it never changes or moves them.
# Example: EXTRA_FOLDERS = [Path("C:/path/to/my/notes")]
EXTRA_FOLDERS = []
SKIP_FOLDER_NAMES = {"archive"}  # old versions that would muddle answers

# Remembers each file's last-modified time, so we can tell when to re-index.
FINGERPRINT_FILE = DB_FOLDER / "files_fingerprint.json"

CHUNK_SIZE = 1000      # characters per chunk (~150-200 words)
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks, so a
                       # sentence cut in half still appears whole in one chunk
TOP_K = 5              # how many chunks to send to Claude per question

CLAUDE_MODEL = "claude-opus-5-5"
MAX_ANSWER_TOKENS = 4000

# The phrase Claude must use when the documents don't contain the answer.
NOT_FOUND_MESSAGE = "I couldn't find the answer to that in your documents."


# ---------------------------------------------------------------------------
# Step 1: LOAD - read files into plain text.
# ---------------------------------------------------------------------------
def list_source_files() -> list[tuple[Path, str]]:
    """
    Find every supported file in docs/ and the EXTRA_FOLDERS (including
    sub-folders, except ones named in SKIP_FOLDER_NAMES).

    Returns (full_path, label) pairs. The label is what answers show as the
    source, e.g. "report.pdf" or "notes\\meeting.md".
    """
    files = []
    for folder in [DOCS_FOLDER, *EXTRA_FOLDERS]:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            sub_folders = {part.lower() for part in path.relative_to(folder).parts[:-1]}
            if sub_folders & SKIP_FOLDER_NAMES:
                continue
            # docs/ files are labelled from inside docs/; extra folders keep
            # their own folder name so you can tell where they came from.
            base = folder if folder == DOCS_FOLDER else folder.parent
            files.append((path, str(path.relative_to(base))))
    return files


def load_documents(files: list[tuple[Path, str]] | None = None) -> list[dict]:
    """
    Read the given (path, label) files - or every file from
    list_source_files() if none are given.

    Returns a list of "pages". Each page is a dict like:
        {"text": "...", "source": "report.pdf", "page": 3}
    PDFs give one entry per page (so answers can cite page numbers).
    Text files give one entry for the whole file (page = None).
    """
    pages = []
    for path, source in (files if files is not None else list_source_files()):
        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(path)
            except Exception as error:  # a broken PDF shouldn't stop everything
                print(f"  ! Skipping {source}: could not open PDF ({error})")
                continue
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"text": text, "source": source, "page": page_number})
            # Scanned PDFs are just pictures of text - pypdf can't read those.
            if not any(p["source"] == source for p in pages):
                print(f"  ! {source}: no text found (is it a scanned image? needs OCR)")
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                pages.append({"text": text, "source": source, "page": None})

    return pages


# ---------------------------------------------------------------------------
# Step 2: CHUNK - cut long text into overlapping pieces.
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into pieces of about `chunk_size` characters.

    Why chunk at all? A search over small pieces finds the exact paragraph
    that answers your question, instead of a whole 50-page document.

    We try to end each chunk at the end of a sentence (". ") so chunks read
    naturally, and each chunk repeats the last `overlap` characters of the
    previous one so no idea gets lost at a boundary.
    """
    text = " ".join(text.split())  # collapse messy PDF line breaks/spaces
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Look backwards for a sentence end, but not too far back.
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # step forward, keeping overlap
    return chunks


# ---------------------------------------------------------------------------
# Steps 3 + 4: EMBED and STORE in Chroma.
# ---------------------------------------------------------------------------
def get_collection():
    """
    Open (or create) the Chroma collection on disk.

    The embedding model is Chroma's built-in default ("all-MiniLM-L6-v2").
    It runs locally on your computer for free - no extra API key. The first
    run downloads it (~80 MB); after that it's cached.
    """
    client = chromadb.PersistentClient(path=str(DB_FOLDER))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},  # compare meanings by angle (cosine)
    )


def build_index() -> dict:
    """
    (Re)build the whole database from docs/ and the EXTRA_FOLDERS.

    We delete the old collection first so removed or edited files don't
    leave stale chunks behind. Simple and always correct - fine for
    personal-sized document folders.

    Returns a small summary: {"files": n, "chunks": n}.
    """
    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    files = list_source_files()
    fingerprint = make_fingerprint(files)  # taken first, so edits made mid-build get caught next time

    client = chromadb.PersistentClient(path=str(DB_FOLDER))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    pages = load_documents(files)
    chunk_count = add_pages(get_collection(), pages)

    FINGERPRINT_FILE.write_text(json.dumps(fingerprint))
    return {"files": len({p["source"] for p in pages}), "chunks": chunk_count}


def update_index() -> list[str]:
    """
    Bring the index up to date, re-reading ONLY the files that were added,
    edited or deleted since last time. Returns their labels (an empty list
    means nothing changed).

    Spotting changes is instant (it compares file dates and sizes), and only
    changed files are re-embedded, so editing one script costs seconds - not
    the minutes a full rebuild of every file takes.
    """
    try:
        saved = json.loads(FINGERPRINT_FILE.read_text())
    except (OSError, ValueError):  # never built, or the record got damaged
        build_index()
        return ["(all files - first build)"]

    files = list_source_files()
    current = make_fingerprint(files)
    changed = sorted(label for label in saved.keys() | current.keys()
                     if saved.get(label) != current.get(label))
    if not changed:
        return []

    collection = get_collection()
    for label in changed:  # drop the old chunks of every changed/deleted file
        collection.delete(where={"source": label})
    add_pages(collection, load_documents([f for f in files if f[1] in changed]))

    FINGERPRINT_FILE.write_text(json.dumps(current))
    return changed


def make_fingerprint(files: list[tuple[Path, str]]) -> dict:
    """{label: [last-modified time, size]} for each file."""
    return {label: [path.stat().st_mtime, path.stat().st_size] for path, label in files}


def add_pages(collection, pages: list[dict]) -> int:
    """Chunk the pages, embed the chunks and store them. Returns chunk count."""
    ids, texts, metadatas = [], [], []
    for page in pages:
        for i, chunk in enumerate(split_into_chunks(page["text"])):
            # Every chunk needs a unique ID inside the database.
            ids.append(f"{page['source']}|p{page['page']}|c{i}")
            texts.append(chunk)
            # Chroma metadata can't hold None, so store "page 0" for text files.
            metadatas.append({"source": page["source"], "page": page["page"] or 0})

    # Add in batches - Chroma embeds each batch as it goes (this is the slow
    # part on the first run, a few seconds per hundred chunks).
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[start:start + batch_size],
            documents=texts[start:start + batch_size],
            metadatas=metadatas[start:start + batch_size],
        )
    return len(ids)


# ---------------------------------------------------------------------------
# Step 5: RETRIEVE - find the chunks most relevant to a question.
# ---------------------------------------------------------------------------
def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Return the `top_k` chunks closest in meaning to `question`.

    Each result: {"text", "source", "page", "distance"}.
    distance: 0 = identical meaning, bigger = less related.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )
    # Chroma returns lists-of-lists (one inner list per question); we asked
    # one question, so take element [0] of each.
    chunks = []
    for text, meta, distance in zip(results["documents"][0],
                                    results["metadatas"][0],
                                    results["distances"][0]):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"] or None,
            "distance": distance,
        })
    return chunks


def format_source(chunk: dict) -> str:
    """'report.pdf, page 3' or just 'notes.txt'."""
    if chunk["page"]:
        return f"{chunk['source']}, page {chunk['page']}"
    return chunk["source"]


# ---------------------------------------------------------------------------
# Step 6: GENERATE - ask Claude, using ONLY the retrieved chunks.
# ---------------------------------------------------------------------------

# The system prompt is how we stop Claude from making things up. It says:
# use only the excerpts, cite them, and admit when the answer isn't there.
SYSTEM_PROMPT = f"""You answer questions using ONLY the document excerpts provided in the user's message.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge, even if you know the answer.
- Cite the excerpt(s) you used with their numbers in square brackets, e.g. [1] or [2][4].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}" You may then add one short sentence saying what the excerpts do cover.
- If the excerpts only partly answer the question, answer the part they support and say clearly which part is not covered.
- Do not guess, speculate, or fill gaps. Quoting short phrases from the excerpts is encouraged."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Put the numbered excerpts and the question into one message."""
    excerpts = "\n\n".join(
        f"<excerpt number=\"{n}\" source=\"{format_source(c)}\">\n{c['text']}\n</excerpt>"
        for n, c in enumerate(chunks, start=1)
    )
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"


class MissingAPIKeyError(RuntimeError):
    """Raised when no ANTHROPIC_API_KEY is set, with a friendly message."""


def get_claude_client() -> anthropic.Anthropic:
    """
    Create the Anthropic client. It reads ANTHROPIC_API_KEY from the
    environment (our .env file was loaded at the top of this file).
    """
    # Check up front: without a key the SDK fails later with a confusing error.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "No API key found. Copy .env.example to .env in the rag_chatbot "
            "folder and put your key after ANTHROPIC_API_KEY= (see README.md)."
        )
    return anthropic.Anthropic()


def answer_stream(question: str, chunks: list[dict], client: anthropic.Anthropic | None = None):
    """
    Ask Claude and yield the answer piece by piece as it is written
    ("streaming"), so you see words appear instead of waiting.

    If no chunks were found we don't call the API at all - there is nothing
    to answer from, so we can't (and shouldn't) answer.
    """
    if not chunks:
        yield NOT_FOUND_MESSAGE
        return

    client = client or get_claude_client()

    # `fallbacks="default"`: if Claude's safety system ever declines a request,
    # the API automatically retries it on a suitable fallback model instead
    # of just stopping. It needs the beta flag below, which is why this uses
    # client.beta.messages rather than client.messages.
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        yield "\n\n(Claude declined to answer this question.)"
    elif final.stop_reason == "max_tokens":
        yield "\n\n(Answer cut off - raise MAX_ANSWER_TOKENS in rag_core.py.)"
