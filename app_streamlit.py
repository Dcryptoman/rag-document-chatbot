"""
app_streamlit.py - the same chatbot as a web page.

Run:
    streamlit run app_streamlit.py

Your browser opens at http://localhost:8501. Everything still runs on your
own computer; only the question + matching excerpts are sent to Claude.

How Streamlit works (the one thing to know): every time you click or type,
Streamlit re-runs this whole file from top to bottom. Anything that must
survive between re-runs (like the chat history) goes in `st.session_state`.
"""

import anthropic
import streamlit as st

import rag_core

st.set_page_config(page_title="Document Chatbot", page_icon="📄")
st.title("📄 Chat with your documents")
st.caption("Answers come only from the files in your docs folder.")


# ---------------------------------------------------------------------------
# Sidebar: manage documents.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Your documents")
    rag_core.DOCS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Upload files straight into the docs folder.
    uploads = st.file_uploader(
        "Add PDF or text files",
        type=["pdf", "txt", "md", "pine"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Save uploads and re-index", type="primary"):
        for upload in uploads:
            (rag_core.DOCS_FOLDER / upload.name).write_bytes(upload.getbuffer())
        with st.spinner("Reading and indexing..."):
            summary = rag_core.build_index()
        st.success(f"Indexed {summary['files']} file(s), {summary['chunks']} chunk(s).")

    if st.button("Re-index docs folder"):
        with st.spinner("Reading and indexing..."):
            summary = rag_core.build_index()
        st.success(f"Indexed {summary['files']} file(s), {summary['chunks']} chunk(s).")

    files = [label for _, label in rag_core.list_source_files()]
    with st.expander(f"**{len(files)} file(s)** being read"):
        for label in files:
            st.write(f"- {label}")

    if st.button("Clear chat"):
        st.session_state.messages = []


# ---------------------------------------------------------------------------
# Re-index automatically if any file was added, edited or deleted. This runs
# on every page interaction, but the check itself is instant.
# ---------------------------------------------------------------------------
with st.spinner("Checking for changed files..."):
    changed = rag_core.update_index()
if changed:
    st.toast(f"Updated the index for {len(changed)} changed file(s).")
if rag_core.get_collection().count() == 0:
    st.info("No documents found. Add files in the sidebar to get started.")


# ---------------------------------------------------------------------------
# Chat history (kept in session_state so it survives re-runs).
# Each message: {"role": "user"/"assistant", "content": str, "chunks": list}
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(chunks):
    """A collapsible box listing the excerpts used for an answer."""
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)} excerpts)"):
        for n, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{n}] {rag_core.format_source(chunk)}**")
            st.caption(chunk["text"])


# Re-draw the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("chunks"))


# ---------------------------------------------------------------------------
# Handle a new question.
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        chunks = rag_core.retrieve(question)
        try:
            # st.write_stream shows the words as they arrive and returns
            # the full text at the end.
            answer = st.write_stream(rag_core.answer_stream(question, chunks))
        except rag_core.MissingAPIKeyError as error:
            answer = str(error)
            st.error(answer)
        except anthropic.AuthenticationError:
            answer = "Your API key was rejected. Check ANTHROPIC_API_KEY in the .env file."
            st.error(answer)
        except anthropic.RateLimitError:
            answer = "Rate limited by the API - wait a moment and try again."
            st.error(answer)
        except anthropic.APIConnectionError:
            answer = "Couldn't reach the Anthropic API - check your internet connection."
            st.error(answer)
        except anthropic.APIStatusError as error:
            answer = f"API error ({error.status_code}): {error.message}"
            st.error(answer)
        show_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
