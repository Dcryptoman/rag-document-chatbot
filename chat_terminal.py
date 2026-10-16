"""
chat_terminal.py - chat with your documents in the terminal.

Run:
    python chat_terminal.py

Type a question and press Enter. Special commands:
    reindex   force a full re-read (normally automatic when files change)
    sources   show the excerpts used for the last answer
    quit      exit (or press Ctrl+C)
"""

import anthropic

import rag_core


def print_sources(chunks):
    """List which file/page each excerpt came from."""
    if not chunks:
        return
    print("\nSources:")
    for n, chunk in enumerate(chunks, start=1):
        print(f"  [{n}] {rag_core.format_source(chunk)}")


def refresh_if_changed():
    """Re-index any file that was added, edited or deleted (instant if none)."""
    changed = rag_core.update_index()
    if changed:
        print(f"Updated the index for {len(changed)} changed file(s): "
              + ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else ""))
    if rag_core.get_collection().count() == 0:
        print(f"\nNo readable files found. Add .pdf / .txt / .md / .pine files "
              f"to {rag_core.DOCS_FOLDER}")


def main():
    print("=" * 60)
    print(" Document chatbot - answers only from your files")
    print("=" * 60)

    refresh_if_changed()

    try:
        client = rag_core.get_claude_client()
    except rag_core.MissingAPIKeyError as error:
        print(f"\n{error}")
        return

    print("\nAsk a question (or type 'quit').")
    last_chunks = []

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        command = question.lower()
        if command in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if command == "reindex":
            summary = rag_core.build_index()
            print(f"Re-indexed {summary['files']} file(s), {summary['chunks']} chunk(s).")
            continue
        if command == "sources":
            for n, chunk in enumerate(last_chunks, start=1):
                print(f"\n[{n}] {rag_core.format_source(chunk)} "
                      f"(distance {chunk['distance']:.2f})\n{chunk['text']}")
            if not last_chunks:
                print("No sources yet - ask a question first.")
            continue

        # 0) pick up any file edits, 1) find relevant chunks,
        # 2) stream Claude's answer, 3) show sources.
        refresh_if_changed()
        last_chunks = rag_core.retrieve(question)
        print("\nBot: ", end="", flush=True)
        try:
            for piece in rag_core.answer_stream(question, last_chunks, client):
                print(piece, end="", flush=True)
            print()
            print_sources(last_chunks)
        except anthropic.AuthenticationError:
            print("\nYour API key was rejected. Check ANTHROPIC_API_KEY in .env.")
        except anthropic.RateLimitError:
            print("\nRate limited by the API - wait a moment and try again.")
        except anthropic.APIConnectionError:
            print("\nCouldn't reach the Anthropic API - check your internet connection.")
        except anthropic.APIStatusError as error:
            print(f"\nAPI error ({error.status_code}): {error.message}")


if __name__ == "__main__":
    main()
