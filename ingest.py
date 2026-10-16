"""
ingest.py - build (or rebuild) the document database.

Run this after you add, change, or delete files in the docs/ folder:
    python ingest.py

The chat apps also offer a "reindex" option, so this is just a shortcut.
"""

import rag_core


def main():
    for folder in [rag_core.DOCS_FOLDER, *rag_core.EXTRA_FOLDERS]:
        print(f"Reading files from: {folder}")
    print("Splitting, embedding and saving chunks (first run downloads the "
          "embedding model, ~80 MB)...")
    summary = rag_core.build_index()
    print(f"Done. Indexed {summary['files']} file(s) into {summary['chunks']} chunk(s).")
    if summary["chunks"] == 0:
        print("No text found. Put .pdf, .txt or .md files in the docs/ folder and run again.")


if __name__ == "__main__":
    main()
