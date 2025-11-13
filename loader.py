import re
import os
import argparse

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
BASE_URL = os.environ.get("BASE_URL", "")

def chunk_markdown_file(doc_path: str, splitter_type: str) -> list:
    """
    Reads a Markdown file, splits it into chunks based on headers,
    and returns the chunks.

    Args:
        file_path (str): The path to the markdown file.
    Returns:
        A list of Document objects (chunks).
    """
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{doc_path}' was not found.")
        return []

    if splitter_type == 'header':
        # Define the headers we want to split on, and their corresponding metadata keys.
        # The order matters, from largest header to smallest.
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # Create the MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=True # Set to False to keep headers in the content
        )
        chunks = markdown_splitter.split_text(markdown_text)
    elif splitter_type == 'recursive':
        # Use a recursive character splitter for a different chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        # create_documents will return Document objects with metadata
        chunks = text_splitter.create_documents([markdown_text], metadatas=[{"source": doc_path}])
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")


    return chunks


def main():
    parser = argparse.ArgumentParser(description="Load and chunk markdown documents into a ChromaDB vector store.")
    parser.add_argument(
        '--splitter',
        type=str,
        choices=['header', 'recursive'],
        default='header',
        help="The type of text splitter to use. 'header' splits by markdown headers, 'recursive' splits by character count."
    )
    args = parser.parse_args()

    doc_dir = "docs/fleet"  # Directory containing your markdown files
    all_chunks = []

    if not os.path.isdir(doc_dir):
        print(f"Error: The directory '{doc_dir}' was not found.")
        return

    print(f"Recursively loading documents from: {doc_dir} using '{args.splitter}' splitter.")
    # Use os.walk to recursively find all markdown files
    for root, _, files in os.walk(doc_dir):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                print(f"  - Processing {file_path}")
                chunks = chunk_markdown_file(file_path, splitter_type=args.splitter)
                all_chunks.extend(chunks)

    if not all_chunks:
        print("No markdown documents found or processed in the directory.")
        return

    print("starting to create chroma db")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)
    Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    print("finished to create chroma db")

if __name__ == "__main__":
    main()
