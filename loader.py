import re
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
BASE_URL = os.environ.get("BASE_URL", "")

def chunk_markdown_file(file_path: str) -> list:
    """
    Reads a Markdown file, splits it into chunks based on headers,
    and returns the chunks.

    Args:
        file_path (str): The path to the markdown file.
    Returns:
        A list of Document objects (chunks).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    # Pre-process the markdown to convert front matter title to H1 header
    # This regex finds a 'title' in YAML front matter and replaces the block with a Markdown H1 header.
    # It handles potential whitespace and other front matter keys.
    # The `[^`---`]` part ensures it only matches content *between* the '---' delimiters.
    pattern = re.compile(r'^---\s*\n(?P<front_matter>.*?title:\s*(?P<title>.*?)\n.*?)^---\s*\n', re.MULTILINE | re.DOTALL)
    markdown_text = pattern.sub(r'# \g<title>', markdown_text)

    # Remove the <head> block as it's not useful for the LLM context
    markdown_text = re.sub(r'<head>.*?</head>', '', markdown_text, flags=re.DOTALL)

    # Define the headers we want to split on, and their corresponding metadata keys.
    # The order matters, from largest header to smallest.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Create the MarkdownHeaderTextSplitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True # Set to False to keep headers in the content
    )

    # Split the text
    chunks = markdown_splitter.split_text(markdown_text)

    return chunks


def main():
    file_path = "backup-configuration.md"
    chunks = chunk_markdown_file(file_path)
    if not chunks:
        return

    print("starting to create chroma db")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    print("finished to create chroma db")

if __name__ == "__main__":
    main()
