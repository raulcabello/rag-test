import re
import os
import argparse
import uuid

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, ExperimentalMarkdownSyntaxTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.chains.summarize.chain import load_summarize_chain
from langchain_core.documents import Document
from langchain_ollama import ChatOllama 

BASE_URL = os.environ.get("BASE_URL", "")

def _load_hierarchical(persist_dir, persist_dir_docstore, embeddings, raw_docs):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    local_file_store = LocalFileStore(persist_dir_docstore)
    docstore = create_kv_docstore(local_file_store)

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f"Adding {len(raw_docs)} new documents to hierarchical persistent stores...")
    parent_retriever.add_documents(raw_docs)
    
def _load_summary(persist_dir, persist_dir_docstore, embeddings, raw_docs):
    print(f"RAG → {len(raw_docs)} raw documents loaded")

    # 2. Define Splitter and Stores
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    local_file_store = LocalFileStore(persist_dir_docstore)
    docstore = create_kv_docstore(local_file_store)

    # PARENT splitter: Splits by Markdown headers
    headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
    parent_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    # 3. Create the Summarization Chain
    # We use 'stuff' to summarize each document file as a whole.
    summarization_chain = load_summarize_chain(
        ChatOllama(model="gpt-oss:20b", base_url=BASE_URL), 
        chain_type="stuff"
    )

    parent_summaries = []
    parent_chunks_to_store = {}

    print("Starting document sectioning and summarization...")

    # 4. Process docs one by one
    for doc in raw_docs:
        # 4a. Split doc into PARENT chunks (sections)
        parent_chunks = parent_splitter.split_text(doc.page_content)

        print(f"Splitting '{doc.metadata.get('source')}' into {len(parent_chunks)} sections.")

        for parent_chunk in parent_chunks:
            # Generate a unique ID for this parent chunk
            current_chunk_id = str(uuid.uuid4())

            # 4b. Store the PARENT CHUNK (section) in the doc_store
            # We update its metadata to include the original source
            parent_chunk.metadata["original_source"] = doc.metadata.get("source")
            parent_chunks_to_store[current_chunk_id] = parent_chunk

            # 4c. Create and store the summary
            try:
                # Summarize the parent chunk
                summary_result = summarization_chain.invoke([parent_chunk]) 
                summary = summary_result['output_text']

                # Create the parent summary doc (for the vector store)
                summary_doc = Document(
                    page_content=summary, 
                    metadata={
                        "source": doc.metadata.get("source"),
                        "title": parent_chunk.metadata.get("H1", doc.metadata.get("title", "Untitled Section")),
                        "section_header": parent_chunk.metadata.get("H2", parent_chunk.metadata.get("H3")),
                        # This ID is the link to the parent chunk in the doc_store
                        "original_id": current_chunk_id 
                    }
                )
                parent_summaries.append(summary_doc)
            except Exception as e:
                print(f"Error summarizing chunk from {doc.metadata.get('source')}: {e}")
                # Fallback: use a snippet
                summary_doc = Document(
                    page_content=parent_chunk.page_content[:400] + "...",
                    metadata={
                        "source": doc.metadata.get("source"),
                        "title": parent_chunk.metadata.get("H1", doc.metadata.get("title", "Untitled Section")),
                        "section_header": parent_chunk.metadata.get("H2", parent_chunk.metadata.get("H3")),
                        "original_id": current_chunk_id
                    }
                )
                parent_summaries.append(summary_doc)

    # 5. Manually Populate the Stores

    # 5a. Populate Vector Store with SUMMARIES of sections
    print(f"Adding {len(parent_summaries)} section summaries to vector store.")
    vector_store.add_documents(parent_summaries)

    # 5b. Populate Doc Store with full PARENT SECTION chunks
    print(f"Adding {len(parent_chunks_to_store)} parent sections to doc store.")
    docstore.mset(list(parent_chunks_to_store.items()))

    # 6. Instantiate the Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        id_key="original_id", # Links summaries to parent chunks
        child_splitter=child_splitter,
        #search_kwargs={"k": 10}
        # We can also set search_kwargs here, e.g., {'k': 5}
        # to control how many sections are retrieved.
    )

    print(f"Hierarchical (Section-based) RAG initialized successfully.")

    return retriever

def _load_recursive(persist_dir, embeddings, raw_docs, chunk_size, chunk_overlap):
    print("loading recursive...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(raw_docs)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

def _load_markdown(persist_dir, embeddings, raw_docs):
    print("loading markdown...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    all_chunks = []
    for doc in raw_docs:
        chunks = markdown_splitter.split_text(doc.page_content)
        for chunk in chunks:
            if len(chunk.page_content) > 2000:
                print(f"Chunk too large ({len(chunk.page_content)}). Applying recursive split...")
                # Split the large chunk's content into smaller pieces
                recursively_split_documents = recursive_splitter.split_documents([chunk])
                all_chunks.extend(recursively_split_documents)
            else:
                all_chunks.append(chunk)

    Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory=persist_dir)

def main():
    parser = argparse.ArgumentParser(description="Load and chunk markdown documents into a ChromaDB vector store.")
    parser.add_argument(
        '--splitter',
        type=str,
        choices=['markdown', 'recursive','hierarchical','summary','all'],
        default='markdown',
        help="The type of text splitter to use. 'header' splits by markdown headers, 'recursive' splits by character count."
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help="The size of each chunk for the recursive splitter."
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help="The overlap between chunks for the recursive splitter."
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default="chroma_db",
        help="The directory to persist the ChromaDB vector store."
    )
    parser.add_argument(
        '--persist-dir-docstore',
        type=str,
        default="docstore",
        help="The directory to persist the ChromaDB vector store."
    )

    args = parser.parse_args()

    doc_dir = "docs/fleet"
    if not os.path.isdir(doc_dir):
        print(f"Error: The directory '{doc_dir}' was not found.")
        return
    
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)
    loader = DirectoryLoader(path=doc_dir, glob="**/*.md")
    raw_docs = loader.load()

    if args.splitter == 'hierarchical':
        _load_hierarchical(args.persist_dir, args.persist_dir_docstore, embeddings, raw_docs)
    elif args.splitter == 'summary':
        _load_summary(args.persist_dir, args.persist_dir_docstore, embeddings, raw_docs)
    elif args.splitter == 'recursive':
        _load_recursive(args.persist_dir,embeddings, args.chunk_size, args.chunk_overlap)
    elif args.splitter == 'markdown':
        _load_markdown(args.persist_dir, embeddings, raw_docs)
    else:
        # load all
        _load_markdown(args.persist_dir+"/markdown", embeddings, raw_docs)
        _load_recursive(args.persist_dir+"/recursive",embeddings, raw_docs, args.chunk_size, args.chunk_overlap)
        _load_hierarchical(args.persist_dir+"/hierarchical", args.persist_dir_docstore+"/hierarchical", embeddings, raw_docs)
        _load_summary(args.persist_dir+"/summary", args.persist_dir_docstore+"/summary", embeddings, raw_docs)

if __name__ == "__main__":
    main()
