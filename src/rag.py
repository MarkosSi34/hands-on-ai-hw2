import os
import logging
import argparse
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for the Adult Census Income domain.

    Domain  : Socioeconomic / Demographics (income > $50K classification)
    Task    : Provide semantic retrieval over domain documents so the
              LangGraph agent can answer factual / conceptual questions.
    Source  : User-supplied documents in data/documents/

    Design choices
    ──────────────
    • Embedding model: 'sentence-transformers/all-MiniLM-L6-v2'.
      Runs locally (no API cost), 384-dim, fast on CPU, good quality
      for short factual passages.
    • Vector store: Chroma with on-disk persistence — the store is built
      once and reloaded on subsequent runs.
    • Chunking: RecursiveCharacterTextSplitter with chunk_size=500 and
      overlap=50. Small enough to keep retrieved context focused, large
      enough that a single passage still carries semantic meaning.
    • Retrieval: similarity search returning the top-k chunks
      concatenated into a single context string (the format the agent
      receives as tool output).
    """

    DOCUMENTS_DIR  = "data/documents"
    VECTOR_DIR     = "data/vector_store"
    COLLECTION     = "adult_income_kb"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE     = 500
    CHUNK_OVERLAP  = 50
    TOP_K          = 3

    def __init__(self, documents_dir: str = None, vector_dir: str = None):
        self.documents_dir = documents_dir or self.DOCUMENTS_DIR
        self.vector_dir    = vector_dir    or self.VECTOR_DIR
        self.embeddings   = None
        self.vector_store = None
        self.documents: List[Document] = []
        self.chunks:    List[Document] = []

    # Pipeline stages
    def load_documents(self):
        """
        Read every .txt and .pdf file under documents_dir.
        Each file becomes one (or more, for PDFs) LangChain Document.
        """
        logging.info(f"Loading documents from '{self.documents_dir}'...")

        docs_path = Path(self.documents_dir)
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_dir}"
            )

        txt_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        pdf_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
        )

        self.documents = txt_loader.load() + pdf_loader.load()

        if not self.documents:
            raise RuntimeError(
                f"No .txt or .pdf documents found in {self.documents_dir}. "
                f"Add at least 5 domain documents before building the store."
            )

        # Track unique source files for clean logging
        sources = {d.metadata.get("source", "?") for d in self.documents}
        logging.info(
            f"Loaded {len(self.documents)} document object(s) "
            f"from {len(sources)} unique file(s)."
        )

    def chunk_documents(self):
        """
        Recursive character splitter. Splits on paragraph → line → sentence
        → word boundaries so chunks are coherent.
        """
        logging.info(
            f"Chunking documents "
            f"(chunk_size={self.CHUNK_SIZE}, overlap={self.CHUNK_OVERLAP})..."
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(self.documents)

        logging.info(f"Produced {len(self.chunks)} chunks.")

    def build_embeddings(self):
        """Instantiate the local sentence-transformer embedding model."""
        logging.info(f"Loading embedding model '{self.EMBEDDING_MODEL}'...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logging.info("Embedding model ready.")

    def build_vector_store(self):
        """
        Build a new Chroma store from self.chunks and persist it to disk.
        Overwrites any existing store at vector_dir.
        """
        logging.info(f"Building Chroma vector store at '{self.vector_dir}'...")

        os.makedirs(self.vector_dir, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            collection_name=self.COLLECTION,
            persist_directory=self.vector_dir,
        )

        logging.info(
            f"Vector store built and persisted. "
            f"Collection: '{self.COLLECTION}' | "
            f"Vectors: {len(self.chunks)}"
        )

    def load_existing_store(self):
        """
        Reload a previously persisted Chroma store from disk without
        rebuilding. Used by the FastAPI app on startup.
        """
        if self.embeddings is None:
            self.build_embeddings()

        logging.info(f"Loading existing vector store from '{self.vector_dir}'...")
        self.vector_store = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=self.vector_dir,
        )
        count = self.vector_store._collection.count()
        logging.info(f"Vector store loaded. {count} vectors available.")
        return self.vector_store

    # Public retrieval API (used by the agent tool)

    def retrieve(self, query: str, k: int = None):
        """
        Return the top-k most semantically similar chunks as a single
        concatenated string, ready to be passed to the LLM as context.

        Each chunk is prefixed with its source filename so the agent can
        cite where information came from.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "Vector store not initialised. Call run_pipeline() or "
                "load_existing_store() first."
            )

        k = k or self.TOP_K
        results = self.vector_store.similarity_search(query, k=k)

        if not results:
            return "No relevant information found in the knowledge base."

        passages = []
        for i, doc in enumerate(results, start=1):
            source = Path(doc.metadata.get("source", "unknown")).name
            passages.append(f"[Source {i}: {source}]\n{doc.page_content.strip()}")

        return "\n\n---\n\n".join(passages)

    # Orchestrator

    def run_pipeline(self, rebuild: bool = False):
        """
        Build (or reload) the vector store end-to-end.

        If rebuild=False and a persisted store already exists at
        vector_dir, the store is loaded without re-embedding.
        Otherwise the full pipeline runs:
            load_documents → chunk_documents → build_embeddings
            → build_vector_store

        Returns the live RAGPipeline instance (chainable).
        """
        store_exists = (
            Path(self.vector_dir).exists()
            and any(Path(self.vector_dir).iterdir())
        )

        if store_exists and not rebuild:
            logging.info("Persisted vector store detected — loading.")
            self.load_existing_store()
            return self

        logging.info("Building vector store from scratch.")
        self.load_documents()
        self.chunk_documents()
        self.build_embeddings()
        self.build_vector_store()
        return self



# CLI entry point — `python -m src.rag --rebuild`

def main():
    parser = argparse.ArgumentParser(description="Build or reload the RAG vector store.")
    parser.add_argument("--rebuild", action="store_true", help="Force a full rebuild even if a persisted store exists.")
    parser.add_argument("--query", type=str, default=None, help="Optional smoke-test query to run against the built store.")
    args = parser.parse_args()

    rag = RAGPipeline().run_pipeline(rebuild=args.rebuild)

    if args.query:
        logging.info(f"Running smoke-test query: '{args.query}'")
        print("\n" + "=" * 72)
        print(rag.retrieve(args.query))
        print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
