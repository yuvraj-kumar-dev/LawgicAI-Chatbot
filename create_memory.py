import argparse
import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    DATA_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

VECTORSTORE_PATH = VECTORSTORE_DIR / "index"


# Helper functions

def load_pdfs(data_dir: Path) -> list:
    """Load all PDFs from the given directory and return a flat list of Documents."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", data_dir)
        return []

    all_docs = []
    for pdf_path in pdf_files:
        log.info("Loading: %s", pdf_path.name)
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Attach source metadata
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            all_docs.extend(docs)
            log.info("  → %d pages loaded", len(docs))
        except Exception as exc:
            log.error("Failed to load %s: %s", pdf_path.name, exc)

    log.info("Total pages loaded: %d", len(all_docs))
    return all_docs


def split_documents(docs: list) -> list:
    """Split raw Document pages into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def build_embeddings() -> HuggingFaceEmbeddings:
    """Instantiate the (free, local) HuggingFace embedding model."""
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Build a FAISS vectorstore from document chunks."""
    log.info("Building FAISS vectorstore from %d chunks…", len(chunks))
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(VECTORSTORE_PATH))
    log.info("Vectorstore saved → %s", VECTORSTORE_PATH)
    return db


def update_vectorstore(new_chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Merge new chunks into an existing vectorstore (incremental update)."""
    if not VECTORSTORE_PATH.exists():
        return create_vectorstore(new_chunks, embeddings)

    log.info("Merging %d new chunks into existing vectorstore…", len(new_chunks))
    db = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    new_db = FAISS.from_documents(new_chunks, embeddings)
    db.merge_from(new_db)
    db.save_local(str(VECTORSTORE_PATH))
    log.info("Vectorstore updated → %s", VECTORSTORE_PATH)
    return db

# Main

def main(reset: bool = False) -> None:
    if reset and VECTORSTORE_PATH.exists():
        import shutil
        shutil.rmtree(VECTORSTORE_PATH)
        log.info("Existing vectorstore removed (--reset flag).")

    docs = load_pdfs(DATA_DIR)
    if not docs:
        log.error("No documents to process. Add PDFs to the '%s' directory.", DATA_DIR)
        sys.exit(1)

    chunks    = split_documents(docs)
    embeddings = build_embeddings()

    if reset or not VECTORSTORE_PATH.exists():
        create_vectorstore(chunks, embeddings)
    else:
        update_vectorstore(chunks, embeddings)

    log.info("✅ Memory creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lawgic AI vectorstore from PDFs.")
    parser.add_argument("--reset", action="store_true", help="Wipe existing vectorstore before rebuilding.")
    args = parser.parse_args()
    main(reset=args.reset)