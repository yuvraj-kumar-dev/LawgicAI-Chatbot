import logging
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K_RESULTS

log = logging.getLogger(__name__)

VECTORSTORE_PATH = VECTORSTORE_DIR / "index"


# Embeddings (Cached)

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# Auto-download from Supabase

def _try_download_from_supabase() -> bool:
    """
    Attempt to pull the FAISS index from Supabase Storage.
    Returns True on success, False if storage is not configured or download fails.
    Silent on import errors so local-only setups still work.
    """
    try:
        from storage import download_vectorstore
        return download_vectorstore(force=False)
    except Exception as exc:
        log.warning("Supabase download skipped: %s", exc)
        return False


# Vectorstore helpers

def _index_files_exist() -> bool:
    """True only when both FAISS binary files are present on disk."""
    return (
        (VECTORSTORE_PATH / "index.faiss").exists()
        and (VECTORSTORE_PATH / "index.pkl").exists()
    )


def is_vectorstore_ready() -> bool:
    """
    Return True if the FAISS index is available for use.

    On a fresh Render instance the local files won't exist yet, so we
    attempt a Supabase download first before returning False.
    """
    if _index_files_exist():
        return True

    log.info("FAISS index not found locally — trying Supabase download…")
    _try_download_from_supabase()
    return _index_files_exist()


def load_vectorstore() -> FAISS:
    """Load the FAISS vectorstore from disk.

    Raises:
        FileNotFoundError: if the vectorstore cannot be found locally
                           and could not be downloaded from Supabase.
    """
    if not _index_files_exist():
        # One more attempt in case is_vectorstore_ready() wasn't called first
        _try_download_from_supabase()

    if not _index_files_exist():
        raise FileNotFoundError(
            f"Vectorstore not found at '{VECTORSTORE_PATH}'.\n"
            "  • Locally: run `python create_memory.py` then `python storage.py --upload`\n"
            "  • On Render: ensure SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET are set."
        )

    log.info("Loading FAISS vectorstore from %s", VECTORSTORE_PATH)
    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )

# Using MMR (Maximum Marginal Relevance) over cosine similarity to retrieve diverse yet relevant top k results

def get_retriever(
    search_type: str = "mmr",
    k: int = TOP_K_RESULTS,
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
) -> VectorStoreRetriever:
    """Return a configured retriever from the FAISS vectorstore.

    Uses Maximum Marginal Relevance (MMR) by default to return
    diverse yet relevant results, reducing redundant context.
    """
    db = load_vectorstore()

    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})

    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    log.info("Retriever ready (type=%s, k=%d)", search_type, k)
    return retriever


def get_relevant_docs(query: str, k: int = TOP_K_RESULTS) -> list:
    """Convenience: retrieve top-k documents for a query string."""
    return get_retriever(k=k).invoke(query)