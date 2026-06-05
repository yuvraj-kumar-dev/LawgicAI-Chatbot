import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR = BASE_DIR / os.getenv("VECTORSTORE_DIR", "vectorstore")

DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")          


SUPABASE_BUCKET      = os.getenv("SUPABASE_BUCKET", "vectorstore")

FAISS_INDEX_REMOTE   = os.getenv("FAISS_INDEX_REMOTE",   "index/index.faiss")
FAISS_PKL_REMOTE     = os.getenv("FAISS_PKL_REMOTE",     "index/index.pkl")


LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")   
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))

COMMUNITY_TAGS = [
    "⚖️ Lawyer",
    "🙋 Victim",
    "📰 News",
    "🏛️ Court Ruling",
    "📖 Rights & Laws",
    "🚨 Police & FIR",
    "👩‍👧 Family Law",
    "🏠 Property",
    "💼 Labour Law",
    "🌐 Constitutional",
    "❓ General Query",
]


# Validation/ Checks

def validate_config() -> list[str]:
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in .env")
    if not SUPABASE_URL:
        errors.append("SUPABASE_URL is not set in .env")
    if not SUPABASE_KEY:
        errors.append("SUPABASE_KEY is not set in .env")
    return errors