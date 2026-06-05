import argparse
import logging
import sys
from pathlib import Path

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_BUCKET,
    VECTORSTORE_DIR,
    FAISS_INDEX_REMOTE,
    FAISS_PKL_REMOTE,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Local FAISS index paths
LOCAL_INDEX_DIR  = VECTORSTORE_DIR / "index"
LOCAL_FAISS_FILE = LOCAL_INDEX_DIR / "index.faiss"
LOCAL_PKL_FILE   = LOCAL_INDEX_DIR / "index.pkl"

# Remote paths mapped to local files
REMOTE_TO_LOCAL: dict[str, Path] = {
    FAISS_INDEX_REMOTE: LOCAL_FAISS_FILE,
    FAISS_PKL_REMOTE:   LOCAL_PKL_FILE,
}

# Supabase Storage integrations
# Supabase Client

def _get_client():
    """Return a Supabase client, raising clearly if credentials are missing."""
    try:
        from supabase import create_client, Client  
    except ImportError:
        raise ImportError(
            "supabase-py is not installed. Run: pip install supabase"
        )

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env / Render env vars."
        )

    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _ensure_bucket_exists(client) -> None:
    """Create the storage bucket if it doesn't already exist."""
    try:
        buckets = [b.name for b in client.storage.list_buckets()]
        if SUPABASE_BUCKET not in buckets:
            client.storage.create_bucket(SUPABASE_BUCKET, options={"public": False})
            log.info("Created Supabase bucket: %s", SUPABASE_BUCKET)
        else:
            log.info("Bucket '%s' already exists.", SUPABASE_BUCKET)
    except Exception as exc:
        log.warning("Could not verify bucket (may already exist): %s", exc)


# Upload

def upload_vectorstore() -> None:
    """Upload both FAISS index files to Supabase Storage."""
    if not LOCAL_FAISS_FILE.exists() or not LOCAL_PKL_FILE.exists():
        log.error(
            "FAISS index files not found at '%s'. "
            "Run `python create_memory.py` first.",
            LOCAL_INDEX_DIR,
        )
        sys.exit(1)

    client = _get_client()
    _ensure_bucket_exists(client)
    storage = client.storage.from_(SUPABASE_BUCKET)

    for remote_path, local_path in REMOTE_TO_LOCAL.items():
        log.info("Uploading %s → supabase://%s/%s …", local_path.name, SUPABASE_BUCKET, remote_path)
        with open(local_path, "rb") as f:
            data = f.read()

        try:
            # Try update first (file already exists), then insert
            storage.update(remote_path, data, {"content-type": "application/octet-stream"})
            log.info("  Updated (already existed).")
        except Exception:
            try:
                storage.upload(remote_path, data, {"content-type": "application/octet-stream"})
                log.info("  Uploaded (new file).")
            except Exception as exc:
                log.error("  Upload failed for %s: %s", remote_path, exc)
                raise

    log.info("✅ Vectorstore upload complete.")


# Download

def download_vectorstore(force: bool = False) -> bool:
    """
    Download FAISS index files from Supabase Storage to local disk.

    Args:
        force: Re-download even if local files already exist.

    Returns:
        True if download succeeded, False if files were not found in bucket.
    """
    # Skip if already present 
    if not force and LOCAL_FAISS_FILE.exists() and LOCAL_PKL_FILE.exists():
        log.info("FAISS index already present locally — skipping download.")
        return True

    LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client  = _get_client()
        storage = client.storage.from_(SUPABASE_BUCKET)
    except (ImportError, EnvironmentError) as exc:
        log.error("Cannot connect to Supabase: %s", exc)
        return False

    all_ok = True
    for remote_path, local_path in REMOTE_TO_LOCAL.items():
        log.info("Downloading supabase://%s/%s → %s …", SUPABASE_BUCKET, remote_path, local_path.name)
        try:
            data = storage.download(remote_path)
            local_path.write_bytes(data)
            log.info("  ✓ %s (%d KB)", local_path.name, len(data) // 1024)
        except Exception as exc:
            log.error("  ✗ Failed to download %s: %s", remote_path, exc)
            all_ok = False

    if all_ok:
        log.info("✅ Vectorstore download complete.")
    else:
        log.error("❌ Some files failed to download — vectorstore may be incomplete.")

    return all_ok


def is_vectorstore_in_bucket() -> bool:
    """Check whether the FAISS files exist in the Supabase bucket (no download)."""
    try:
        client  = _get_client()
        storage = client.storage.from_(SUPABASE_BUCKET)
        files   = storage.list("index")
        names   = {f["name"] for f in files}
        return "index.faiss" in names and "index.pkl" in names
    except Exception as exc:
        log.warning("Could not check bucket contents: %s", exc)
        return False


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Lawgic AI vectorstore in Supabase Storage.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--upload",   action="store_true", help="Upload local FAISS index to Supabase.")
    group.add_argument("--download", action="store_true", help="Download FAISS index from Supabase.")
    group.add_argument("--check",    action="store_true", help="Check if index exists in bucket.")
    parser.add_argument("--force",   action="store_true", help="Force re-download even if local files exist.")
    args = parser.parse_args()

    if args.upload:
        upload_vectorstore()
    elif args.download:
        ok = download_vectorstore(force=args.force)
        sys.exit(0 if ok else 1)
    elif args.check:
        exists = is_vectorstore_in_bucket()
        print("✅ Found in bucket." if exists else "❌ Not found in bucket.")
        sys.exit(0 if exists else 1)