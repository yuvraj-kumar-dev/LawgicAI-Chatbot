import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import SUPABASE_URL, SUPABASE_KEY, COMMUNITY_TAGS

log = logging.getLogger(__name__)


# Supabase client

def _get_client():
    """Return a cached Supabase client."""
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError("Run: pip install supabase")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in env.")

    return create_client(SUPABASE_URL, SUPABASE_KEY)


# Helpers 

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_time_ago(iso_str: str) -> str:
    """Convert ISO timestamp to 'X ago' """
    try:
        # Supabase returns timestamps with timezone offset
        ts = iso_str.replace("Z", "+00:00")
        then  = datetime.fromisoformat(ts)
        delta = datetime.now(timezone.utc) - then
        secs  = int(delta.total_seconds())

        if secs < 60:
            return "just now"
        elif secs < 3600:
            m = secs // 60
            return f"{m} min{'s' if m > 1 else ''} ago"
        elif secs < 86400:
            h = secs // 3600
            return f"{h} hr{'s' if h > 1 else ''} ago"
        else:
            d = secs // 86400
            return f"{d} day{'s' if d > 1 else ''} ago"
    except Exception:
        return iso_str


def get_all_tags() -> list[str]:
    return COMMUNITY_TAGS


# Posts

def create_post(author: str, title: str, body: str, tags: list[str]) -> dict:
    """Insert a new post. Returns the created row."""
    if not title.strip():
        raise ValueError("Post title cannot be empty.")
    if not body.strip():
        raise ValueError("Post body cannot be empty.")

    client = _get_client()
    row = {
        "id":         str(uuid.uuid4()),
        "author":     author.strip() or "Anonymous",
        "title":      title.strip(),
        "body":       body.strip(),
        "tags":       tags,          
        "likes":      0,
        "created_at": _now(),
    }
    result = client.table("posts").insert(row).execute()
    data   = result.data
    if not data:
        raise RuntimeError("Post insert returned no data — check Supabase RLS policies.")
    log.info("Post created: %s (id=%s)", row["title"][:40], row["id"])
    return data[0]


def get_posts(
    tag_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Fetch posts ordered newest-first, with comment counts and like counts.
    Optionally filter by a single tag (Postgres @> array containment).
    """
    client = _get_client()

    query = (
        client.table("posts")
        .select("*, comments(count)")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    # Tag filter using Postgres array containment operator via PostgREST
    if tag_filter and tag_filter != "All":
        # cs = "contains" tags column contains this element
        query = query.filter("tags", "cs", f'["{tag_filter}"]')

    result = query.execute()
    posts  = result.data or []

    # Flatten comment count from nested aggregate
    for p in posts:
        raw = p.pop("comments", [])
        # PostgREST returns [{"count": N}] for aggregate selects
        p["comment_count"] = raw[0]["count"] if raw else 0

    return posts


def get_post(post_id: str) -> Optional[dict]:
    """Return a single post with its comments, or None."""
    client = _get_client()
    result = (
        client.table("posts")
        .select("*, comments(*)")
        .eq("id", post_id)
        .maybe_single()
        .execute()
    )
    return result.data


# Comments

def get_comments(post_id: str) -> list[dict]:
    """Return all comments for a post, oldest first."""
    client = _get_client()
    result = (
        client.table("comments")
        .select("*")
        .eq("post_id", post_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data or []


def add_comment(post_id: str, author: str, body: str) -> dict:
    """Insert a comment on a post. Returns the created row."""
    if not body.strip():
        raise ValueError("Comment cannot be empty.")

    client = _get_client()

    # Verify post exists
    check = client.table("posts").select("id").eq("id", post_id).maybe_single().execute()
    if not check.data:
        raise LookupError(f"Post '{post_id}' not found.")

    row = {
        "id":         str(uuid.uuid4()),
        "post_id":    post_id,
        "author":     author.strip() or "Anonymous",
        "body":       body.strip(),
        "created_at": _now(),
    }
    result = client.table("comments").insert(row).execute()
    data   = result.data
    if not data:
        raise RuntimeError("Comment insert returned no data.")
    return data[0]


# Likes 

def toggle_like(post_id: str, session_id: str) -> int:
    """
    Toggle a like for the given session_id on a post.
    Uses the `likes` join table (post_id, session_id unique pair).
    Returns the updated like count.
    """
    client = _get_client()

    # Check if already liked
    existing = (
        client.table("likes")
        .select("id")
        .eq("post_id",    post_id)
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    if existing.data:
        # Unlike delete row and decrement counter
        client.table("likes").delete().eq("id", existing.data["id"]).execute()
        client.rpc("decrement_likes", {"post_id_input": post_id}).execute()
    else:
        # Like insert row and increment counter
        client.table("likes").insert({
            "id":         str(uuid.uuid4()),
            "post_id":    post_id,
            "session_id": session_id,
        }).execute()
        client.rpc("increment_likes", {"post_id_input": post_id}).execute()

    # Fetch fresh count
    result = client.table("posts").select("likes").eq("id", post_id).single().execute()
    return result.data.get("likes", 0)


def has_liked(post_id: str, session_id: str) -> bool:
    """Return True if this session has already liked the post."""
    client = _get_client()
    result = (
        client.table("likes")
        .select("id")
        .eq("post_id",    post_id)
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    return result.data is not None


# Delete

def delete_post(post_id: str) -> bool:
    """Delete a post and cascade-delete its comments and likes. Returns True if deleted."""
    client = _get_client()
    result = client.table("posts").delete().eq("id", post_id).execute()
    return bool(result.data)