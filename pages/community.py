import uuid
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
from config import DATA_DIR, GROQ_API_KEY, validate_config
from connect_memory import is_vectorstore_ready

from community import (
    create_post,
    get_posts,
    get_comments,
    add_comment,
    toggle_like,
    has_liked,
    delete_post,
    format_time_ago,
    get_all_tags,
)

# Page config 
st.set_page_config(
    page_title="Lawgic AI — Community",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sidebar 
with st.sidebar:
    st.markdown("### ⚙️ Knowledge Base")

    # Config warnings
    errors = validate_config()
    if errors:
        for e in errors:
            st.error(e)

    # Vectorstore status
    ready = is_vectorstore_ready()
    pill_cls = "ready" if ready else "not-ready"
    pill_txt = "✅ Knowledge base ready" if ready else "⚠️ Not built yet"
    st.markdown(f'<span class="status-pill {pill_cls}">{pill_txt}</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Current PDFs**")
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if pdfs:
        for p in pdfs:
            size_kb = p.stat().st_size // 1024
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 `{p.name}` ({size_kb} KB)")
    else:
        st.caption("No PDFs yet.")

    st.markdown("---")
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.lc_messages  = []
        st.rerun()

    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption(
        "Lawgic AI"
        f"Model: `{os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')}`  "
        "Embeddings: `all-MiniLM-L6-v2`"
    )


# CSS 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0e1a;
    color: #e0dbd2;
  }

  /* ─── Navbar ─── */
  .lawgic-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 32px;
    background: linear-gradient(90deg, #111327 0%, #1a1e38 100%);
    border-bottom: 1px solid #2e3355;
    border-radius: 0 0 16px 16px;
    margin-bottom: 12px;
  }
  .lawgic-nav .logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700; color: #c9a84c;
  }
  .lawgic-nav .logo span { color: #e8e4dc; }
  .nav-links { display: flex; gap: 28px; }
  .nav-links a {
    color: #9ea8c8; text-decoration: none;
    font-size: .9rem; font-weight: 500;
  }
  .nav-links a:hover, .nav-links a.active { color: #c9a84c; }

  /* ─── Page header ─── */
  .community-header {
    background: linear-gradient(120deg, #111b35 0%, #1a1430 60%, #0f1a30 100%);
    border: 1px solid #2a3055;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .community-header::before {
    content: '🌐';
    position: absolute; right: 32px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: .08;
  }
  .community-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; color: #c9a84c; margin: 0 0 6px;
  }
  .community-header p { color: #7b89b8; font-size: .95rem; margin: 0; }

  /* ─── New post card ─── */
  .post-composer {
    background: #131728;
    border: 1px solid #242c48;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 22px;
  }
  .post-composer h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; color: #c9a84c;
    margin: 0 0 16px;
  }

  /* ─── Tag pills ─── */
  .tag-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: .72rem; font-weight: 600;
    border: 1px solid;
    margin: 2px;
  }
  .tag-lawyer    { background:#1a2a1a; border-color:#3a6a3a; color:#7ec87e; }
  .tag-victim    { background:#2a1a1a; border-color:#6a3a3a; color:#c87e7e; }
  .tag-news      { background:#1a1a2a; border-color:#3a3a6a; color:#7e7ec8; }
  .tag-court     { background:#2a2218; border-color:#6a5a30; color:#c8ae6a; }
  .tag-rights    { background:#1a2228; border-color:#3a5a68; color:#6ab0c8; }
  .tag-police    { background:#241820; border-color:#583050; color:#b87ea8; }
  .tag-family    { background:#1e2218; border-color:#4a5030; color:#a0b07e; }
  .tag-property  { background:#201e18; border-color:#504a30; color:#b0a07e; }
  .tag-labour    { background:#1e1a26; border-color:#4a3a60; color:#9a7eb8; }
  .tag-const     { background:#182026; border-color:#304860; color:#7ea0b8; }
  .tag-general   { background:#202020; border-color:#484848; color:#909090; }

  /* ─── Post card ─── */
  .post-card {
    background: #12162a;
    border: 1px solid #1e2540;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: border-color .2s, box-shadow .2s;
  }
  .post-card:hover {
    border-color: #3a4470;
    box-shadow: 0 4px 24px rgba(0,0,0,.4);
  }
  .post-card .post-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; color: #d4c9a8;
    margin: 0 0 8px;
  }
  .post-card .post-meta {
    font-size: .75rem; color: #4a5580;
    margin-bottom: 10px;
  }
  .post-card .post-body {
    font-size: .9rem; color: #9ea8c8;
    line-height: 1.6; margin-bottom: 12px;
  }
  .post-card .post-footer {
    display: flex; gap: 12px; align-items: center;
    font-size: .78rem; color: #4a5580;
    border-top: 1px solid #1e2540;
    padding-top: 10px;
    margin-top: 4px;
  }

  /* ─── Comment ─── */
  .comment-box {
    background: #0e1120;
    border-left: 3px solid #2a3460;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: .87rem; color: #8a94b8;
  }
  .comment-box .comment-author {
    font-size: .72rem; font-weight: 600;
    color: #c9a84c; margin-bottom: 4px;
  }

  /* ─── Filter bar ─── */
  .filter-bar {
    background: #111428;
    border: 1px solid #1e2540;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 18px;
    font-size: .85rem;
  }

  /* ─── Divider ─── */
  hr { border-color: #1e2540 !important; }

  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0c0e1a; }
  ::-webkit-scrollbar-thumb { background: #2e3355; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Global reset */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f1a;
    color: #e8e4dc;
  }

  /* ── Custom navbar */
  .lawgic-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 32px;
    background: linear-gradient(90deg,#111327 0%,#1a1e38 100%);
    border-bottom: 1px solid #2e3355;
    margin-bottom: 8px;
    border-radius: 0 0 16px 16px;
  }
  .lawgic-nav .logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700;
    color: #c9a84c;
    letter-spacing: .5px;
  }
  .lawgic-nav .logo span { color: #e8e4dc; }
  .nav-links { display: flex; gap: 28px; }
  .nav-links a {
    color: #9ea8c8; text-decoration: none;
    font-size: .9rem; font-weight: 500;
    transition: color .2s;
  }
  .nav-links a:hover, .nav-links a.active { color: #c9a84c; }

  /* ── Hero strip */
  .hero-strip {
    background: linear-gradient(100deg,#1a1e38 0%,#111b2e 100%);
    border: 1px solid #2e3355;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 20px;
  }
  .hero-strip h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; margin: 0 0 6px;
    color: #c9a84c;
  }
  .hero-strip p { margin: 0; color: #9ea8c8; font-size: .95rem; }

  /* ── Chat bubbles */
  .chat-wrapper { max-height: 60vh; overflow-y: auto; padding: 4px 0; }
  .bubble {
    padding: 14px 18px;
    border-radius: 14px;
    margin: 8px 0;
    max-width: 82%;
    line-height: 1.65;
    font-size: .93rem;
  }
  .bubble.user {
    background: #1e2540;
    border: 1px solid #2e3a5e;
    margin-left: auto;
    border-bottom-right-radius: 4px;
  }
  .bubble.ai {
    background: #141928;
    border: 1px solid #252d4a;
    border-bottom-left-radius: 4px;
  }
  .bubble .sender {
    font-size: .75rem; font-weight: 600;
    letter-spacing: .6px; margin-bottom: 6px;
  }
  .bubble.user .sender { color: #c9a84c; }
  .bubble.ai   .sender { color: #6b82d6; }

  /* ── Source badge */
  .src-badge {
    display: inline-block;
    background: #1a2236;
    border: 1px solid #2e3a5e;
    border-radius: 6px;
    padding: 3px 9px;
    font-size: .72rem;
    color: #7b93c8;
    margin: 2px;
  }

  /* ── Status pill */
  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: .78rem; font-weight: 500;
  }
  .status-pill.ready   { background:#132213; border:1px solid #2a4a2a; color:#5dba5d; }
  .status-pill.not-ready{ background:#221313; border:1px solid #4a2a2a; color:#ba5d5d; }

  /* ── Sidebar tweaks */
  [data-testid="stSidebar"] {
    background: #111327 !important;
    border-right: 1px solid #2e3355 !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    width: 100%; border-radius: 8px;
  }

  /* ── Input area */
  .stChatInput textarea {
    background: #1a1e38 !important;
    border-color: #2e3355 !important;
    color: #e8e4dc !important;
  }

  /* ── Scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0d0f1a; }
  ::-webkit-scrollbar-thumb { background: #2e3355; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# Session state 
if "session_id"      not in st.session_state: st.session_state.session_id      = str(uuid.uuid4())
if "expanded_post"   not in st.session_state: st.session_state.expanded_post   = None
if "compose_open"    not in st.session_state: st.session_state.compose_open    = False
if "tag_filter"      not in st.session_state: st.session_state.tag_filter      = "All"


# Tag style mapper
TAG_CLASS_MAP = {
    "⚖️ Lawyer":        "tag-lawyer",
    "🙋 Victim":        "tag-victim",
    "📰 News":          "tag-news",
    "🏛️ Court Ruling":  "tag-court",
    "📖 Rights & Laws": "tag-rights",
    "🚨 Police & FIR":  "tag-police",
    "👩‍👧 Family Law":  "tag-family",
    "🏠 Property":      "tag-property",
    "💼 Labour Law":    "tag-labour",
    "🌐 Constitutional":"tag-const",
    "❓ General Query": "tag-general",
}

def render_tags(tags: list[str]) -> str:
    parts = []
    for t in tags:
        cls = TAG_CLASS_MAP.get(t, "tag-general")
        parts.append(f'<span class="tag-pill {cls}">{t}</span>')
    return " ".join(parts)


# Top action bar 
col_btn, col_spacer = st.columns([2, 8])
with col_btn:
    if st.button("✏️  New Post", use_container_width=True, type="primary"):
        st.session_state.compose_open = not st.session_state.compose_open


# Composer 
if st.session_state.compose_open:
    st.markdown('<div class="post-composer"><h3>✏️ Create a New Post</h3></div>', unsafe_allow_html=True)

    with st.form("compose_form", clear_on_submit=True):
        author_col, _ = st.columns([3, 7])
        with author_col:
            author = st.text_input("Your name", placeholder="Anonymous", max_chars=50)

        title  = st.text_input("Post title *", placeholder="E.g. What does Article 32 say about writs?", max_chars=120)
        body   = st.text_area("Post body *", placeholder="Describe your question or insight in detail…", height=140, max_chars=2000)
        tags   = st.multiselect("Tags", options=get_all_tags(), default=["❓ General Query"], max_selections=3)

        sub_col, cancel_col, _ = st.columns([2, 2, 6])
        submitted = sub_col.form_submit_button("🚀 Publish", type="primary", use_container_width=True)
        cancelled = cancel_col.form_submit_button("Cancel", use_container_width=True)

        if submitted:
            try:
                create_post(author or "Anonymous", title, body, tags)
                st.success("Post published!")
                st.session_state.compose_open = False
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

        if cancelled:
            st.session_state.compose_open = False
            st.rerun()


# Filter bar 
all_tags = ["All"] + get_all_tags()
st.markdown('<div class="filter-bar"><b>Filter by tag:</b></div>', unsafe_allow_html=True)
selected_tag = st.selectbox(
    "Filter",
    options=all_tags,
    index=all_tags.index(st.session_state.tag_filter),
    label_visibility="collapsed",
)
if selected_tag != st.session_state.tag_filter:
    st.session_state.tag_filter   = selected_tag
    st.session_state.expanded_post = None
    st.rerun()


# Posts feed 
posts = get_posts(tag_filter=st.session_state.tag_filter)

if not posts:
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#2e3560;">
      <div style="font-size:3rem">📭</div>
      <div style="font-size:1.1rem;margin-top:12px;font-family:'Playfair Display',serif;color:#3d4770;">
        No posts yet. Be the first to start a discussion!
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for post in posts:
        pid         = post["id"]
        is_expanded = st.session_state.expanded_post == pid
        tags_html   = render_tags(post.get("tags", []))

        comment_count = post.get("comment_count", 0)
        liked         = has_liked(pid, st.session_state.session_id)
        like_label    = f"{'❤️' if liked else '🤍'} {post.get('likes', 0)}"

        # ── Post card ──
        st.markdown(f"""
        <div class="post-card">
          <div class="post-title">{post["title"]}</div>
          <div class="post-meta">
            👤 {post["author"]} &nbsp;·&nbsp; 🕐 {format_time_ago(post["created_at"])}
            &nbsp;·&nbsp; 💬 {comment_count} comment{"s" if comment_count != 1 else ""}
            &nbsp;·&nbsp; ❤️ {post.get("likes", 0)}
          </div>
          {tags_html}
          <div class="post-body" style="margin-top:10px;">{post["body"][:350]}{"…" if len(post["body"]) > 350 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        # Action row
        btn_a, btn_b, _ = st.columns([2, 2, 12])

        if btn_a.button("💬 Comments", key=f"exp_{pid}", use_container_width=True):
            st.session_state.expanded_post = None if is_expanded else pid
            st.rerun()

        if btn_b.button(like_label, key=f"like_{pid}", use_container_width=True):
            try:
                toggle_like(pid, st.session_state.session_id)
            except Exception as exc:
                st.error(f"Could not toggle like: {exc}")
            st.rerun()

        if is_expanded:
            # Comments section 
            comments = get_comments(pid)
            st.markdown(f"**💬 Comments ({len(comments)})**")

            if comments:
                for c in comments:
                    st.markdown(f"""
                    <div class="comment-box">
                      <div class="comment-author">👤 {c["author"]} · {format_time_ago(c["created_at"])}</div>
                      {c["body"]}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No comments yet — be the first!")

            # Add comment
            with st.form(f"comment_form_{pid}", clear_on_submit=True):
                c_author = st.text_input("Name", placeholder="Anonymous", max_chars=50, key=f"ca_{pid}")
                c_body   = st.text_area("Comment *", placeholder="Share your thoughts…", height=80, max_chars=800, key=f"cb_{pid}")
                if st.form_submit_button("Post Comment", type="primary"):
                    try:
                        add_comment(pid, c_author or "Anonymous", c_body)
                        st.success("Comment added!")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        st.markdown("<hr>", unsafe_allow_html=True)