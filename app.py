import logging
import shutil
import tempfile
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import HumanMessage

from chatbot import ask
from config import DATA_DIR, GROQ_API_KEY, validate_config
from connect_memory import is_vectorstore_ready

log = logging.getLogger(__name__)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Lawgic AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "lc_messages"   not in st.session_state: st.session_state.lc_messages   = []
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())

st.markdown("""
<style>
.lawgic-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    background: #111327;
    border-bottom: 1px solid #2a2f4a;
    border-radius: 10px;
    margin-bottom: 15px;
}

.lawgic-logo {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e8e4dc;
    display: flex;
    align-items: center;
    gap: 6px;
}

.lawgic-logo span {
    color: #c9a84c;
}

.lawgic-links {
    display: flex;
    gap: 18px;
}

.lawgic-links a {
    text-decoration: none;
    color: #9aa3c2;
    font-size: 0.9rem;
    padding: 6px 10px;
    border-radius: 6px;
    transition: all 0.2s ease;
}

.lawgic-links a:hover {
    background: #1e2340;
    color: #ffffff;
}

.lawgic-links a.active {
    background: #c9a84c;
    color: #111327;
    font-weight: 600;
}
</style>

""", unsafe_allow_html=True)



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


# Chat area
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#3d4770;">
          <div style="font-size:3rem">⚖️</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin:10px 0;color:#5a6490;">
            Ask me anything about the Constitution of India
          </div>
          <div style="font-size:.85rem;color:#3d4770;margin-top:8px;">
            Try: "What are the Fundamental Rights?" · "Explain Article 21" · "What is the Preamble?"
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f"""
                <div class="bubble user">
                  <div class="sender">YOU</div>
                  {turn['content']}
                </div>""", unsafe_allow_html=True)
            else:
                sources_html = ""
                if turn.get("sources"):
                    badges = "".join(
                        f'<span class="src-badge">📄 {s}</span>'
                        for s in turn["sources"]
                    )
                    sources_html = f"<div style='margin-top:8px;'><b style='font-size:.75rem;color:#4a5680;'>SOURCES</b><br>{badges}</div>"

                st.markdown(f"""
                <div class="bubble ai">
                  <div class="sender">LAWGIC AI</div>
                  {turn['content']}
                  {sources_html}
                </div>""", unsafe_allow_html=True)


# Chat input
if prompt := st.chat_input("Ask about the Constitution of India…"):
    if not GROQ_API_KEY:
        st.error("Please set GROQ_API_KEY in your .env file.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.lc_messages.append(HumanMessage(content=prompt))

    with st.spinner("Consulting the Constitution…"):
        result = ask(query=prompt, history=st.session_state.lc_messages[:-1])

    answer = result["answer"]
    sources = list({
        doc.metadata.get("source_file", "document")
        for doc in result.get("graded_docs", [])
    })

    st.session_state.chat_history.append({
        "role":    "ai",
        "content": answer,
        "sources": sources,
    })

    st.rerun()