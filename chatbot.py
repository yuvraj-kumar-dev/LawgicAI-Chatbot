import logging
from typing import Annotated, TypedDict, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import GROQ_API_KEY, LLM_MODEL
from connect_memory import get_retriever, is_vectorstore_ready

log = logging.getLogger(__name__)

# Graph topology:
#     START -> retrieve -> grade_docs -> generate -> check_hallucination -> END
#                            | (no relevant docs)
#                         no_docs → END

# Nodes:
#   retrieve            : fetch top-k docs from FAISS
#   grade_docs          : filter docs for relevance using LLM
#   generate            : produce answer grounded in context
#   check_hallucination : verify the answer is grounded (retry once if not)
#   no_docs             : graceful fallback when nothing relevant is found


# State

class RAGState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    query:           str
    retrieved_docs:  list[Document]
    graded_docs:     list[Document]
    answer:          str
    hallucination:   bool   # True if answer is NOT grounded
    retry_count:     int


# LLM

def _build_llm(temperature: float = 0.0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Set it in your .env file.")
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )


# Prompt Template

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance grader for a legal AI system about the Constitution of India. "
        "Given a user question and a retrieved document chunk, decide whether the chunk is "
        "relevant to answering the question.\n"
        "Respond with ONLY 'yes' or 'no'.",
    ),
    ("human", "Question: {query}\n\nDocument chunk:\n{doc_content}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Lawgic AI - an expert legal assistant specialised in the Constitution of India.
Your answers must be:
- Grounded ONLY in the provided context documents.
- Accurate, structured, and easy to understand for a non-lawyer.
- Cite the relevant Article, Part, or Schedule when applicable.
- If the context does not contain enough information, say so clearly instead of guessing.

Context documents:
{context}

Remember: Do NOT hallucinate. Every claim must be traceable to the context above.""",
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{query}"),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a hallucination checker for a legal AI.\n"
        "Given a generated answer and the source context used to produce it, "
        "decide whether the answer is FULLY grounded in the context.\n"
        "Respond with ONLY 'grounded' or 'hallucinated'.",
    ),
    (
        "human",
        "Context:\n{context}\n\nAnswer:\n{answer}",
    ),
])

NO_DOCS_RESPONSE = (
    "I'm sorry, I couldn't find relevant information in the uploaded documents "
    "to answer your question."
)


# Graph nodes 

def retrieve_node(state: RAGState) -> dict:
    """Retrieve top-k relevant document chunks from FAISS."""
    query     = state["query"]
    retriever = get_retriever()
    docs      = retriever.invoke(query)
    log.info("Retrieved %d docs for query: %.60s…", len(docs), query)
    return {"retrieved_docs": docs}


def grade_docs_node(state: RAGState) -> dict:
    """Filter retrieved docs for relevance using the LLM as a judge."""
    llm    = _build_llm()
    chain  = GRADER_PROMPT | llm | StrOutputParser()
    query  = state["query"]
    graded = []

    for doc in state["retrieved_docs"]:
        verdict = chain.invoke({"query": query, "doc_content": doc.page_content}).strip().lower()
        if verdict.startswith("yes"):
            graded.append(doc)

    log.info("Graded docs: %d/%d relevant", len(graded), len(state["retrieved_docs"]))
    return {"graded_docs": graded}


def generate_node(state: RAGState) -> dict:
    """Generate an answer grounded in the graded context."""
    llm     = _build_llm(temperature=0.1)
    chain   = RAG_PROMPT | llm | StrOutputParser()
    context = "\n\n---\n\n".join(doc.page_content for doc in state["graded_docs"])
    answer  = chain.invoke({
        "context":  context,
        "messages": state["messages"],
        "query":    state["query"],
    })
    log.info("Generated answer (%d chars)", len(answer))
    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }


def check_hallucination_node(state: RAGState) -> dict:
    """Verify the answer is grounded in the source context."""
    llm     = _build_llm()
    chain   = HALLUCINATION_PROMPT | llm | StrOutputParser()
    context = "\n\n---\n\n".join(doc.page_content for doc in state["graded_docs"])
    verdict = chain.invoke({"context": context, "answer": state["answer"]}).strip().lower()
    is_hallucinated = not verdict.startswith("grounded")
    log.info("Hallucination check: %s", "HALLUCINATED" if is_hallucinated else "grounded")
    return {
        "hallucination": is_hallucinated,
        "retry_count":   state.get("retry_count", 0) + 1,
    }


def no_docs_node(state: RAGState) -> dict:
    """Fallback when no relevant documents are found."""
    return {
        "answer":   NO_DOCS_RESPONSE,
        "messages": [AIMessage(content=NO_DOCS_RESPONSE)],
    }


# Edges

def route_after_grading(state: RAGState) -> Literal["generate", "no_docs"]:
    return "generate" if state["graded_docs"] else "no_docs"


def route_after_hallucination_check(
    state: RAGState,
) -> Literal["generate", END]:           # type: ignore[valid-type]
    # Retry generation once if hallucinated, then accept on second attempt
    if state["hallucination"] and state.get("retry_count", 0) < 2:
        log.warning("Retrying generation due to hallucination…")
        return "generate"
    return END


# Graph 

def build_rag_graph() -> StateGraph:
    """Compile and return the LangGraph RAG pipeline."""
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("retrieve",            retrieve_node)
    graph.add_node("grade_docs",          grade_docs_node)
    graph.add_node("generate",            generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)
    graph.add_node("no_docs",             no_docs_node)

    # Edges
    graph.add_edge(START,          "retrieve")
    graph.add_edge("retrieve",     "grade_docs")
    graph.add_conditional_edges("grade_docs", route_after_grading)
    graph.add_edge("generate",     "check_hallucination")
    graph.add_conditional_edges("check_hallucination", route_after_hallucination_check)
    graph.add_edge("no_docs",      END)

    return graph.compile()


# Public API

_compiled_graph = None  # module-level singleton


def get_graph():
    """Return the compiled RAG graph (built once, reused)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph


def ask(query: str, history: list[BaseMessage] | None = None) -> dict:
    """
    Run the RAG pipeline for a single user query.

    Args:
        query:   The user's question.
        history: Previous chat messages for multi-turn context.

    Returns:
        dict with keys: answer, graded_docs, hallucination
    """
    if not is_vectorstore_ready():
        return {
            "answer": (
                "⚠️ The knowledge base is not ready yet. "
                "Please upload a PDF and click **Build Knowledge Base** first."
            ),
            "graded_docs":  [],
            "hallucination": False,
        }

    graph = get_graph()
    initial_state: RAGState = {
        "messages":       history or [],
        "query":          query,
        "retrieved_docs": [],
        "graded_docs":    [],
        "answer":         "",
        "hallucination":  False,
        "retry_count":    0,
    }

    result = graph.invoke(initial_state)
    return {
        "answer":        result["answer"],
        "graded_docs":   result.get("graded_docs", []),
        "hallucination": result.get("hallucination", False),
    }