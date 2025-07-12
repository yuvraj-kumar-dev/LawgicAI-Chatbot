import streamlit as st
import os
import requests
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
API_MARKET = os.getenv("API_MARKET")
DB_FAISS_PATH = "C:/Users/yuvra/OneDrive/Desktop/legal chatbot/vectorstore/db_faiss"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Translation API
SARVAM_URL = "https://api.magicapi.dev/api/v1/sarvam/ai-models/translate"

def translate_text(text, source_lang, target_lang):
    headers = {
        'x-magicapi-key': API_MARKET,
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    data = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False
    }

    try:
        response = requests.post(SARVAM_URL, headers=headers, json=data)
        return response.json().get("result", text)
    except Exception as e:
        return f"[Translation Error] {str(e)}"

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
        st.error("FAISS index not found. Please create it before running the app.")
        st.stop()
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

#update the load_llm function to use the InferenceClient
def load_llm():
    return InferenceClient(
        model=MODEL_ID,
        provider="together",
        token=HF_TOKEN
    )

def get_llm_answer(llm, question, context):
    prompt = f"""
You are LawgicAI, a legal information assistant. Only answer questions strictly related to the legal context provided.

- If the user's question is not clearly about a legal matter, politely reply:
  "I can only help with legal questions. Please ask a question related to law or legal information."

- If you do not find the answer in the context, reply:
  "Please contact a legal official for such information."

Use only the context provided below.

Context:
{context}

Question:
{question}
"""
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.5
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="LawgicAI Chatbot", page_icon="📚")
    st.title("📚 LawgicAI Chatbot")
    language = st.radio("Choose your language:", options=["English", "Hindi"])

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    user_input = st.chat_input("Ask your legal question here...")

    if user_input:
        original_input = user_input

        if language == "Hindi":
            user_input = translate_text(user_input, "hi-IN", "en-IN")

        st.chat_message("user").markdown(original_input)
        st.session_state.messages.append({"role": "user", "content": original_input})

        try:
            vectorstore = get_vectorstore()
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])

            llm = load_llm()
            response = get_llm_answer(llm, user_input, context)

            final_response = response
            if language == "Hindi":
                final_response = translate_text(response, "en-IN", "hi-IN")

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
