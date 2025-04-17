import streamlit as st
import os
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load env vars (HF_TOKEN and API_MARKET should be in .env file)
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_MARKET = os.getenv("API_MARKET")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "C:/Users/yuvra/OneDrive/Desktop/legal chatbot/vectorstore/db_faiss"

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
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        model_kwargs={"max_length": 512}
    )

def custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.title("📚 Lawgic AI Chatbot")
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
            llm = load_llm()

            template = """Use the pieces of information provided in context to answer user's question in a simple user friendly language.
If you don't know the answer, tell the user to contact any legal official for such information.
Don't try to answer anything out of the given context.

Context: {context}
Question: {question}"""

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': custom_prompt(template)}
            )

            response = qa_chain.invoke({'query': user_input})
            english_response = response["result"]
            if "Answer:" in english_response:
                english_response = english_response.split("Answer:")[-1].strip()
            sources = response["source_documents"]

            final_response = english_response

            if language == "Hindi":
                final_response = translate_text(english_response, "en-IN", "hi-IN")

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
