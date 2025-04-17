from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS
import requests

# Set Hugging Face token (if not set in environment)
if os.getenv("HF_TOKEN") is None:
    os.environ["HF_TOKEN"] = "your_huggingface_api_token"

# APIs

HF_TOKEN = os.getenv("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

API_MARKET = os.getenv("API_MARKET")
url = 'https://api.magicapi.dev/api/v1/sarvam/ai-models/translate'

# Setup LLM with Hugging Face model

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN, 
        model_kwargs={"max_length": 512}
    )
    return llm

# FAISS database path
DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom prompt template
custom_prompt_template = """Use the pieces of information provided in context to answer user's question in a simple user friendly language.
If you don't know the answer, tell the user to contact any legal official for such information.
Don't try to answer anything out of the given context.

Context: {context}
Question: {question}"""

def custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load FAISS database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt(custom_prompt_template)}
)

# Invoke with a single query
user_query = input("Write your query here: ")
response = qa_chain.invoke({'query': user_query})  # Fixed key

print("Result:", response["result"])
print("Source documents:", response["source_documents"])  # Fixed typo

result = response["result"]

headers = {
    'x-magicapi-key': API_MARKET,
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

data = {
    "input": result,
    "source_language_code": "en-IN",
    "target_language_code": "hi-IN",
    "speaker_gender": "Male",
    "mode": "formal",
    "model": "mayura:v1",
    "enable_preprocessing": False
}

response = requests.post(url, headers=headers, json=data)

try:
    translated_text = response.json()
    print(translated_text)
except Exception as e:
    print("Error:", e)
    print("Response text:", response.text)

response = response["result"]
