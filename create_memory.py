from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


load_dotenv()

# Access the HF token from the environment
hf_token = os.getenv("HF_TOKEN")
 
DATA_PATH = "data"

#storing data in form of pdf to be used for training

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("length of documents: ", len(documents))

#extracting text from pdf files in form of chunks

def create_chunks(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(extracted_data)
    return chunks


chunks = create_chunks(documents)
print("length of chunks: ", len(chunks))

#converting the chunks to embeddings (vector embeddings so machine can understand)

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

#store embeddings in FAISS 

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)



