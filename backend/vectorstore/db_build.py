# build_vector_db.py
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

DOC_PATH = os.getenv("DOC_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def build_vector_db():
    print("🚀 Building vector database from:", DOC_PATH)

    # Load your documentation
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Create and persist Chroma vector DB
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_PATH)
    vectordb.persist()

    print(f"✅ Vector DB created successfully at: {VECTOR_DB_PATH}")

if __name__ == "__main__":
    build_vector_db()
