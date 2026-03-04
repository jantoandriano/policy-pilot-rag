import os
import yaml
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Import your chunking function
from processing import clean_and_chunk 

# Load environment variables (API Key)
load_dotenv()

# ==========================================
# BULLETPROOF PATH RESOLUTION
# ==========================================
# 1. Get the folder where THIS script lives (ingestion/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Get the root project folder (policy-pilot-rag/)
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 3. Build the exact absolute path to config.yaml
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def main():
    # Construct exact path to the data folder
    data_path = os.path.join(ROOT_DIR, "data")
    print(f"1. Scanning for PDFs inside: {data_path}")
    
    loader = PyPDFDirectoryLoader(data_path)
    raw_docs = loader.load()
    
    # --- CIRCUIT BREAKER 1 ---
    if not raw_docs:
        print(f"❌ STOPPING: No PDFs found! Please put a .pdf file in:\n{data_path}")
        return
    # -------------------------

    print(f"✅ Found {len(raw_docs)} pages. Cleaning and chunking text...")
    chunks = clean_and_chunk(raw_docs)
    
    # --- CIRCUIT BREAKER 2 ---
    if not chunks:
        print("❌ STOPPING: Text splitting resulted in an empty list.")
        return
    # -------------------------

    print(f"✅ Created {len(chunks)} chunks. Connecting to OpenAI to generate embeddings...")
    embeddings = OpenAIEmbeddings(model=config.get("embedding_model", "text-embedding-3-small"))
    
    # Construct exact path to the vector database folder
    db_relative_path = config.get("db_path", "vector_db/chroma_storage")
    db_path = os.path.join(ROOT_DIR, db_relative_path)
    print(f"3. Upserting to Vector Database at: {db_path}")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"🎉 Success! Ingested {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    main()