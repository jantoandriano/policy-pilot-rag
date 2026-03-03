import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def run_ingestion(file_path: str):
    print(f"--- Starting Ingestion for: {file_path} ---")
    
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Chunking (The 'Senior' settings: 1000 chars with 20% overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True # Good for referencing exactly where info was found
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Add Metadata (Crucial for filtering later)
    for chunk in chunks:
        chunk.metadata["policy_type"] = "reimbursement"
        chunk.metadata["source_file"] = os.path.basename(file_path)
    
    # 4. Create and Persist the Vector Store
    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=os.getenv("DATABASE_PATH")
    )
    
    print("--- Ingestion Complete! ---")
    return vector_db

if __name__ == "__main__":
    # Test it with a file
    # Ensure you put a pdf in data/reimbursement/policy.pdf
    pdf_path = "data/reimbursement/policy.pdf"
    if os.path.exists(pdf_path):
        run_ingestion(pdf_path)
    else:
        print(f"Please place a PDF at {pdf_path} to test.")