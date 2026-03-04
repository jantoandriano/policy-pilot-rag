# ingestion/loaders/load_pdfs.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def get_pdf_documents(directory_path="./data/pdfs"):
    print(f"Scanning {directory_path} for PDFs...")
    
    # Uses PyPDFLoader under the hood for every .pdf file found
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    
    # Standardize Metadata
    for doc in docs:
        doc.metadata["channel"] = "local_files"
        doc.metadata["file_type"] = "pdf"
        
    return docs