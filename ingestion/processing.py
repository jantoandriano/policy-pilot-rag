import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def clean_and_chunk(raw_docs):
    """Splits documents based on config settings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    return splitter.split_documents(raw_docs)