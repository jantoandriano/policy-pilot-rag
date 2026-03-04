import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load configuration settings
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_company_retriever(search_filters: dict = None, top_k: int = 3):
    """
    Connects to the vector database and configures how we search it.
    
    Args:
        search_filters (dict, optional): Restrict search to specific metadata. 
                                         Example: {"department": "HR"}
        top_k (int): Number of document chunks to return.
    """
    
    print("Connecting to Vector Database...")
    
    # 1. Initialize the embedding model (must match the one used during ingestion)
    embeddings = OpenAIEmbeddings(model=config["embedding_model"])
    
    # 2. Connect to the local Chroma database
    vector_db = Chroma(
        persist_directory=config["db_path"],
        embedding_function=embeddings
    )
    
    # 3. Set up the search parameters
    search_kwargs = {"k": top_k}
    
    # 4. Apply security/context filters if they exist
    if search_filters:
        search_kwargs["filter"] = search_filters
        
    # 5. Convert the database into a LangChain Retriever
    # Using 'similarity' search as the default
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    
    return retriever