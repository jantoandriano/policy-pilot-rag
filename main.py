from fastapi import FastAPI
from src.core.rag_logic import get_rag_chain
import uvicorn

app = FastAPI()
chain = get_rag_chain()

@app.get("/ask")
async def ask_policy(question: str):
    # 'invoke' now returns a dictionary with 'answer' and 'context'
    result = chain.invoke({"input": question})
    
    # Extract unique source filenames
    sources = list(set([doc.metadata.get("source_file", "Unknown") for doc in result["context"]]))
    
    return {
        "answer": result["answer"],
        "sources": sources
    }