from fastapi import FastAPI
import uvicorn

app = FastAPI(title="PolicyPilot RAG")

@app.get("/")
def read_root():
    return {"status": "PolicyPilot API is Online", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)