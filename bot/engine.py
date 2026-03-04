# bot/engine.py
import yaml
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import our dedicated files!
from bot.prompts import contextualize_q_prompt, qa_prompt
from bot.retriever import get_company_retriever

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def build_bot_engine(user_filters=None):
    """Assembles the final RAG chain."""
    
    # 1. Get the Retriever (passing in any security filters)
    retriever = get_company_retriever(search_filters=user_filters)
    
    # 2. Initialize the LLM
    llm = ChatOpenAI(model=config["llm_model"], temperature=0)

    # 3. Build the chains
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain