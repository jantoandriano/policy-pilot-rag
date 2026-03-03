from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

def get_rag_chain():
    vectorstore = Chroma(
        persist_directory=os.getenv("DATABASE_PATH"),
        embedding_function=OpenAIEmbeddings()
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 1. Define the System Prompt
    system_prompt = (
        "You are an HR assistant. Use the context to answer the question."
        "If you don't know the answer, say you don't know."
        "Limit your answer to three sentences max."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 2. Create the Chains
    # This chain handles combining the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # This chain handles the retrieval AND the answering
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain