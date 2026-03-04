from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Prompt to rewrite the user's question based on chat history
# (e.g., turns "Does it roll over?" into "Does PTO roll over?")
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Prompt for the actual answer generation
qa_system_prompt = (
    "You are a helpful HR and IT assistant for our company. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say 'I cannot find the answer in the company policies.' "
    "Do not make up rules or policies. Keep the answer concise. \n\n"
    "Context: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])