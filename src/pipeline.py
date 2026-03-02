"""
pipeline.py — RAG Pipeline
Imports vectorstore from ingestion.py
Builds the prompt template, LLM and LCEL chain.
"""

from dotenv import load_dotenv
load_dotenv()

from ingestion import vectorstore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  Configuration 

CHAT_MODEL = "llama3.2:3b"
TEMPERATURE = 0   
NUM_PREDICT = -1

#  Prompt Template 

SYSTEM_PROMPT = """
You are a helpful customer service assistant.
Answer the user's question based ONLY on the provided context documents below.
Do NOT use any outside knowledge or training data.
If the answer is not found in the context, respond ONLY with: 'I don't have information about that in my knowledge base.'
Do not guess.
Be concise, professional, and friendly.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

#  LLM 

llm = ChatOllama(model=CHAT_MODEL, temperature=TEMPERATURE, num_predict=NUM_PREDICT)

#  LCEL Chain 

rag_chain = prompt | llm | StrOutputParser()
