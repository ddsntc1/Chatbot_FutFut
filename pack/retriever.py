from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

embedding_model = SentenceTransformerEmbeddings(model_name='BM-K/KoSimCSE-roberta-multitask', model_kwargs={"trust_remote_code":True}) 

def retriever():
        # Pinecone vector store 초기화  
    vectorstore = PineconeVectorStore(  
    index_name=os.getenv("INDEX_NAME"), embedding=embedding_model
    )  

    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    return retriever


