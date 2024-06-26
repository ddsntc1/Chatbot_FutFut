from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings

def make_chain_llm(retriever,llm):
    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
    # llm = ChatOllama(model="zephyr:latest")

    template = "\"```\" Below is an instruction that describes a task. Write a response that appropriately completes the request."\
    "제시하는 context에서만 대답하고 context에 없는 내용은 생성하지마"\
    "make answer in korean. 한국어로 대답하세요"\
    "\n\nContext:\n{context}\n;"\
    "Question: {question}"\
    "\n\nAnswer:"

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
    {"context": retriever| format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain
