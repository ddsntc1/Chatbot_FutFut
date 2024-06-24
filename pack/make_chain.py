from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings

def make_chain(retriever):
    def format_docs(docs):
        # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
    llm = ChatOllama(model="foot:latest",max_lengths=30, max_tokens=50)

    template2 = "\"```\" 문맥(`context`)으로부터 주어진 질문에 대한 답변을 제공하세요."\
    "텍스트에 제시된 주요 요점과 정보를 요약하는 명확하고 간결한 응답을 제공하세요."\
    "당신의 답변은 당신의 말로 이루어져야 하며 50 단어(100 tokens) 내로 대답하세요."\
    "문맥(`context`)에서 질문에 답할 수 없는 경우 '문맥에서 필요한 정보를 찾을 수 없습니다.'로 답하십시오."\
    "답을 모를 경우, 임의로 답변을 생성하지 않고 '문맥에서 필요한 정보를 찾을 수 없습니다.'로 답하십시오."\
    "make answer in korean. 한국어로 대답하세요"\
    "\n\nContext:\n{context}\n;"\
    "Question: {question}"\
    "\n\nAnswer:"

    prompt = ChatPromptTemplate.from_template(template2)

    rag_chain = (
    {"context": retriever| format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain
