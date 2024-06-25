
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from kiwipiepy import Kiwi
load_dotenv()

kiwi = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

def retriever(PCVDB, bm25list):
    PCretriever = PCVDB.as_retriever(search_kwargs={'k': 2})

    kiwi_bm25 = BM25Retriever.from_documents(bm25list, preprocess_func=kiwi_tokenize)
    kiwi_bm25.k=2

    kiwibm25_faiss_37 = EnsembleRetriever(
        retrievers=[kiwi_bm25, PCretriever],  # 사용할 검색 모델의 리스트
        weights=[0.3, 0.7],  # 각 검색 모델의 결과에 적용할 가중치
        search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
    )

    ensembleretrievers = kiwibm25_faiss_37

    return ensembleretrievers


