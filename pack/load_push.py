import glob
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time 
from langchain_community.embeddings import SentenceTransformerEmbeddings

from dotenv import load_dotenv
load_dotenv()


# 데이터 받으면 갈라줘
def come_data(splits):
    docs = []
    for i in range(len(splits)):
        spcon = splits[i].page_content
        url = splits[i].metadata['source']
        con = Document(page_content=spcon, metadata={'source': url})
        docs.append(con)
    return docs





# 평탄화
def flatten_list(lst):
    return [item for sublist in lst for item in flatten_list(sublist)] if isinstance(lst, list) else [lst]


# 모델 불러와서 VectorDB로 올리는 부분
def all_files(path):
    print(f'RAG에 들어갈 모든 데이터는 {path}에 담아주세요.\n\n\n')
    f = glob.glob(path + '/**', recursive=True)
    f_docs = []
    for file in f:
        a = False
        if file.endswith('.txt'):
            loader = TextLoader(file)
            document = loader.load()
            a = True
        elif file.endswith('.csv'):
            loader = CSVLoader(file)
            document = loader.load()
            docs = come_data(document)
            f_docs.append(docs)
            a = False
        elif file.endswith('.pdf'):
            loader = PyMuPDFLoader(file)
            document = loader.load()
            a = True
        # ------------------- 파일 타입 추가 사항 있을 시 위에 추가 ----------------#
        if a:
            print(file.split('/')[-1] + ' split 진행 중')
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator=".",
                chunk_size=500,
                chunk_overlap=0,
            )
            splits = text_splitter.split_documents(document)
            docs = come_data(splits)
            f_docs.append(docs)
            print(file.split('/')[-1] + ' split 진행 완료. \n' + file.split('/')[-1] + ' split 갯수 : ' + str(len(docs)))
    flattened_list = flatten_list(f_docs)
    
    '''
    flattened 된 docs를 벡터 db로 넣어줄 것
    '''


    
    # 임베딩 모델 선언
    embedding_model = SentenceTransformerEmbeddings(model_name='BM-K/KoSimCSE-roberta-multitask', model_kwargs={"trust_remote_code":True}) 
    
    # 벡터스토어 선언

    api_key = os.environ['PINECONE_API_KEY']
    pc = Pinecone(api_key=api_key)   

    index_name = os.getenv('INDEX_NAME')

    print('Vector DB 초기화. Index_name = ' + str(index_name))
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    # 인덱스 존재여부 확인 및 삭제
    collect_name = []
    for n in pc.list_indexes().indexes:
        collect_name.append(n.name)
    
    if index_name in collect_name:  
        pc.delete_index(index_name) 
        print('기존 인덱스 삭제완료')  
    time.sleep(3)
    
    # 파인콘 인덱스 생성
    pc.create_index(  
        index_name,  
        dimension=768, 
        metric='cosine',  
        spec=spec  
    )  
    
    # 인덱스 재생성 및 데이터 입력
    # index = pc.Index(index_name)
    print('Vector DB 들어가는 중. Index_name = ' + str(index_name))

    # # 텍스트 임베딩 생성
    # texts = [doc.page_content for doc in flattened_list]
    # embedded_texts = []
    # for txt in texts:
    #     embedded_texts.append(embedding_model.embed_query(txt))

    
    # # 벡터 DB에 임베딩 추가
    # ids = [str(i) for i in range(len(embedded_texts))]
    # metadata = [doc.metadata for doc in flattened_list]
    
    # # db올릴때 무료버전이기때문에 용량 터짐 -> 나눠서 올리자
    # batch_size = 28
    # for i in range(0, len(embedded_texts), batch_size):
    #     batch_vectors = [{"id": id, "values": vector, "metadata": meta} for id, vector, meta in zip(ids[i:i + batch_size], embedded_texts[i:i + batch_size], metadata[i:i + batch_size])]
    #     index.upsert(vectors=batch_vectors)
        
    
    Vectorstore = PineconeVectorStore.from_documents(
    documents=flattened_list,
    index_name=index_name,
    embedding=embedding_model
    )

    print('저장 완료')
    return Vectorstore, flattened_list