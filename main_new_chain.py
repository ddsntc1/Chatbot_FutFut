
from pack.load_push import *
from pack.retriever import *
from pack.retrieve_docs import *
from load_model_for_newchain import load_model_token
from pack.make_answer import *
from for_nochain.mt_chat import chat
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

model, tokenizer = load_model_token()
pinecone,bm25 = all_files('files')
retriever=retriever(pinecone,bm25)



# 요청 바디 모델 정의
class QueryRequest(BaseModel):
    query: str


# 응답 바디 모델 정의
class QueryResponse(BaseModel):
    response: str


@app.post("/query", response_model=QueryResponse)
async def get_query_response(query_request: QueryRequest):
    try:
        # 쿼리 텍스트를 받아서 LLM 모델에 전달
        query_text = query_request.query
        response_text = chat(model, tokenizer,retriever,query_text)
        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

