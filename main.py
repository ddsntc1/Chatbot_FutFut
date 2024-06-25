from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer

app = FastAPI()

# 모델과 토크나이저 불러오기
model_name = 'Dongwookss/small_fut_final'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# 요청 바디 모델 정의
class QueryRequest(BaseModel):
    query: str


# 응답 바디 모델 정의
class QueryResponse(BaseModel):
    response: str



def resp(input_text: str) -> str:

    PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.'''
    instruction = input_text

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
        ]

    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)

    terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    text_streamer = TextStreamer(tokenizer)
    response_text = model.generate(
        input_ids,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        streamer = text_streamer,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty = 1.1
    )
    return f"답변: {response_text}"



@app.post("/query", response_model=QueryResponse)
async def get_query_response(query_request: QueryRequest):
    try:
        # 쿼리 텍스트를 받아서 LLM 모델에 전달
        query_text = query_request.query
        response_text = resp(query_text)
        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Uvicorn을 통해 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
