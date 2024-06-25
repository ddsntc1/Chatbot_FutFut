import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

'''
AutoModelForCausalLm을 사용하여 모델 불러오기

속도가 느린편에 속하나 문제 없이 돌아갈 수 있는 안정적인 방식
'''
def load_Auto():
    model_id = 'Dongwookss/small_fut_final'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
    )

    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={"temperature": 0.7},
    )
    return llm