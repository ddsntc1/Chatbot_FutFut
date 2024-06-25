import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from unsloth import FastLanguageModel

import dotenv
dotenv()


'''
FastLanguageModel 사용하여 모델 불러오기

Fine-tuning을 도와주는 Unsloth 패키지를 사용하여 모델 불러오기
불러오는 모델 repo안에 adapter_config가 존재하면 안된다.

빠른 추론 할 수 있도록 도와준다.

'''

max_seq_length = 2048
hf_token = os.getenv('HUGGINGFACE_TOKEN')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Dongwookss/last_small_pre", # adapter_config가 존재하지 않는 모델명으로 불러와야한다.
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=False,
    token = hf_token,
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