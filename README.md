# Chatbot_FutFut

### About FutFut
<img src="https://github.com/ddsntc1/Chatbot_FutFut/assets/38596856/cb1cd8b7-c556-46a8-ab8d-e093af713433.jpg" width="400" height="400">

- Domain : 풋살 플랫폼 
- Concept : '해요'체를 사용하며 친절하게 답하는 챗봇. 말끝에 '언제든지 물어보세요! 풋풋~!'을 붙여 풋풋이 컨셉을 유지 
- Model : 
- Dataset :
  
  [Dongwookss/q_a_korean_futsal](https://huggingface.co/datasets/Dongwookss/q_a_korean_futsal)
  
  [mintaeng/llm_futsaldata_yo](https://huggingface.co/datasets/mintaeng/llm_futsaldata_yo)
- How-to? LLM fine-tuning & RAG 
---
### How to use? 
- Fine-tuned Model : Llama3-8b, Zephyr-7b

<details>
  <summary>Scripts Description</summary>
  
   load_model_type_a.py : transformers의 AutoModelForCausalLM을 이용하여 모델을 불러옵니다
    
   load_model_type_b.py : Unsloth 패키지의 FastLanguageModel을 이용하여 모델을 불러옵니다. 답변 생성속도가 빠르지만 튜닝을 위한 패키지이기 때문에 Huggingface에 adapter_config가 존재하면 모델을 불러오지 못합니다.

   main.py : pack에 있는 모듈을 활용하여 Fine-tuned Model을 불러오고 RAG를 적용시켜 FastAPI로 요청과 응답을 받을 수 있습니다.

   ```python
  uvicorn main:app --reload -p <포트번호지정>
  ```

</details>

  
---


#### About Fine-tuning

- Method : By Unsloth
- Trainer : SFTrainer, ~~DPOTrainer~~

#### About RAG

- Method

#### Fine-tuned Result(HuggingFace🤗): 


- [Dongwooks](https://huggingface.co/Dongwookss)


- [mintaeng](https://huggingface.co/mintaeng)

- [bigr00t](https://huggingface.co/bigr00t)

<details>
  <summary>Using HuggingFace Model with out RAG </summary>
  
``` python
# Using HuggingFace Model with out RAG 
# !pip install transformers==4.40.0 accelerate

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer

model_id = 'Dongwookss/원하는모델'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

PROMPT = '''
Below is an instruction that describes a task. Write a response that appropriately completes the request.
'''
instruction = "question"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text_streamer = TextStreamer(tokenizer)
output = model.generate(
    input_ids,
    max_new_tokens=4096,
    eos_token_id=terminators,
    do_sample=True,
    streamer = text_streamer,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty = 1.1
)

```
</details>
