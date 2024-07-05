# Chatbot_FutFut

### Team 

<a href = "https://dongwooks.notion.site/LLM-2-0-e67ead8ba79a4acd9b4d1b815e3dfa94?pvs=4"><img src="https://img.shields.io/badge/팀 Notion-ffffff?style=flat&logo=Notion&logoColor=black" /></a>

#### Members
[😎강동욱](https://github.com/ddsntc1)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [🦄강민지](https://github.com/lucide99)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [😺신대근](https://github.com/bigroot0504)

#### 🚀 Use Tech
![Ubuntu](https://img.shields.io/badge/ubuntu-orange?style=for-the-badge&logo=ubuntu)
![Slack](https://img.shields.io/badge/slack-blue?style=for-the-badge&logo=slack)
![HuggingFace](https://img.shields.io/badge/huggingface-yellow?style=for-the-badge&logo=HuggingFace)
![Colab](https://img.shields.io/badge/Colab-black?style=for-the-badge&logo=GoogleColab)
![Colab](https://img.shields.io/badge/FastAPI-cyanblue?style=for-the-badge&logo=FastAPI)

---

### About FutFut
<img src="https://github.com/ddsntc1/Chatbot_FutFut/assets/38596856/cb1cd8b7-c556-46a8-ab8d-e093af713433.jpg" width="400" height="400">

- **Domain** : 풋살 플랫폼에 친절한 설명을 해주는 챗봇을 구축하였습니다.
- **Concept** : '해요'체를 사용하며 친절하게 답하는 챗봇. 말끝에 '언제든지 물어보세요! 풋풋~!'을 붙여 풋풋이 컨셉을 유지 
- **Model** : Mistral 기반의 [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) 모델과 Meta의 [Llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 모델을 대상으로 진행하였습니다. 
- **Dataset** : 말투 학습을 위한 데이터셋을 구축하여 진행하였습니다. [Dongwookss/q_a_korean_futsal](https://huggingface.co/datasets/Dongwookss/q_a_korean_futsal), [mintaeng/llm_futsaldata_yo](https://huggingface.co/datasets/mintaeng/llm_futsaldata_yo)
- **How-to**? 말투 학습을 위한 Fine-tuning과 정보 제공을 위한 RAG를 적용시켰습니다. 구현은 **FastAPI**를 이용하여 Back-end와 소통할 수 있도록 진행하였습니다. 
---
### How to use? 

<details>
  <summary>FastAPI 실행</summary>

   ```python
  uvicorn main:app --reload -p <포트번호지정>
  ```

</details>

---

#### About Fine-tuning
- **Fine-tuned Model** : Llama3-8b, Zephyr-7b 각각 튜닝을 진행하였습니다.
- **GPU** : Colab L4 
- **Method** : LoRA(Low Rank Adaptation) & QLoRA(Quantized LoRA)
- **Trainer** : SFTrainer, ~~DPOTrainer~~
- **Dataset** : [Dongwookss/q_a_korean_futsal](https://huggingface.co/datasets/Dongwookss/q_a_korean_futsal), [mintaeng/llm_futsaldata_yo](https://huggingface.co/datasets/mintaeng/llm_futsaldata_yo)

- **Finetune** : <img src="https://github.com/ddsntc1/Chatbot_FutFut/assets/38596856/6bd84b2b-5ba2-4205-8203-3ec539d33899.jpg" width="400" height="200">

```python
TrainOutput(global_step=1761, training_loss=1.1261051157399513, metrics={'train_runtime': 26645.6613, 'train_samples_per_second': 2.644, 'train_steps_per_second': 0.066, 'total_flos': 7.784199669311078e+17, 'train_loss': 1.1261051157399513, 'epoch': 3.0})
```

- **추후 방향** : SFT(Supervised Fine-Tune) Trainer 을 이용하여 튜닝을 진행하였고 말투에 집중한 데이터셋으로 인해 모델 성능에 아쉬운 점이 많았습니다. 향후 Q-A Task에 맞는 Fine-Tuning을 진행할 예정이며 강화학습을 통해 모델성능을 개선할 예정입니다.

#### Fine-tuned Result(HuggingFace🤗): 

- [Dongwooks](https://huggingface.co/Dongwookss) -> 최종 모델명 : big_fut_final & small_fut_final

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



---


#### About RAG

- 풋살 규정, 구장 정보, 풋살 칼럼 등 다양한 정보를 제공하기 위해 데이터를 수집하고 RAG를 구축하여 정보제공을 하였습니다.

- **Retrieval** : **Kiwipiepy+BM25** 와 **Embedding_Model + VectorDB** 조합을 통해 Semantic search를 목표로 진행하였습니다.


### Directory Structure
```Linux
.
├── backupfiles
│   └── # 예비 파일 경로입니다.
├── files
│   └── # RAG를 통해 전달할 파일 경로입니다.

├── for_nochain
│   ├── __init__.py
│   └── mt_chat.py #  Langchain 을 이용하지 않고 구성하였습니다. 모델 답변 속도가 저하될 수 있습니다.

├── load_model_for_newchain.py
├── load_model_type_a.py # AutoModelForCausalLM을 이용하여 모델을 불러옵니다.
├── load_model_type_b.py # Unsloth 패키지의 FastLanguageModel을 이용하여 모델을 불러옵니다. 이때 adapter.config가 존재하면 불러오지 못하여 새로운 경로에 모델을 복사하였습니다.

├── main.py # Fast API 를 이용하여 모델을 서빙합니다. request를 통해 모델과 소통할 수 있습니다.
├── main_new_chain.py # 새 체인을 이용하여 FastAPI를 실행합니다.

├── pack
│   ├── __init__.py
│   ├── load_push.py # files에 있는 데이터를 Load,Chunk,Embed, Vector DB에 저장합니다.
│   ├── make_answer.py # 답변생성 함수를 만들었습니다.
│   ├── make_chain_gguf.py # gguf 파일을 대상으로 ollama 를 적용시킵니다.
│   ├── make_chain_model.py # Safetensors로 이루어진 모델로 Chain을 생성합니다. 이때 GPU자원이 많이 요구됩니다.
│   ├── retrieve_docs.py # Retrieval을 이용하여 원하는 데이터를 찾습니다.
│   └── retriever.py # Retrieval을 설정합니다.

├── sft_tuning # 모델 파인튜닝 과정입니다. 중요 파라미터에 대한 값이 비어있을 수 있습니다.
│   └── Unsloth_sft.ipynb
└── test.ipynb
```

---

#### 🤗 HuggingFace account 🤗

- [Dongwooks](https://huggingface.co/Dongwookss) 
- [mintaeng](https://huggingface.co/mintaeng)

- [bigr00t](https://huggingface.co/bigr00t)
