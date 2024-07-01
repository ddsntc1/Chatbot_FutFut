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

### How to use? 
- Fine-tuned Model : Llama3-8b, Zephyr-7b

<details>
  <summary>Scripts Description</summary>
    -  load_model_type_a.py : transformers의 AutoModelForCausalLM을 이용하여 모델을 불러옵니다.
    -  load_model_type_b.py : Unsloth 패키지의 FastLanguageModel을 이용하여 모델을 불러옵니다. 답변 생성속도가 빠르지만 튜닝을 위한 패키지이기 때문에 Huggingface에 adapter_config가 존재하면 모델을 불러오지 못합니다.
</details>

  



#### About Fine-tuning

- Method : By Unsloth
- Trainer : SFTrainer, ~~DPOTrainer~~

#### About RAG

- Method

#### Fine-tuned Result:
- [Dongwooks](https://huggingface.co/Dongwookss)
- [mintaeng](https://huggingface.co/mintaeng)


