import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_token():
    model_id = 'Dongwookss/small_fut_final'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return model, tokenizer