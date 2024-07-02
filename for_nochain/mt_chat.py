from transformers import TextStreamer

def search_results(retrievers, query):
    rt_qr = []
    for i in range(len(retrievers.invoke(query))):
        rt_qr.append(retrievers.invoke(query)[i].page_content)
    rt_str = ''
    for i in range(len(rt_qr)):
        rt_str += str(rt_qr[i]) + '\n'

    return rt_str



def chat(model,tokenizer,retriever,instruction):
    PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.
    제시하는 context에서만 대답하고 context에 없는 내용은 모르겠다고 대답해. 모르는 답변을 임의로 생성하지마.
    '''

    Context = search_results(retriever,instruction)

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {'role': 'assistant','content':f"{Context}"},
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
        temperature=0.7,
        top_p=0.9,
        repetition_penalty = 1.1
    )
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)