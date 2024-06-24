def result(rag_chain,question):
    response= rag_chain.invoke(question)
    print(f"[풋풋이의 답변]\n{response}")