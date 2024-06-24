def result(rag_chain):
    response= rag_chain.invoke(input('질문을 입력하세요: '))
    print(f"[풋풋이의 답변]\n{response}")