def print_search_results(retrievers, query):
    print(f"Query: {query}")
    for i in range(len(retrievers.invoke(query))):
        print(f'찾은 문장{i+1}:',
            retrievers.invoke(query)[i].page_content)
        


def search_results(retrievers, query):
    rt_qr = []
    for i in range(len(retrievers.invoke(query))):
        rt_qr.append(retrievers.invoke(query)[i].page_content)
    rt_str = ''
    for i in range(len(rt_qr)):
        rt_str += str(rt_qr[i]) + '\n'

    return rt_str