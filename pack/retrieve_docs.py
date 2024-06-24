def print_search_results(retrievers, query):
    print(f"Query: {query}")
    for i in range(len(retrievers.invoke(query))):
        print(f'찾은 문장{i+1}:',
            retrievers.invoke(query)[i].page_content)