#takes llm's answer and retrieved chunks as argument
def format_answer_with_citations(answer, retrieved_docs):
    #empty list to collect citation strings
    sources = []
    #loop over chunks; for each chunk...
    for doc in retrieved_docs:
        #pulls page number from chunk metadata
        page = doc.metadata.get("page")
        #pulls source from chunk metadata
        source = doc.metadata.get("source")
        #appends page, source to sources list
        sources.append(f"[Page {page}, {source}]")
    #create string of sources separated by |
    citations = " | ".join(sources)
    #make whole llm response joined by citations
    return f"{answer}\n\nSources: {citations}"