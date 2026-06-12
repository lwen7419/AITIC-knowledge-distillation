#takes llm's answer and retrieved chunks as argument
def format_answer_with_citations(answer, retrieved_docs):
    #empty list to collect citation strings
    sources = []
    #loop over chunks; for each chunk...
    for doc in retrieved_docs:
        #pulls page number and source filename from chunk metadata
        page = doc.metadata.get("page")
        
        source = doc.metadata.get("source")
        sources.append(f"[Page {page}, {source}]")
    citations = " | ".join(sources)
    return f"{answer}\n\nSources: {citations}"