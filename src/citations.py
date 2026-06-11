def format_answer_with_citations(answer, retrieved_docs):
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        sources.append(f"[Page {page}, {source}]")
    citations = " | ".join(sources)
    return f"{answer}\n\nSources: {citations}"