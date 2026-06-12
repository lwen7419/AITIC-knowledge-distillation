import environment

#clear existing collection so recursive and semantic chunks don't mix
environment.vector_store.delete_collection()

from langchain_community.document_loaders import PyPDFLoader

#pdf reader, makes list of docs
doc_name = "data/cs111_coursepack.pdf"
loader = PyPDFLoader(doc_name)
docs = loader.load()

#check first page character count and give first 500 char preview of first page
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])

##Splitting
from heading_based_semantic_chunker import semantic_chunk
from langchain_core.documents import Document

all_splits = []
for doc in docs:
    chunks = semantic_chunk(doc.page_content, max_chunk_size = 300)
    for chunk in chunks:
        all_splits.append(Document(
            page_content=chunk.pop("text"),
            metadata={**doc.metadata, **chunk}
        ))

print(f"Split {doc_name} into {len(all_splits)} sub-documents.")

##Storing
#add all_splits into previously specified vectorDB
document_ids = environment.vector_store.add_documents(documents=all_splits)

print(document_ids[:3])