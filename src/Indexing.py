import environment

#clear existing collection so recursive and semantic chunks don't mix
environment.vector_store.delete_collection()

from langchain_community.document_loaders import PyPDFLoader

#pdf reader, makes list of docs
loader = PyPDFLoader("/Users/lidasmac/Desktop/Foucault_Michel_Power_Knowledge_Selected_Interviews_and_Other_Writings_1972-1977.pdf")
docs = loader.load()

#check first page character count and give first 500 char preview of first page
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])

##Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

#use langchain text splitter with specified chunk size and overlap
#instantiate langchain textsplitter with my configs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
#call split_documents method from langchain on text_splitter also from langchain, passed docs
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

##Storing
#add all_splits into previously specified vectorDB
document_ids = environment.vector_store.add_documents(documents=all_splits)

print(document_ids[:3])