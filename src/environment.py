
#import model for final generation locally via ollama
from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen3-vl:8b")

#import embeddings model
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#import ChromaDB as a vectorDB platform with embeddings model previously specified
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

#import PyTorch cross-encoder for reranking retrieved chunks before passing to LLM
from reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()