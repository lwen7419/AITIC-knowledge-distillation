# AITIC knowledge distillation RAG pipeline
Local-first RAG pipeline that distills textbooks into verifiable, citation-grounded AI agents: no cloud, no API keys, no data leaving the machine.

## Installation
### Prerequisites
- macOS (Apple Silicon) or Linux — Windows untested
- [Ollama](https://ollama.com) installed and running
- Python 3.10+ (Anaconda recommended)
### Setup
 
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AITIC-knowledge-distillation.git
cd AITIC-knowledge-distillation
 
# 2. Install Python dependencies
pip install langchain langchain-ollama langchain-chroma langchain-community pypdf
 
# 3. Pull the required models
ollama pull qwen3-vl:8b
ollama pull nomic-embed-text
 
# 4. Verify Ollama is running
ollama list
```

## Usage
```bash
# 1. Place source PDFs in data/
 
# 2. Index documents into the vector store
python src/indexing_optimized.py
 
# 3. Query the RAG pipeline
python src/RAG_chain.py
 
# 4. Run evaluation across all metrics
python eval/run_eval.py
# Results are saved per ablation condition in eval/results/
```

## Technologies
| Component | Choice | Role |
|-----------|--------|------|
| LLM | Qwen3-vl-8b (via Ollama) | Answer generation, multimodal parsing |
| Embeddings | nomic-embed-text (via Ollama) | Text-to-vector conversion for similarity search |
| Vector store | ChromaDB | Local persistent storage of document embeddings |
| Orchestration | LangChain | Pipeline coordination, middleware, document handling |
| PDF parsing | PyPDFLoader | Document loading with page-level metadata |
| Chunking | Custom heading-based semantic chunker | Structure-aware splitting for academic text |
| Inference | Ollama (localhost) | Local model serving, no external API calls |
## Architecture & Design Decisions

### Chunking Strategy
Used custom heading-based semantic chunker instead of LangChain's 
RecursiveCharacterTextSplitter. Academic PDFs have clear heading 
hierarchy, so splitting on structure preserves semantic coherence 
per chunk and improves retrieval quality.
Not refactoring the chunker with Document to make optimized indexing 
same as normal indexing to save time on editing semantic chunker, 
might change later. 

### RAG Pattern: Chain vs Agent
Chose chain approach with @dynamic_prompt middleware over agent+tool. 
Single LLM call per query reduces latency on consumer hardware. 
Agent approach is a planned upgrade for multi-step queries.
Will also make citations inline instead for clarity
and learning purposes later during agent approach upgrade. 

### Local Inference
All inference runs locally via Ollama, no external API calls. 
Chosen for data privacy and zero per-query cost.
