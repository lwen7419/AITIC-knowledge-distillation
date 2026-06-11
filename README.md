# AITIC knowledge distillation RAG pipeline
description

## Installation


## Usage


## Technologies

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

### Local Inference
All inference runs locally via Ollama, no external API calls. 
Chosen for data privacy and zero per-query cost.