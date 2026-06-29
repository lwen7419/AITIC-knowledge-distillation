import torch
#load two classes from HuggingFace autotokenizer converts text into numbers
#model can process. AutoModelForSequenceClassification is a model that takes 
#pair of texts and outputs single score of their relevancy to each other. 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoderReranker:
    #default model is cross-encoder trained on passage retrieval
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        #loads tokenizer for model from HuggingFace, loaded into instance attribute 
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #downloads and loads model weights from HuggingFace
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        #switch model to evaluation mode
        self.model.eval()
    #take search query and list of (Document,score) tuples returned by ChromaDB
    def rerank(self, query, hits):
        #builds list of (query, chunk_text) pairs, throw away previous embedding 
        # scores
        pairs = [(query, doc.page_content) for doc, _ in hits]
        #tokenize all pairs in single batch 
        features = self.tokenizer(
            #first list is all queries, second list is chunk texts
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            #make all inputs same length by adding padding tokens to shorter ones
            padding=True,
            #cut off anything larger than max_length, which is max tokens model 
            #can handle at once
            truncation=True,
            max_length=512,
            #return PyTorch tensors instead of Python lists
            return_tensors="pt"
        )

        #tell PyTorch to not track gradients during rerank. Gradients needed
        #for update weights in training, skipping makes inference faster and 
        #uses less memory
        with torch.no_grad():
            #runs huggingface model, getting raw output scores and removing 
            # last dimension
            scores = self.model(**features).logits.squeeze(-1)

        #zip original hit with cross-encoder score to sort highest score first
        # and converts Pytorch tensor into Python list so zip can work with
        scored = sorted(zip(hits, scores.tolist()), key=lambda x: x[1], reverse=True)

        #return just hits in new reranked order, discarding cross-encoder scores,
        # keeping original (Document, chromadb_score) format
        return [hit for hit, _ in scored]
