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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query, hits):
        pairs = [(query, doc.page_content) for doc, _ in hits]

        features = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)

        scored = sorted(zip(hits, scores.tolist()), key=lambda x: x[1], reverse=True)

        return [hit for hit, _ in scored]
