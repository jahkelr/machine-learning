import faiss
from transformers import AutoTokenizer, AutoModel
import torch


class MemoryManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Load the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.memory = []
        self.index = faiss.IndexFlatL2(384)  # Assuming embeddings of size 384

    def _encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def store_memory(self, text):
        embeddings = self._encode(text)
        self.memory.append(text)
        self.index.add(embeddings)

    def retrieve_memory(self, query, top_k=3):
        query_embedding = self._encode(query)
        _, indices = self.index.search(query_embedding, top_k)
        return [self.memory[i] for i in indices[0]]
