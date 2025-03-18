import os
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize a lightweight embedding model

        Args:
            model_name: HuggingFace model ID or path to local model
        """
        # Check if model exists locally first
        local_path = os.path.join("models", "embeddings", model_name)
        if os.path.exists(local_path):
            self.model = SentenceTransformer(local_path)
        else:
            self.model = SentenceTransformer(model_name)
            # Save model for future use
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.model.save(local_path)

        # Set to CPU explicitly
        self.model.to("cpu")

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text"""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a batch of documents"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True
        ).tolist()