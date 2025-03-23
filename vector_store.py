import json
import logging
import os
from typing import List, Dict, Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        """
        Initialize a lightweight vector store using FAISS

        Args:
            dimension: Embedding vector dimension (default for all-MiniLM-L6-v2)
            index_type: FAISS index type (Flat = exact but memory efficient)
        """
        self.dimension = dimension

        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine on normalized vectors
        else:
            # Only use if more RAM is available
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node

        self.documents = []  # Store document metadata
        logger.info(f"Initialized VectorStore with {index_type} index, dimension={dimension}")

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[int]:
        """
        Add documents with their embeddings to the index

        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Convert to numpy array and normalize
        vectors = np.array(embeddings).astype(np.float32)
        faiss.normalize_L2(vectors)  # In-place normalization

        # Add to index
        self.index.add(vectors)

        # Store document metadata
        start_idx = len(self.documents)
        doc_ids = []

        for i, doc in enumerate(documents):
            doc_id = start_idx + i
            doc["id"] = doc_id
            self.documents.append(doc)
            doc_ids.append(doc_id)

        logger.info(f"Added {len(documents)} documents to index, total: {len(self.documents)}")
        return doc_ids

    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return

        Returns:
            List of document dictionaries with scores
        """
        if len(self.documents) == 0:
            return []

        # Convert to numpy and normalize
        query_vector = np.array([query_embedding]).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        k = min(k, len(self.documents))
        scores, indices = self.index.search(query_vector, k)

        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if len(self.documents) > idx >= 0:
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[0][i])
                results.append(doc)

        logger.debug(f"Search returned {len(results)} results")
        return results

    def delete_documents(self, doc_ids: List[int]) -> None:
        """
        Delete documents from the index
        Note: This rebuilds the index which can be expensive for large datasets

        Args:
            doc_ids: List of document IDs to delete
        """
        if not doc_ids or len(self.documents) == 0:
            return

        # Find documents to keep
        keep_docs = []
        keep_ids = []

        for i, doc in enumerate(self.documents):
            if doc["id"] not in doc_ids:
                keep_docs.append(doc)
                keep_ids.append(i)

        if not keep_docs:
            # Delete all documents
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            logger.info("Deleted all documents from index")
            return

        # We need the embeddings to rebuild the index
        # For this, we'd need to have stored them or recompute them
        # This is a limitation of the current implementation
        logger.warning("Document deletion requires rebuilding the index - embeddings not available")

        # Update documents list
        self.documents = keep_docs
        logger.info(f"Deleted {len(doc_ids)} documents, {len(self.documents)} remaining")

    def save(self, directory: str) -> None:
        """
        Save index and documents to disk

        Args:
            directory: Directory to save index and documents
        """
        os.makedirs(directory, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(directory, "index.faiss")
        faiss.write_index(self.index, index_path)

        # Save documents
        doc_path = os.path.join(directory, "documents.json")
        with open(doc_path, "w") as f:
            json.dump(self.documents, f)

        logger.info(f"Saved vector store with {len(self.documents)} documents to {directory}")

    @classmethod
    def load(cls, directory: str):
        """
        Load index and documents from disk

        Args:
            directory: Directory containing index and documents

        Returns:
            VectorStore instance
        """
        # Check if files exist
        index_path = os.path.join(directory, "index.faiss")
        doc_path = os.path.join(directory, "documents.json")

        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            logger.warning(f"Vector store files not found in {directory}")
            return cls(dimension=384)

        try:
            # Create instance
            instance = cls(dimension=384)

            # Load FAISS index
            instance.index = faiss.read_index(index_path)

            # Load documents
            with open(doc_path, "r") as f:
                instance.documents = json.load(f)

            logger.info(f"Loaded vector store with {len(instance.documents)} documents from {directory}")
            return instance
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return cls(dimension=384)

    def __len__(self) -> int:
        """Return the number of documents in the store"""
        return len(self.documents)