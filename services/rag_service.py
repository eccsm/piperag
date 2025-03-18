from typing import List, Dict, Any, Optional
import os
from langdetect import detect, LangDetectException

from services.chat_service import ChatService
from llm_utils import DirectLLM
from embedding_utils import EmbeddingModel
from vector_store import VectorStore
from data_loader import DataLoader
from config import Config


def _is_turkish(text: str) -> bool:
    try:
        return detect(text) == "tr"
    except LangDetectException:
        return False


class RAGService(ChatService):
    def __init__(self, config: Config):
        super().__init__(config)

        # Initialize embedding model - shared across all vector stores
        self.embedding_model = EmbeddingModel(model_name=config.EMBEDDING_MODEL)

        # Initialize LLM
        model_path = os.path.join("llm", "gguf", config.GGUF_MODEL)
        self.llm = DirectLLM(
            model_path=model_path,
            n_ctx=1024,
            n_threads=32
        )

        # Initialize vector stores
        self.vector_stores = {}
        self._initialize_vector_stores()

    def _initialize_vector_stores(self):
        """Initialize vector stores for each data source"""
        vector_store_dir = "storage/vectors"
        os.makedirs(vector_store_dir, exist_ok=True)

        # Initialize vector stores for each language and collection
        collections = ["ekincan", "pep", "sug"]
        languages = ["tr", "en"]

        for collection in collections:
            for lang in languages:
                store_key = f"{collection}_{lang}"
                vs_path = os.path.join(vector_store_dir, store_key)

                # Try to load if exists, otherwise create new
                if os.path.exists(vs_path) and os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    try:
                        self.vector_stores[store_key] = VectorStore.load(vs_path)
                        print(f"Loaded vector store for {store_key}")
                    except Exception as e:
                        print(f"Failed to load vector store for {store_key}: {str(e)}")
                        # Create new store
                        self.vector_stores[store_key] = self._create_vector_store(collection, lang, vs_path)
                else:
                    # Create new store
                    self.vector_stores[store_key] = self._create_vector_store(collection, lang, vs_path)

    def _create_vector_store(self, collection: str, lang: str, save_path: str) -> VectorStore:
        """Create and populate a vector store for a collection and language"""
        json_path_attr = f"{collection.upper()}_JSON_PATH_{lang.upper()}"

        if not hasattr(self.config, json_path_attr):
            print(f"No JSON path configured for {collection}_{lang}")
            return VectorStore()  # Return empty store

        json_path = getattr(self.config, json_path_attr)
        if not json_path or not os.path.exists(json_path):
            print(f"JSON file not found at {json_path}")
            return VectorStore()  # Return empty store

        # Create vector store
        vector_store = VectorStore()

        # Load and process documents in batches
        loader = DataLoader(json_path)

        for docs_batch in loader.load_documents_stream(batch_size=20):
            texts = [doc.page_content for doc in docs_batch]
            metadata = [doc.metadata for doc in docs_batch]

            # Get embeddings
            embeddings = self.embedding_model.embed_documents(texts)

            # Create documents with content and metadata
            documents = [
                {"content": text, "metadata": metadata}
                for text, metadata in zip(texts, metadata)
            ]

            # Add to vector store
            vector_store.add_documents(documents, embeddings)

        # Save to disk
        os.makedirs(save_path, exist_ok=True)
        vector_store.save(save_path)

        return vector_store

    def generate_response(self, query: str, **kwargs) -> str:
        """Generate response using RAG approach"""
        # Detect language
        is_turkish = _is_turkish(query)  # Fixed: removed asterisks around is_turkish
        lang = "tr" if is_turkish else "en"

        # Determine which collection to query based on keywords
        query_lower = query.lower()
        if "ekincan" in query_lower:
            collection = "ekincan"
        elif "pep" in query_lower:
            collection = "pep"
        elif "sug" in query_lower:
            collection = "sug"
        else:
            # Free-form conversation using the LLM directly
            cat_name = self.config.CAT_NAME_TR if is_turkish else self.config.CAT_NAME_EN
            prompt = self.config.FREE_PROMPT_TEMPLATE.format(cat_name=cat_name, query=query)
            result = self.llm.generate(prompt)

            # Clean up result
            if result.startswith(prompt):
                result = result[len(prompt):].strip()
            if "User:" in result:
                # Fixed: incomplete conditional and missing return statement
                result = result.split("User:")[0].strip()

            return result  # Added the missing return statement