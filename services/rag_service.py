from typing import List, Dict, Any, Optional
import os
import logging
from langdetect import detect, LangDetectException

from services.chat_service import ChatService
from llm_utils import DirectLLM
from embedding_utils import EmbeddingModel
from vector_store import VectorStore
from data_loader import DataLoader
from config import Config

# Set up logging
logger = logging.getLogger(__name__)


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

        # Get the model filename - handle special cases like "sug" and "pep" that are personas
        model_name = config.GGUF_MODEL

        # If the model name is a persona designation like "sug" or "pep", use the default model
        if model_name in ["sug", "pep"]:
            # Use the default model from the map or fall back to a common default
            model_map = config.get_gguf_model_map()
            model_name = model_map.get("default", "vicuna-7b-v1.5-q4_1.gguf")
            logger.info(f"Using default model {model_name} for persona {config.GGUF_MODEL}")

        # Get the full path to the model file (use cache_dir and not just filename)
        if os.path.exists(model_name):  # If it's already a full path
            model_path = model_name
        else:
            # Ensure model path is properly constructed
            model_path = os.path.join(config.GGUF_CACHE_DIR, model_name)

        logger.info(f"Initializing LLM with model path: {model_path}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            logger.error(f"GGUF model file not found at: {model_path}")
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}. Please check your configuration.")

        # Initialize LLM
        try:
            self.llm = DirectLLM(
                model_path=model_path,
                n_ctx=1024,
                n_threads=32
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
            raise RuntimeError(f"LLM initialization failed: {str(e)}")

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
                        logger.info(f"Loaded vector store for {store_key}")
                    except Exception as e:
                        logger.error(f"Failed to load vector store for {store_key}: {str(e)}")
                        # Create new store
                        self.vector_stores[store_key] = self._create_vector_store(collection, lang, vs_path)
                else:
                    # Create new store
                    self.vector_stores[store_key] = self._create_vector_store(collection, lang, vs_path)

    def _create_vector_store(self, collection: str, lang: str, save_path: str) -> VectorStore:
        """Create and populate a vector store for a collection and language"""
        json_path_attr = f"{collection.upper()}_JSON_PATH_{lang.upper()}"

        if not hasattr(self.config, json_path_attr):
            logger.warning(f"No JSON path configured for {collection}_{lang}")
            return VectorStore()  # Return empty store

        json_path = getattr(self.config, json_path_attr)
        if not json_path or not os.path.exists(json_path):
            logger.warning(f"JSON file not found at {json_path}")
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
        is_turkish = _is_turkish(query)
        lang = "tr" if is_turkish else "en"

        # Get the current persona from config
        persona = self.config.GGUF_MODEL if self.config.GGUF_MODEL in ["sug", "pep"] else None

        # Determine which collection to query based on keywords or use the current persona
        query_lower = query.lower()
        if "ekincan" in query_lower:
            collection = "ekincan"
        elif "pep" in query_lower or persona == "pep":
            collection = "pep"
        elif "sug" in query_lower or persona == "sug":
            collection = "sug"
        else:
            # Free-form conversation using the LLM directly
            # Get the appropriate cat name based on the current configuration
            cat_name = self.config.CAT_NAME_TR if is_turkish else self.config.CAT_NAME_EN

            # If no cat name is set, use a default
            if not cat_name:
                cat_name = "Şeker" if is_turkish else "Sugar"

            # Format the prompt template with the cat name and query
            try:
                prompt_template = self.config.FREE_PROMPT_TEMPLATE
                if not prompt_template:
                    # Fallback template if none is provided in config
                    prompt_template = "User: {query}\n{cat_name}:"

                prompt = prompt_template.format(cat_name=cat_name, query=query)

                result = self.llm.generate(prompt)

                # Clean up result
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
                if "User:" in result:
                    result = result.split("User:")[0].strip()

                return result
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                return f"Sorry, I encountered an error: {str(e)}"

        # For RAG queries using a specific collection
        try:
            # Get the vector store for the selected collection
            store_key = f"{collection}_{lang}"

            # Check if we have a vector store for this collection and language
            if store_key not in self.vector_stores:
                # Fall back to direct LLM response if no vector store available
                return self.generate_direct_response(query, is_turkish)

            vector_store = self.vector_stores[store_key]

            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)

            # Search for relevant documents
            results = vector_store.search(query_embedding, k=3)

            if not results:
                # Fall back to direct LLM response if no relevant documents found
                return self.generate_direct_response(query, is_turkish)

            # Build context from retrieved documents
            context = "\n\n".join([doc["content"] for doc in results])

            # Build prompt with retrieved context
            cat_name = self.config.CAT_NAME_TR if is_turkish else self.config.CAT_NAME_EN

            # Format a RAG prompt that includes both the retrieved context and the query
            prompt = f"Context information:\n{context}\n\nUser: {query}\n{cat_name}:"

            # Generate response
            result = self.llm.generate(prompt)

            # Clean up result
            if "User:" in result:
                result = result.split("User:")[0].strip()

            return result

        except Exception as e:
            logger.error(f"Error in RAG query processing: {str(e)}", exc_info=True)
            # Fall back to direct response
            return self.generate_direct_response(query, is_turkish)

    def generate_direct_response(self, query: str, is_turkish: bool) -> str:
        """Generate a direct response without RAG when needed as fallback"""
        try:
            cat_name = self.config.CAT_NAME_TR if is_turkish else self.config.CAT_NAME_EN
            if not cat_name:
                cat_name = "Şeker" if is_turkish else "Sugar"

            prompt_template = self.config.FREE_PROMPT_TEMPLATE
            if not prompt_template:
                prompt_template = "User: {query}\n{cat_name}:"

            prompt = prompt_template.format(cat_name=cat_name, query=query)

            result = self.llm.generate(prompt)

            # Clean up result
            if result.startswith(prompt):
                result = result[len(prompt):].strip()
            if "User:" in result:
                result = result.split("User:")[0].strip()

            return result
        except Exception as e:
            logger.error(f"Error in fallback response generation: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"