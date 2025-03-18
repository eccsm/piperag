import logging
from typing import Union
from config import Config

logger = logging.getLogger("service_factory")


class ServiceFactory:
    @staticmethod
    def create_service(config: Config):
        """
        Factory method to create the appropriate chat service based on config.

        Args:
            config: Application configuration

        Returns:
            An instance of a ChatService implementation
        """
        if config.MODEL_TYPE.lower() == "mlc":
            from services.direct_chat_service import DirectChatService
            logger.info(f"Creating MLCEngine with model: {config.MLC_MODEL}")
            return DirectChatService(config)
        elif config.MODEL_TYPE.lower() == "gguf":
            from services.rag_service import RAGService
            logger.info(f"Creating LlamaCpp service with model: {config.GGUF_CACHE_DIR}")
            return RAGService(config)
        else:
            raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")