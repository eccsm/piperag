import logging
from typing import Type

from config import Config
from services.chat_service import ChatService

logger = logging.getLogger(__name__)


class ServiceFactory:
    _services = {}

    @classmethod
    def register_service(cls, model_type: str, service_class: Type[ChatService]):
        """
        Register a new service implementation
        Args:
            model_type: The model type key to register
            service_class: The service class to associate with the model type
        """
        logger.info(f"Registering service for model type: {model_type}")
        cls._services[model_type] = service_class

    @classmethod
    def create_service(cls, config: Config) -> ChatService:
        """
        Create service based on model_type in config
        Args:
            config: Configuration object containing MODEL_TYPE
        Returns:
            Instantiated chat service
        Raises:
            ValueError: If model type is not supported
        """
        # Import concrete service implementations here
        # This avoids circular imports
        from services.rag_service import RAGService

        # Register default services if not already registered
        if "gguf" not in cls._services:
            cls._services["gguf"] = RAGService
            logger.info(f"Registered GGUF service: {RAGService.__name__}")

        # Try to register MLC service if not already registered
        if "mlc" not in cls._services:
            try:
                from services.direct_chat_service import DirectChatService

                # Verify the DirectChatService properly implements the interface
                # by checking if it has the required method
                if not hasattr(DirectChatService, "generate_response"):
                    raise TypeError("DirectChatService doesn't implement generate_response")

                cls._services["mlc"] = DirectChatService
                logger.info(f"Registered MLC service: {DirectChatService.__name__}")
            except (ImportError, TypeError) as e:
                logger.warning(f"MLC service registration failed: {str(e)}. MLC service will not be available.")

        model_type = config.MODEL_TYPE.lower()
        logger.info(f"Creating service for model type: {model_type}")

        # Handle model switching and fallbacks
        if model_type == "mlc" and "mlc" not in cls._services:
            logger.warning("MLC model requested but not available. Falling back to GGUF.")
            model_type = "gguf"
            config.MODEL_TYPE = "gguf"

        # Get the appropriate service class
        service_class = cls._services.get(model_type)

        if not service_class:
            supported_types = list(cls._services.keys())
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {supported_types}"
            )

        try:
            # Create service instance
            if model_type == "gguf":
                model_path = config.get_gguf_model_path()
                logger.info(f"Creating {service_class.__name__} with GGUF model: {model_path}")
            elif model_type == "mlc":
                model_path = config.MLC_MODEL
                logger.info(f"Creating {service_class.__name__} with MLC model: {model_path}")
            else:
                model_path = "unknown"
                logger.warning(f"Unknown model type: {model_type}, path resolution may fail")

            # Initialize the service with the provided config
            service_instance = service_class(config)
            logger.info(f"Successfully created {service_class.__name__} for {model_type}")
            return service_instance
        except Exception as e:
            logger.error(f"Failed to instantiate {model_type} service: {str(e)}", exc_info=True)

            # If service creation fails and it was MLC, fall back to GGUF
            if model_type == "mlc" and "gguf" in cls._services:
                logger.warning("Falling back to GGUF model after MLC service initialization failed")
                config.MODEL_TYPE = "gguf"
                return cls.create_service(config)  # Recursive call with updated config

            # Re-raise the exception for other failures
            raise
