import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlc-llm'))
from mlc_llm import MLCEngine

from services.chat_service import ChatService
from config import Config

logger = logging.getLogger("mlc_llm_integration")
logger.setLevel(logging.DEBUG)

# Efficient stop tokens
STOP_TOKENS = ["User:", "< EOT >", "Human:", "Assistant:"]

# Global model cache to avoid reloading
_MODEL_CACHE = {}
_MODEL_LOCK = threading.RLock()


class MLCModelManager:
    @staticmethod
    def get_model(model_path):
        with _MODEL_LOCK:
            if model_path not in _MODEL_CACHE:
                logger.info(f"Loading MLC model: {model_path}")
                _MODEL_CACHE[model_path] = MLCEngine(model_path)
                # Perform warmup in background
                threading.Thread(
                    target=MLCModelManager._warmup_model,
                    args=(_MODEL_CACHE[model_path],),
                    daemon=True
                ).start()
            return _MODEL_CACHE[model_path]

    @staticmethod
    def _warmup_model(engine):
        try:
            warmup_msg = [{"role": "user", "content": "Hello"}]
            engine.chat.completions.create(
                messages=warmup_msg,
                stream=False,
                max_tokens=20
            )
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup had an issue: {e}")


# Optimized batch accumulator
class TokenBatchAccumulator:
    def __init__(self, batch_size=10, batch_timeout=0.05):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.tokens = []
        self.last_flush_time = time.time()

    def add(self, token):
        self.tokens.append(token)
        current_time = time.time()
        if len(self.tokens) >= self.batch_size or (current_time - self.last_flush_time) >= self.batch_timeout:
            return self.flush()
        return None

    def flush(self):
        if not self.tokens:
            return None
        result = "".join(self.tokens)
        self.tokens = []
        self.last_flush_time = time.time()
        return result


class DirectChatService(ChatService):
    def __init__(self, config: Config):
        super().__init__(config)
        # Use the model manager to get a cached instance
        self.llm = MLCModelManager.get_model(self.config.MLC_MODEL)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate_response(self, query: str, **kwargs) -> str:
        full_response = ""
        messages = [{"role": "user", "content": query}]

        try:
            # Optimize generation parameters
            params = {
                "messages": messages,
                "stream": True,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.1),  # Lower temperature for faster generation
                "top_p": kwargs.get("top_p", 0.8),  # Add top_p for faster sampling
                "stop": STOP_TOKENS
            }

            # Create a batch accumulator with larger batch size
            accumulator = TokenBatchAccumulator(batch_size=15, batch_timeout=0.03)

            # Use executor for background processing of responses
            def process_chunk(response):
                nonlocal full_response
                batch_content = accumulator.add(response)
                if batch_content:
                    full_response += batch_content
                    return batch_content
                return None

            # Stream responses with optimized batching
            for response in self.llm.chat.completions.create(**params):
                for choice in response.choices:
                    content = choice.delta.content
                    if content:
                        # Process in background to reduce main thread blocking
                        future = self.executor.submit(process_chunk, content)
                        batch = future.result()
                        if batch:
                            logger.debug(f"MLC response batch: {batch}")

            # Flush any remaining tokens
            remaining = accumulator.flush()
            if remaining:
                full_response += remaining
                logger.debug(f"MLC final response batch: {remaining}")

            return full_response

        except Exception as e:
            logger.error("Error in MLC call", exc_info=True)
            raise e

    def generate_response_stream(self, query: str, **kwargs):
        """
        Generator function that yields tokens one by one for streaming responses.
        This is used by the streaming endpoint.
        """
        messages = [{"role": "user", "content": query}]

        try:
            # Optimize generation parameters for streaming
            params = {
                "messages": messages,
                "stream": True,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.8),
                "stop": STOP_TOKENS
            }

            # Small batch accumulator for smoother streaming
            accumulator = TokenBatchAccumulator(batch_size=5, batch_timeout=0.02)

            # Stream responses token by token with minimal batching
            for response in self.llm.chat.completions.create(**params):
                for choice in response.choices:
                    content = choice.delta.content
                    if content:
                        # Add to accumulator
                        batch = accumulator.add(content)
                        if batch:
                            yield batch

            # Flush any remaining tokens
            remaining = accumulator.flush()
            if remaining:
                yield remaining

        except Exception as e:
            logger.error("Error in MLC streaming call", exc_info=True)
            yield f"Error: {str(e)}"