import gc
import os
import logging
from typing import List, Generator
from llama_cpp import Llama

# Set up logging
logger = logging.getLogger(__name__)


class DirectLLM:
    def __init__(self, model_path: str, n_ctx: int = 768, n_threads: int = 12):
        """
        Initialize a direct LLM interface using llama.cpp

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (smaller windows use less RAM)
            n_threads: Number of CPU threads (optimize for your CPU)
        """
        self.model_path = model_path
        self.llm = None

        # Verify the model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")

        # Clean up memory before loading new model
        gc.collect()

        try:
            logger.info(f"Loading Llama model from: {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=256,
                n_threads=n_threads,
                use_mmap=True,
                use_mlock=False,
                f16_kv=True
            )
            logger.info("Llama model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Llama model: {str(e)}")

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, stop: List[str] = None) -> str:
        """Generate text without streaming"""
        if self.llm is None:
            logger.error("Cannot generate: Llama model is not initialized.")
            raise RuntimeError("Llama model is not initialized.")

        stop = stop or ["User:"]

        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}", exc_info=True)
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def stream(self, prompt: str, max_tokens: int = 256,
               temperature: float = 0.7, stop: List[str] = None) -> Generator[str, None, None]:
        """Stream tokens one by one to avoid buffering entire response"""
        if self.llm is None:
            logger.error("Cannot stream: Llama model is not initialized.")
            raise RuntimeError("Llama model is not initialized.")

        stop = stop or ["User:"]

        try:
            for token in self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    stream=True
            ):
                yield token["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error during text streaming: {str(e)}", exc_info=True)
            raise RuntimeError(f"Text streaming failed: {str(e)}")

    def __del__(self):
        # Force cleanup
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                del self.llm
                gc.collect()
                logger.debug("Llama model cleaned up successfully.")
            except Exception as e:
                logger.error(f"Error cleaning up Llama model: {str(e)}")
        else:
            logger.debug("Llama model was not initialized or already deleted.")