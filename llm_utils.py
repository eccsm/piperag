import gc
from typing import List, Generator
from llama_cpp import Llama

class DirectLLM:
    def __init__(self, model_path: str, n_ctx: int = 1024, n_threads: int = 32):
        """
        Initialize a direct LLM interface using llama.cpp

        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (smaller windows use less RAM)
            n_threads: Number of CPU threads (optimize for Ryzen 9 5950x)
        """
        self.model_path = model_path

        gc.collect()

        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=512,
                n_threads=n_threads,
                use_mmap=True,
                use_mlock=False,
                f16_kv=True
            )
            print("Llama model initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Llama model: {e}")
            self.llm = None

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7, stop: List[str] = None) -> str:
        """Generate text without streaming"""
        if self.llm is None:
            raise RuntimeError("Llama model is not initialized.")

        stop = stop or ["User:"]

        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        return output["choices"][0]["text"]

    def stream(self, prompt: str, max_tokens: int = 256,
               temperature: float = 0.7, stop: List[str] = None) -> Generator[str, None, None]:
        """Stream tokens one by one to avoid buffering entire response"""
        if self.llm is None:
            raise RuntimeError("Llama model is not initialized.")

        stop = stop or ["User:"]

        for token in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True
        ):
            yield token["choices"][0]["text"]

    def __del__(self):
        # Force cleanup
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            gc.collect()
        else:
            print("Llama model was not initialized or already deleted.")
