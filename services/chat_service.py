from abc import ABC, abstractmethod

from config import Config


class ChatService(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response to the given prompt"""
        pass
