import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlc-llm'))
from mlc_llm import MLCEngine

from services.chat_service import ChatService  # Your abstract chat service interface
from config import Config

logger = logging.getLogger("mlc_llm_integration")
logger.setLevel(logging.DEBUG)
stop_tokens = ["User:"]
class DirectChatService(ChatService):
    def __init__(self, config: Config):
        super().__init__(config)
        # Instantiate the MLC wrapper with the HF URL from config.
        self.llm = MLCEngine(self.config.MLC_MODEL)

    def generate_response(self, query: str, **kwargs) -> str:
        full_response = ""
        messages = [{"role": "user", "content": query}]
        try:
            for response in self.llm.chat.completions.create(messages=messages, stream=True):
                for choice in response.choices:
                    content = choice.delta.content.rstrip()
                    full_response += content
                    logger.debug(f"MLC response chunk: {content}")
            return full_response
        except Exception as e:
            logger.error("Error in MLC _call", exc_info=True)
            raise e
