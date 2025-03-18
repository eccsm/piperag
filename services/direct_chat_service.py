import logging
import sys
import os
import re
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Union

from posthog import flush
from huggingface_hub import snapshot_download, hf_hub_download

# Add the mlc-llm directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlc-llm'))
from mlc_llm import MLCEngine

from services.chat_service import ChatService  # Your abstract chat service interface
from config import Config

logger = logging.getLogger("mlc_llm_integration")
logger.setLevel(logging.DEBUG)
stop_tokens = ["User:"]


def _process_model_path(model_path: str) -> str:
    """
    Process the model path, downloading from HuggingFace if necessary.

    Supports the following formats:
    - Local path: '/app/models/model_name'
    - HF URL format: 'HF://username/repo_name'

    Returns the local path to the model.
    """
    # Check if it's a HuggingFace URL
    hf_pattern = r"^HF://([^/]+)/([^/]+)$"
    match = re.match(hf_pattern, model_path)

    if not match:
        # It's a local path, return as is
        logger.info(f"Using local model at: {model_path}")
        return model_path

    # Extract username and repo name from the HF URL
    username, repo_name = match.groups()
    repo_id = f"{username}/{repo_name}"

    # Define the local directory where the model will be saved
    models_dir = os.environ.get("MLC_MODEL_DIR", "/app/models")
    local_model_dir = os.path.join(models_dir, repo_name)

    # Check if model already exists locally
    if os.path.exists(local_model_dir) and os.listdir(local_model_dir):
        logger.info(f"Model already downloaded at: {local_model_dir}")
        return local_model_dir

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Download the model from HuggingFace
    logger.info(f"Downloading model from HuggingFace: {repo_id} to {local_model_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False,  # Avoid symlinks for Docker compatibility
            resume_download=True
        )
        logger.info(f"Model downloaded successfully to: {local_model_dir}")
        return local_model_dir
    except Exception as e:
        logger.error(f"Error downloading model from HuggingFace: {e}", exc_info=True)
        raise RuntimeError(f"Failed to download model from {repo_id}: {str(e)}")


class DirectChatService(ChatService):
    def __init__(self, config: Config):
        super().__init__(config)
        # Process the model path - could be local or HF URL
        model_path = _process_model_path(config.MLC_MODEL)
        logger.info(f"Initializing MLCEngine with model path: {model_path}")
        self.llm = MLCEngine(model_path)

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
            logger.error("Error in MLC call", exc_info=True)
            raise e