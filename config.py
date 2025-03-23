import json
import os
from typing import Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _load_json_env(env_name: str, default: Any = None) -> Any:
    """Load JSON from environment variable"""
    env_value = os.getenv(env_name)
    if not env_value:
        return default
    try:
        return json.loads(env_value)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from {env_name}")
        return default


class Config:
    def __init__(self):
        # API Security
        self.API_KEY = os.getenv('API_KEY')

        # Cat Persona Names
        self.CAT_NAME_EN = os.getenv('CAT_NAME_EN')
        self.CAT_NAME_TR = os.getenv('CAT_NAME_TR')

        # Data Paths
        self.EKINCAN_JSON_PATH_TR = os.getenv('EKINCAN_JSON_PATH_TR')
        self.EKINCAN_JSON_PATH_EN = os.getenv('EKINCAN_JSON_PATH_EN')
        self.PEP_JSON_PATH_TR = os.getenv('PEP_JSON_PATH_TR')
        self.PEP_JSON_PATH_EN = os.getenv('PEP_JSON_PATH_EN')
        self.SUG_JSON_PATH_TR = os.getenv('SUG_JSON_PATH_TR')
        self.SUG_JSON_PATH_EN = os.getenv('SUG_JSON_PATH_EN')

        # Model Directories
        self.MODELS_BASE_DIR = os.getenv('MODELS_BASE_DIR', '/app/models')
        self.GGUF_CACHE_DIR = os.getenv('GGUF_CACHE_DIR', './llm/gguf')
        self.MLC_MODEL_DIR = os.getenv('MLC_MODEL_DIR', '/app/models/mlc')

        # Model Configuration
        self.MODEL_TYPE = os.getenv('MODEL_TYPE', 'gguf')

        # Model mappings
        self._mlc_model_map = _load_json_env('MLC_MODEL_MAP', {"default": "eccsm/mlc_llm"})
        self._gguf_model_map = _load_json_env('GGUF_MODEL_MAP', {"default": "gguf-vicuna-7b-v1.5-q4_1.gguf"})
        self.IMAGE_MODEL_MAP = _load_json_env('IMAGE_MODEL_MAP', {})

        # Default active models
        self._default_mlc_model = os.getenv('DEFAULT_MLC_MODEL', 'default')
        self._default_gguf_model = os.getenv('DEFAULT_GGUF_MODEL', 'default')

        # Legacy direct model paths (for backward compatibility)
        self._direct_mlc_model_path = os.getenv('MLC_MODEL')
        self._direct_gguf_model_path = os.getenv('GGUF_MODEL')

        # RAG Configuration
        self.RAG_KEYWORDS = os.getenv('RAG_KEYWORDS')
        self.FREE_PROMPT_TEMPLATE = os.getenv('FREE_PROMPT_TEMPLATE')

        # Embedding Model
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

    @property
    def MLC_MODEL(self) -> str:
        """
        Get the active MLC model path based on current default.
        Always returns HF:// format for remote models.
        """
        # If there's a direct path set, use it (backwards compatibility)
        if self._direct_mlc_model_path:
            if self._direct_mlc_model_path.startswith('HF://'):
                return self._direct_mlc_model_path
            # If local path exists, return as is
            if os.path.exists(self._direct_mlc_model_path):
                return self._direct_mlc_model_path
            return f"HF://{self._direct_mlc_model_path}"

        # Get model from mapping
        model_id = self._mlc_model_map.get(self._default_mlc_model)
        if not model_id:
            model_id = self._mlc_model_map.get('default', '')

        # Always return HF:// format for MLC models unless it's a local path
        if os.path.exists(model_id):
            return model_id
        return f"HF://{model_id}"

    @MLC_MODEL.setter
    def MLC_MODEL(self, value: str) -> None:
        """Set the active MLC model"""
        # Store the direct path for backwards compatibility
        self._direct_mlc_model_path = value

        if value.startswith('HF://'):
            # Store without the HF:// prefix in the mapping
            model_id = value[5:]
            # Check if this matches any existing key
            for key, val in self._mlc_model_map.items():
                if val == model_id:
                    self._default_mlc_model = key
                    return
            # If not found, store as custom
            self._default_mlc_model = 'custom'
            self._mlc_model_map['custom'] = model_id
        else:
            # Check if this is a key in our map
            if value in self._mlc_model_map:
                self._default_mlc_model = value
            else:
                # If not found, treat as a direct path
                self._default_mlc_model = 'custom'
                self._mlc_model_map['custom'] = value

    @property
    def GGUF_MODEL(self) -> str:
        """
        Get the active GGUF model filename.
        For GGUF, we always expect a local file in GGUF_CACHE_DIR.
        """
        # If there's a direct path set, use it (backwards compatibility)
        if self._direct_gguf_model_path:
            # If it's a full path and exists, return as is
            if os.path.exists(self._direct_gguf_model_path):
                return self._direct_gguf_model_path
            # Otherwise, it's just a filename
            return self._direct_gguf_model_path

        # Get model from mapping
        model_name = self._gguf_model_map.get(self._default_gguf_model)
        if not model_name:
            model_name = self._gguf_model_map.get('default', '')

        # For GGUF, just return the filename (assumed to be in GGUF_CACHE_DIR)
        return model_name

    @GGUF_MODEL.setter
    def GGUF_MODEL(self, value: str) -> None:
        """Set the active GGUF model"""
        self._direct_gguf_model_path = value

        if value in self._gguf_model_map:
            self._default_gguf_model = value
        else:
            self._default_gguf_model = 'custom'


    def get_mlc_model_map(self) -> Dict[str, str]:
        """Get the MLC model mapping dictionary"""
        return self._mlc_model_map

    def get_gguf_model_map(self) -> Dict[str, str]:
        """Get the GGUF model mapping dictionary"""
        return self._gguf_model_map

    def get_gguf_model_path(self) -> str:
        """Get the full path to the active GGUF model"""
        model_filename = self.GGUF_MODEL
        # If it's already a full path, return as is
        if os.path.exists(model_filename):
            return model_filename
        # Otherwise, join with the cache dir
        return os.path.join(self.GGUF_CACHE_DIR, model_filename)