"""
Real-ESRGAN WebServer Configuration

This module contains all configuration settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class that loads settings from environment variables."""

    # Default model settings
    DEFAULT_MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME', 'RealESRGAN_x4plus')
    DEFAULT_DENOISE_STRENGTH = float(os.getenv('DEFAULT_DENOISE_STRENGTH', '0.5'))
    DEFAULT_OUTSCALE = float(os.getenv('DEFAULT_OUTSCALE', '4'))
    DEFAULT_TILE = int(os.getenv('DEFAULT_TILE', '0'))
    DEFAULT_TILE_PAD = int(os.getenv('DEFAULT_TILE_PAD', '10'))
    DEFAULT_PRE_PAD = int(os.getenv('DEFAULT_PRE_PAD', '0'))
    DEFAULT_FACE_ENHANCE = os.getenv('DEFAULT_FACE_ENHANCE', 'false').lower() == 'true'
    DEFAULT_FP32 = os.getenv('DEFAULT_FP32', 'false').lower() == 'true'
    DEFAULT_ALPHA_UPSAMPLER = os.getenv('DEFAULT_ALPHA_UPSAMPLER', 'realesrgan')
    DEFAULT_EXT = os.getenv('DEFAULT_EXT', 'auto')
    DEFAULT_GPU_ID = os.getenv('DEFAULT_GPU_ID', None)

    # Server settings
    SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    SERVER_PORT = int(os.getenv('SERVER_PORT', '8000'))
    # Documentation configuration
    ENABLE_DOCS = os.getenv("ENABLE_DOCS", "true").lower() == "true"

    # Model paths
    WEIGHTS_DIR = os.getenv('WEIGHTS_DIR', 'weights')
    GFPGAN_MODEL_URL = os.getenv('GFPGAN_MODEL_URL', 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth')

    # Available models
    AVAILABLE_MODELS = os.getenv('AVAILABLE_MODELS', 'RealESRGAN_x4plus,RealESRNet_x4plus,RealESRGAN_x4plus_anime_6B,RealESRGAN_x2plus,realesr-animevideov3,realesr-general-x4v3').split(',')

    @classmethod
    def get_gpu_id(cls):
        """Get GPU ID as integer or None."""
        gpu_id = cls.DEFAULT_GPU_ID
        if gpu_id and gpu_id.isdigit():
            return int(gpu_id)
        return None

    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate if model name is in available models list."""
        return model_name in cls.AVAILABLE_MODELS

    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration."""
        return {
            'default_model': cls.DEFAULT_MODEL_NAME,
            'server_host': cls.SERVER_HOST,
            'server_port': cls.SERVER_PORT,
            'weights_dir': cls.WEIGHTS_DIR,
            'available_models': cls.AVAILABLE_MODELS,
        }


# Create a global config instance
config = Config()
