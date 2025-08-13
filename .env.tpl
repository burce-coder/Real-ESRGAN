# Real-ESRGAN WebServer Configuration

# Default model settings
DEFAULT_MODEL_NAME=RealESRGAN_x4plus
DEFAULT_DENOISE_STRENGTH=0.5
DEFAULT_OUTSCALE=4
DEFAULT_TILE=0
DEFAULT_TILE_PAD=10
DEFAULT_PRE_PAD=0
DEFAULT_FACE_ENHANCE=true
DEFAULT_FP32=false
DEFAULT_ALPHA_UPSAMPLER=realesrgan
DEFAULT_EXT=auto
DEFAULT_GPU_ID=

# Server settings
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ENABLE_DOCS=false

# Model paths
WEIGHTS_DIR=weights
GFPGAN_MODEL_URL=https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth

# Available models (comma separated)
AVAILABLE_MODELS=RealESRGAN_x4plus,RealESRNet_x4plus,RealESRGAN_x4plus_anime_6B,RealESRGAN_x2plus,realesr-animevideov3,realesr-general-x4v3