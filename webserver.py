import sys
import types
import os
import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Import configuration
from config import config

# Create a module for `torchvision.transforms.functional_tensor`
from torchvision.transforms.functional import rgb_to_grayscale
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# Global upsampler instance
upsampler = None
face_enhancer = None
img_colorization = None

def initialize_model(model_name: str = None, **kwargs):
    """Initialize the Real-ESRGAN model"""
    global upsampler, face_enhancer, img_colorization

    if model_name is None:
        model_name = config.DEFAULT_MODEL_NAME

    # Determine models according to model names
    model_name = model_name.split('.')[0]

    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Determine model paths
    model_path = os.path.join(config.WEIGHTS_DIR, model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, config.WEIGHTS_DIR), progress=True, file_name=None)

    # Use dni to control the denoise strength
    dni_weight = None
    denoise_strength = kwargs.get('denoise_strength', config.DEFAULT_DENOISE_STRENGTH)
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Create restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=kwargs.get('tile', config.DEFAULT_TILE),
        tile_pad=kwargs.get('tile_pad', config.DEFAULT_TILE_PAD),
        pre_pad=kwargs.get('pre_pad', config.DEFAULT_PRE_PAD),
        half=not kwargs.get('fp32', config.DEFAULT_FP32),
        gpu_id=kwargs.get('gpu_id', config.get_gpu_id()))

    # Initialize face enhancer if needed
    if kwargs.get('face_enhance', config.DEFAULT_FACE_ENHANCE):
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path=config.GFPGAN_MODEL_URL,
                upscale=kwargs.get('outscale', config.DEFAULT_OUTSCALE),
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        except ImportError:
            if config.ENABLE_MODEL_INIT_LOG:
                print("GFPGAN not available, face enhancement disabled")
            face_enhancer = None
    else:
        face_enhancer = None

    absolute_path = os.path.abspath(os.path.dirname(__file__))
    color_model_path = os.path.join(absolute_path, 'models/iic/cv_ddcolor_image-colorization')
    print(f"color models path: {color_model_path}")
    img_colorization = pipeline(Tasks.image_colorization, model=color_model_path, model_revision=None)

def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Initialize default model on startup"""
#     try:
#         initialize_model()
#         if config.ENABLE_MODEL_INIT_LOG:
#             print("Default model initialized successfully")
#     except Exception as e:
#         if config.ENABLE_MODEL_INIT_LOG:
#             print(f"Failed to initialize default model: {e}")
#     yield
#     # Clean up the ML models and release the resources

initialize_model()

app = FastAPI(
    docs_url="/docs" if config.ENABLE_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_DOCS else None
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# class UpscaleRequest(BaseModel):
#     model_name: Optional[str] = config.DEFAULT_MODEL_NAME
#     denoise_strength: Optional[float] = config.DEFAULT_DENOISE_STRENGTH
#     outscale: Optional[float] = config.DEFAULT_OUTSCALE
#     tile: Optional[int] = config.DEFAULT_TILE
#     tile_pad: Optional[int] = config.DEFAULT_TILE_PAD
#     pre_pad: Optional[int] = config.DEFAULT_PRE_PAD
#     face_enhance: Optional[bool] = config.DEFAULT_FACE_ENHANCE
#     fp32: Optional[bool] = config.DEFAULT_FP32

class UpscaleResponse(BaseModel):
    success: bool
    message: str
    upscaled_image: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Hello World."}

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale_image_file(file: UploadFile = File(...)):
    """
    Upscale an uploaded image file using Real-ESRGAN
    """
    global upsampler, face_enhancer

    try:
        # Validate file type
        content_type = file.content_type
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        # res_header = f'data:{content_type};base64,'
        res_header = f'data:image/png;base64,'

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Perform upscaling
        try:
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=config.DEFAULT_OUTSCALE)
        except RuntimeError as error:
            if "CUDA out of memory" in str(error):
                raise HTTPException(status_code=500, detail="CUDA out of memory. Try using a smaller tile size.")
            else:
                raise HTTPException(status_code=500, detail=f"Error during upscaling: {str(error)}")

        # Convert result to base64
        upscaled_base64 = image_to_base64(output)

        return UpscaleResponse(
            success=True,
            message="ok",
            upscaled_image=f"{res_header}{upscaled_base64}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class ColorizeResponse(BaseModel):
    success: bool
    message: str
    colorized_image: Optional[str] = None

@app.post("/colorize", response_model=dict)
async def colorize_image(file: UploadFile = File(...)):
    """
    Endpoint to upload a grayscale image and receive the colorized image as Base64.

    Args:
        file: Uploaded image file (expected to be an image, e.g., PNG, JPG)

    Returns:
        dict: Contains 'image_base64' key with the colorized image as a Base64 string
    """

    global img_colorization

    try:
        # Validate file type
        content_type = file.content_type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        # Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # res_header = f'data:{content_type};base64,'
        res_header = f'data:image/png;base64,'

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Ensure grayscale input (convert if colored)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Colorize the image
        output = img_colorization(image[..., ::-1])  # Convert to BGR for model
        result = output[OutputKeys.OUTPUT_IMG].astype(np.uint8)

        # Convert result to Base64
        base64_image = image_to_base64(result)

        return ColorizeResponse(
            success=True,
            message="ok",
            colorized_image=f"{res_header}{base64_image}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# @app.get("/models")
# async def get_available_models():
#     """Get list of available models"""
#     return {
#         "models": config.AVAILABLE_MODELS
#     }
#
# @app.get("/config")
# async def get_config():
#     """Get current configuration summary"""
#     return config.get_config_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
