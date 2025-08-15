FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN python -m venv /app/venv
# FORCE_CUDA 是编译安装 pytorch3d 所需要的环境变量
ENV VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    FORCE_CUDA=1

WORKDIR /app
COPY . /app/
RUN pip install --upgrade -r requirements.txt
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /app/weights
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P /app/gfpgan/weights
RUN wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P /app/gfpgan/weights
RUN wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -P /app/gfpgan/weights
RUN modelscope download --model iic/cv_ddcolor_image-colorization --local_dir /app/models/iic/cv_ddcolor_image-colorization

EXPOSE 8080

CMD ["python", "-u", "webserver.py"]