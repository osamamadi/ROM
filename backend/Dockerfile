# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs: libgl1/libglib2.0-0 for OpenCV; build-essential only as a fallback
# in case xtcocotools has to compile its C extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 0) Global pins: every later `pip install` must respect these, so no transitive dependency
#    can upgrade numpy to 2.x or pull a scipy build that is incompatible with numpy 1.x.
COPY constraints.txt /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# 1) CPU-only PyTorch (small wheels, no CUDA). numpy 1.x is required by torch 2.1 / mmcv 2.1.
RUN pip install numpy==1.26.4 \
    && pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 2) mmcv: use the PREBUILT CPU wheel matching torch 2.1.0 / cp310 (compiling mmcv takes ~1h and OOMs on Railway).
#    --only-binary makes the build fail fast rather than silently falling back to a source build.
RUN pip install mmengine==0.10.4 \
    && pip install --only-binary=:all: mmcv==2.1.0 \
       -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html

# 3) MMPose WITHOUT its dependency list (it pins `chumpy`, which fails to build on modern pip/setuptools),
#    then install only what MMPose imports at runtime.
RUN pip install --no-deps mmpose==1.3.2 \
    && pip install json_tricks matplotlib munkres scipy pycocotools xtcocotools \
       opencv-python-headless==4.10.0.84 pillow rich tabulate \
    && pip install --no-deps mmdet==3.2.0 \
    && pip install shapely terminaltables six

# 4) App dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Fail the BUILD (not the deploy) if the ML stack is broken.
RUN python -c "import torch, mmcv, mmengine, mmpose; from mmpose.apis import inference_topdown, init_model; \
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmpose', mmpose.__version__)"

COPY . .

# Weights are downloaded at app startup (see main.py), not baked into the image.
RUN mkdir -p /app/weights

EXPOSE 8000
# Railway injects $PORT. One worker: the model lives in memory once.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
