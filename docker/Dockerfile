# syntax = docker/dockerfile:1.2

FROM python:3.9.10-slim-buster AS builder

ARG HAT_GDRIVE_FILE_ID=1vUiknHsRuqhZN25dt2y3jnoPu5SWjRkP

RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends git wget libgl1-mesa-glx libglib2.0-0

WORKDIR /root

RUN git clone https://github.com/WongKinYiu/yolov7.git
WORKDIR /root/yolov7

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install gdown

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install -r requirements.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install coremltools>=4.1 onnx>=1.9.0 onnx-simplifier>=0.3.6
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip3 install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

RUN mkdir -p /root/pretrained

WORKDIR /root/pretrained
RUN --mount=type=cache,mode=0755,target=/root/.cache/downloads wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
RUN python3 /root/yolov7/export.py --weights yolov7.pt --grid --end2end --simplify --dynamic-batch --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

RUN --mount=type=cache,mode=0755,target=/root/.cache/downloads gdown ${HAT_GDRIVE_FILE_ID}

FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 AS runner
COPY --from=builder /root/pretrained /root/pretrained

ENV TZ=Asia/Seoul
ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends software-properties-common
RUN --mount=type=cache,target=/var/cache/apt add-apt-repository ppa:deadsnakes/ppa

RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends libgl1-mesa-glx libglib2.0-0
RUN --mount=type=cache,target=/var/cache/apt apt-get install -yqq --no-install-recommends python3.9 python3-pip

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --upgrade pip

WORKDIR /root
COPY requirements.txt /root
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r /root/requirements.txt

COPY . /app
RUN cp /root/pretrained/* /app/server/plugin/yolov7/pretrained
WORKDIR /app

# ADD https://download.pytorch.org/models/resnet50-19c8e357.pth /root/.cache/torch/hub/checkpoints/

CMD ["python3", "-m", "server"]
