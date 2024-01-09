# $ docker build -t local/uniter ./
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:22.08-py3


RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install wget curl

ENV CUDA_HOME /usr/local/cuda

ARG PYTHON_VERSION=3.8
# https://pytorch.org/
ARG TORCH_CUDA_VERSION=cu111
ARG TORCH_VERSION=1.10
ARG TORCHVISION_VERSION=0.11.1
ARG TORCH_SOURCE=https://download.pytorch.org/whl/cu111/torch_stable.html

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake
RUN unattended-upgrade
RUN apt-get -y autoremove

# aptパッケージインストール時のプロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 依存ライブラリのインストール
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        # for development
        vim curl time \
        # for pillow
        libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
        libharfbuzz-dev libfribidi-dev libxcb1-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Update & upgrade
RUN apt-get -y update && apt-get -y upgrade

# DockerコンテナからPythonのOpenCVを使う際に必要
# https://cocoinit23.com/docker-opencv-importerror-libgl-so-1-cannot-open-shared-object-file/
RUN apt-get install -y libgl1-mesa-dev

# pythonのインストール
RUN apt-get install -y python${PYTHON_VERSION} python3-pip && \
    echo alias python=${PYTHON_VERSION} >> ~/.bashrc

# 作業ディレクトリの作成
RUN mkdir /src

# 作業ディレクトリの指定
WORKDIR /src

COPY requirements.txt /src/requirements.txt

# crt-server実行環境の設定
RUN pip install -U pip \
    && pip install --upgrade pip \
    && pip install -r /src/requirements.txt

# PyTorch, mmcv-full, detectron2
# RUN python3 -m pip install detectron2 -f \
#   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# RUN pip install torch==${TORCH_VERSION} \
#                 torchvision==${TORCHVISION_VERSION} \
#                 --extra-index-url ${TORCH_SOURCE}
# RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
# RUN pip install detectron2 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
# RUN wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz \
    # && pip install en_vectors_web_lg-2.1.0.tar.gz

# wandbのために必要
RUN git config --global --add safe.directory /src
# RUN wandb login {your id}
