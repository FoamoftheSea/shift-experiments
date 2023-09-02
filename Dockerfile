# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu117"

#RUN apt update -y && apt upgrade -y && \
#    apt-get install -y \
#    wget build-essential \
#    checkinstall  \
#    libreadline-gplv2-dev  \
#    libncursesw5-dev  \
#    libssl-dev  \
#    libsqlite3-dev \
#    tk-dev \
#    libgdbm-dev \
#    libc6-dev \
#    libbz2-dev \
#    libffi-dev \
#    zlib1g-dev && \
#    cd /usr/src && \
#    wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
#    tar xzf Python-3.10.11.tgz && \
#    cd Python-3.10.11 && \
#    ./configure --enable-optimizations && \
#    make install

RUN apt-get update && \
    apt-get install -y software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin && \
    mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb && \
    dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb && \
    cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda

RUN git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    CUDA_VERSION=117 make cuda11x && \
    python setup.py install

RUN git clone https://github.com/FoamoftheSea/shift-experiments.git
RUN cd shift-experiments && pip install -e .
