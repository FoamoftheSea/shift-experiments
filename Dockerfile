FROM nvcr.io/nvidia/pytorch:23.08-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu117"

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y wget software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10

# Install bitsandbytes
RUN git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    CUDA_VERSION=117 make cuda11x && \
    python setup.py install && \
    cd ..

# Install shift_lab
RUN git clone https://github.com/FoamoftheSea/shift-experiments.git && \
    cd shift-experiments && \
    pip install -e .
