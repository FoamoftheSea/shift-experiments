# SHIFT Lab
This repository contains a set of tools for training models with the SHIFT dataset. [The SHIFT dataset](https://www.vis.xyz/shift/) is a multi-task synthetic driving dataset that was created using the [CARLA simulator](https://carla.org/). As such, it provides an opportunity to train models which can be deployed in autonomous driving stacks operating inside CARLA, providing a rich research opportunity in AV stack development.

### Models

- [Multiformer](https://github.com/FoamoftheSea/transformers/blob/multiformer/src/transformers/models/multiformer/modeling_multiformer.py) - *read the [blog post](https://natecibik.medium.com/multiformer-51b81df826b7) and [WandB Report](https://api.wandb.ai/links/indezera/gy0jftkc)*
  - Expansion of Multitask Segformer into 2D object detection by incorporating a scaled-down version of [Deformable DETR](https://arxiv.org/abs/2010.04159).
  - Usage scripts:
    - [Train/eval](scripts/model_train_eval/train_multiformer.py)
    - [Inference](scripts/inference/multiformer_inference.py)


- [Multitask Segformer](./shift_lab/models/multitask_segformer) - *read the [blog post](https://hiddenlayers.tech/blog/segformer-demonstrates-powerful-multitask-performance) and [WandB Report](https://api.wandb.ai/links/indezera/4ua2bsyk)*
  - Uses a [PVTv2](https://arxiv.org/abs/2106.13797) backbone with two task heads (possibly could be expanded to include others). Selection of B0-B5 model sizes for desired accuracy/efficiency tradeoff.
    - Semantic Segmentation using all-MLP decoding head from [Segformer](https://arxiv.org/abs/2105.15203).
    - Monocular Depth Estimation from [GLPN](https://arxiv.org/abs/2201.07436) decoding head.
  - Usage scripts:
    - [Train/eval](scripts/model_train_eval/train_multitask_segformer.py)
    - [Inference](scripts/inference/multitask_segformer_inference.py)

## Setup

### Local

If you'd prefer to work outside the Docker container, you can set up like this:
1. Install latest [NVIDIA driver](https://www.nvidia.com/download/index.aspx) for your hardware.
2. Install [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
3. Set up a virtual environment:
   - `python -m venv ./venv`
4. Activate virtual environment:
   - Windows Powershell: `./venv/Scripts/Activate.ps1`
   - Linux: `source ./venv/bin/activate`
5. Install repo:
   1. Set environment variable
      - Linux: `export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu117`
      - Windows: `$env:PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu117"`
   2. `pip install -e .`
6. Install bitsandbytes (optional):
   - git clone https://github.com/TimDettmers/bitsandbytes.git
   - cd bitsandbytes
   - CUDA_VERSION=117 make cuda11x
   - python setup.py install
   - cd ..
7. Run a training script (Use --help to discover params):
   - `python ./scripts/model_train_eval/train_multiformer.py --help`

### Docker

Navigate in your terminal into this repository. With [Docker](https://www.docker.com/) installed on your system:
1. Build Docker image (this may take a while):
   - `docker build -f Dockerfile . -t shift_lab:latest`
2. Run Docker image:
   - `docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 shift_lab:latest`
3. Run a training script (Use --help to discover params):
   - `python shift-experiments/scripts/model_train_eval/train_multitask_segformer.py --help`
