# SHIFT Lab
This repository contains a set of tools for training models with the SHIFT dataset. [The SHIFT dataset](https://www.vis.xyz/shift/) is a multi-task synthetic driving dataset that was created using the [CARLA simulator](https://carla.org/). As such, it provides an opportunity to train models which can be deployed in autonomous driving stacks operating inside CARLA, providing a rich research opportunity in AV stack development.

### Models

- [Multitask Segformer](./shift_lab/models/segformer) - *read the [blog post](https://hiddenlayers.tech/blog/segformer-demonstrates-powerful-multitask-performance) and [WandB Report](https://api.wandb.ai/links/indezera/4ua2bsyk)*
  - Uses a single encoder (hierarchical transformer encoder from [Segformer](https://arxiv.org/abs/2105.15203)) to feed features to two task heads (possibly could be expanded to include others). Can be constructed with Segformer B0-B5 for desired accuracy/efficiency tradeoff.
    1. Semantic Segmentation from all-MLP decoding head from Segformer
    2. Monocular Depth Estimation from [GLPN](https://arxiv.org/abs/2201.07436) decoding head

## Setup
### Docker (recommended)
Since it can be difficult to get your pytorch, cuda, and bitsandbytes installation to work nicely together, you can save yourself the headache by building this Docker image and working inside of it.

Navigate in your terminal into this repository. With [Docker](https://www.docker.com/) installed on your system:
1. Build Docker image (this may take a while):
   - `docker build -f Dockerfile . -t shift_lab:latest`
2. Run Docker image:
   - `docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 shift_lab:latest`
3. Run a training script (Use --help to discover params):
   - `python shift-experiments/shift_lab/semantic_segmentation/segformer/train_segformer.py --help`

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
   - If on linux: `export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu117`
   - `pip install -e .`
6. Install bitsandbytes:
   - git clone https://github.com/TimDettmers/bitsandbytes.git
   - cd bitsandbytes
   - CUDA_VERSION=117 make cuda11x
   - python setup.py install
   - cd ..
7. Run a training script (Use --help to discover params):
   - `python shift-experiments/shift_lab/models/segformer/train_segformer.py --help`