FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

RUN python -m pip install -U \
    transformers \
    accelerate \
    pillow \
    tqdm \
    huggingface_hub

WORKDIR /workspace/code

CMD ["bash", "-lc", "sleep infinity"]