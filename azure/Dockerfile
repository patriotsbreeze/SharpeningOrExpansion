# Provider-agnostic image. The primary Azure path uses bootstrap.sh on the Ubuntu-HPC
# marketplace image, which already ships the NVIDIA driver and CUDA. This Dockerfile exists
# for Azure Batch / AKS, or for pinning the exact toolchain independently of the image.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY fixtures ./fixtures
COPY scripts ./scripts

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir ".[gpu]"
ENV PATH=/opt/venv/bin:$PATH

ENTRYPOINT ["soe"]
CMD ["--help"]
