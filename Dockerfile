FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /work

COPY requirements.txt requirements-dev.txt ./

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir -U pip && \
    # TODO: multi-stage build to separate dev and prod images
    python3 -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
