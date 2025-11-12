FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

COPY requirements.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    python3 -m pip install -U pip && \
    python3 -m pip install -r requirements.txt