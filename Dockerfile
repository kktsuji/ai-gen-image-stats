FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

COPY requirements.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    python3 -m pip install -U pip && \
    python3 -m pip install -r requirements.txt