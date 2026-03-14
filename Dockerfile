# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV TZ=Asia/Tokyo \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work

# Copy Python version file
COPY .python-version .

# Install Python and system dependencies
RUN PYTHON_VERSION=$(cat .python-version | tr -d '\n' | cut -d. -f1,2) && \
    apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    tzdata && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements files and install Python packages in one layer
COPY requirements.txt requirements-dev.txt ./
# TODO: multi-stage build to separate dev and prod images
RUN python3 -m pip install --no-cache-dir -U --ignore-installed pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt && \
    rm -rf /root/.cache/pip
