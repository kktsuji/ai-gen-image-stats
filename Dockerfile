# syntax=docker/dockerfile:1

# Configurable timezone and user ID for runtime
ARG TZ=Asia/Tokyo
ARG APP_UID=1000
ARG APP_GID=1000

# ---- Builder stage: install all dependencies (prod + dev) ----
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS builder

ARG TZ
ENV TZ=${TZ} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work

COPY .python-version .

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

COPY requirements.txt requirements-dev.txt ./
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt && \
    rm -rf /root/.cache/pip

# ---- Production stage: only production dependencies ----
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS production

ARG TZ
ARG APP_UID
ARG APP_GID
ENV TZ=${TZ} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work

COPY .python-version .

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
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Create non-root user for runtime (UID/GID configurable via build args)
RUN groupadd --gid ${APP_GID} appuser && \
    useradd --uid ${APP_UID} --gid appuser --create-home appuser
USER appuser
