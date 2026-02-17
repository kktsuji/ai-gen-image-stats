FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set timezone
ENV TZ=Asia/Tokyo

WORKDIR /work

# Install system packages and configure timezone
RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python packages
RUN python3 -m pip install --no-cache-dir -U pip && \
    # TODO: multi-stage build to separate dev and prod images
    python3 -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
