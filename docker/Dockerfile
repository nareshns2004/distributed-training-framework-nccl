FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source and build
COPY . .
RUN pip3 install .

ENTRYPOINT ["python3"]
