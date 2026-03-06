FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspaces/torch-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git curl ca-certificates \
    build-essential \
    libgl1 libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

RUN python3.11 -m pip install --no-cache-dir -U pip setuptools wheel

# ----- 사용자(bhj9876) 생성: /home를 볼륨으로 분리하려면 필수 -----
ARG USERNAME=bhj9876
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && mkdir -p /workspaces \
    && chown -R ${USERNAME}:${USERNAME} /workspaces

# 캐시/데이터용 디렉토리 미리 생성(권한 포함)
RUN mkdir -p /data \
    && chown -R ${USERNAME}:${USERNAME} /data
