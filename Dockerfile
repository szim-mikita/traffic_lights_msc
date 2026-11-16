FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo
ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONPATH=/app
ENV PATH="$VIRTUAL_ENV/bin:$SUMO_HOME/tools:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-setuptools \
        build-essential \
        libgmp-dev \
        wget \
        ca-certificates \
        git \
        sumo \
        sumo-tools \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
