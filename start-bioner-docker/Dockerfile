FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
# FROM ubuntu:focal-20231003
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1050
ARG USER_GID=100
ARG USER_NAME=vscode

RUN apt-get update && \
  apt-get install -y git wget unzip bzip2 build-essential && \
  apt-get install -y gcc g++ make rsync protobuf-compiler && \
  apt-get install -y pkg-config ffmpeg libsm6 libxext6 && \
  # apt-get install -y tesseract-ocr && \
  apt-get install -y python3 python3-pip && \
  apt-get install -y libpq-dev python3-dev curl && \
  rm -rf /var/lib/apt/lists/*

# Create conda environment karateclub
WORKDIR /workspace
COPY . /workspace
# Use bash so we can "conda activate"
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n gen python=3.10 && \
    source activate gen && \
    pip install --upgrade-strategy only-if-needed -r requirements_gen.txt

RUN pip install --upgrade pip && \
  pip install --upgrade-strategy only-if-needed -r requirements_base.txt && \
  python3 -m spacy download en_core_web_lg && \
  python3 -m spacy download en_core_web_trf && \
  python3 -m spacy download en_core_web_md && \
  python3 -m spacy download en_core_web_sm

RUN useradd $USER_NAME -u $USER_UID -g $USER_GID -m -s /bin/bash
RUN chown -R $USER_UID:$USER_GID /workspace

RUN python3 /workspace/test_imports.py \
    && echo "All imports succeeded!" \
    || (echo "Some imports failed!" && exit 1)