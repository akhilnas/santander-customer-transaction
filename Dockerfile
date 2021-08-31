FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
LABEL maintainer="Akhil Nasser <eakhil711@gmail.com>"
LABEL version="1.0"
LABEL description="PyTorch and Xgboost Container"

# Install Required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  nano \
  g++ \
  wget \
  git \
  python3-pip \
  python3-setuptools \
  python3-tk \
  python3-matplotlib 

# Copy Relevant Files
COPY requirements.txt /Code/requirements.txt
COPY training.py /Code/training.py
COPY data /Code/data

# Install Python Libraries
RUN pip install -r requirements.txt