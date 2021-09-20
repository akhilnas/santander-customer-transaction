FROM ubuntu:20.04
FROM nvidia/cuda:11.2.0-base-ubuntu20.04

# Check NVIDIA 
CMD nvidia-smi


LABEL maintainer="Akhil Nasser <eakhil711@gmail.com>"
LABEL version="1.0"
LABEL description="PyTorch and Xgboost Container"

# Prevent Dialog during installation
ENV DEBIAN_FRONTEND=noninteractive 

# Install Required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
  nano \
  g++ \
  wget \
  git \
  python3.7 \
  python3-pip \
  python3-setuptools \
  python3-tk \
  python3-matplotlib 

# Copy Relevant Files
COPY requirements.txt /Code/requirements.txt
COPY training.py /Code/training.py
COPY hyperparameter.py /Code/hyperparameter.py
COPY data /data
COPY main_script.sh /Code/main_script.sh

# Install Python Libraries
RUN pip3 install --upgrade pip==21.2.4
RUN pip3 install -r /Code/requirements.txt

# Expose Port for Mlflow UI
EXPOSE 5000

# Necessary Commands
RUN chmod +x /Code/main_script.sh
CMD ./Code/main_script.sh
