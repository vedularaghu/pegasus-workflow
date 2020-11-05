FROM ubuntu:18.04
WORKDIR /usr/bin
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y \
    python3.6 \
    python3-pip \
    libgl1-mesa-glx
RUN pip3 install --upgrade pip setuptools
RUN pip3 install \
    scikit-learn \
    tensorflow \
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python \
    keras==2.3.1 \
    ray \
    ray[tune] \
    ray[rllib] \
    dataclasses \
    tabulate
COPY preprocess.py ./
RUN chmod 777 ./preprocess.py
COPY data_split.py ./
RUN chmod 777 ./data_split.py
COPY train_model.py ./
RUN chmod 777 ./train_model.py