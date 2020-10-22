FROM ubuntu:16.04 

WORKDIR /usr/bin

COPY preprocess.py ./
RUN chmod 777 ./preprocess.py

COPY data_split.py ./
RUN chmod 777 ./data_split.py

COPY train_model.py ./
RUN chmod 777 ./train_model.py

RUN apt-get update && \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa 

RUN apt-get update && apt-get install python3.6 -y

RUN apt install python3-pip -y

COPY requirements.txt ./
RUN chmod 777 ./requirements.txt
RUN pip3 install -r requirements.txt
