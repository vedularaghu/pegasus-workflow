FROM ubuntu:16.04 

WORKDIR /usr/bin

COPY add_thumbs_up.py ./

RUN chmod 777 ./add_thumbs_up.py 
RUN apt update && apt install python3-pip -y
RUN pip3 install emoji



