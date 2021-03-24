FROM ubuntu:16.04

RUN apt update
RUN apt upgrade
RUN apt install git

RUN git clone https://github.com/renancesarr/leitura-de-placa-de-carros.git

RUN cd leitura-de-placa-de-carro/

RUN cd darknet

RUN apt install cmake

RUN cd ..

RUN apt install python2.7

RUN apt install curl

RUN apt install wget

RUN cd darknet && make
RUN cd ..

RUN bash get-networks.sh

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py

RUN apt install python-pip


RUN pip install keras==2.2.4
RUN pip install tensorflow==1.5.0
RUN pip install opencv-python==4.2.0.32
RUN pip install numpy==1.14
RUN pip install flask==1.1.2

RUN apt install libgtk2.0-dev

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]