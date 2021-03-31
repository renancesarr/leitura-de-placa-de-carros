FROM ubuntu:16.04

RUN apt update
RUN apt upgrade

WORKDIR /home/frotaaki

COPY . .

RUN cd darknet

RUN apt install cmake -y

RUN cd ..

RUN apt install python2.7 -y

RUN apt install python-pip -y

RUN apt install curl -y

RUN apt install wget -y

RUN cd darknet && make
RUN cd ..

RUN bash get-networks.sh

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py

RUN python get-pip.py

RUN apt install python-pip -y


RUN pip install keras==2.2.4
RUN pip install tensorflow==1.5.0
RUN pip install opencv-python==4.2.0.32
RUN pip install numpy==1.14
RUN pip install flask==1.1.2

RUN apt install libgtk2.0-dev -y

RUN export FLASK_APP=app.py

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0" ]