FROM pocl-devel-gpu:llvm-16

RUN apt update

### setup opencv

RUN apt-get install -y git libcudnn8-dev

RUN pip3 install numpy

ADD ./docker_build_opencv.bash /

RUN ./docker_build_opencv.bash 6.1 4.7.0

# OpenCV 4.7.0 still needs libcublaLt.so.12
# and it is easier to install it after opencv is built
RUN apt-get install -y libcublas-12-1 libcublas-dev-12-1

### download onnx runtime

ENV ONNX_VERSION="1.15.0"

RUN wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz | tar -xzvf- --directory /opt/

### adding ssh

RUN apt-get install -y openssh-server

RUN useradd -m -s /bin/bash sshuser

RUN echo "sshuser:logon" | chpasswd

ENTRYPOINT service ssh start && bash

### adding remote clion required stuff

RUN apt-get install -y gdb rsync wget

### ffmpeg requirements

RUN apt-get install -y ffmpeg libavformat-dev libavcodec-dev
