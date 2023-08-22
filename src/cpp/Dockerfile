FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install gcc \
         python3 \
         python3-pip \
         python3-venv \
         make \
         cmake \
         wget \
         unzip \
         libffi-dev \
         libopencv-dev -y

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /

RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
RUN unzip "libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip"

WORKDIR /workspace

COPY CMakeLists.txt .
COPY measure.py .
COPY trainc.py .
COPY unet.cpp .
COPY unet.h .
COPY utils.h .
COPY main.cpp .

RUN mkdir build
WORKDIR /workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release ..
RUN make
WORKDIR /workspace
ENTRYPOINT ["python", "trainc.py"]