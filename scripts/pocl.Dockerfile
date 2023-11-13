FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

### setting up llvm deb
RUN apt-get update

RUN apt-get install -y wget

ENV LLVM_VERSION=16

RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" >> /etc/apt/sources.list.d/llvm.list

RUN echo "deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" >> /etc/apt/sources.list.d/llvm.list

RUN apt-get update

### installing pocl dependencies

RUN apt-get install -y python3-dev \
  python3 \
  python3-pip \
  cmake \
  build-essential \
  libpython3-dev \
  ocl-icd-libopencl1 \
  pkg-config \
  libclang-${LLVM_VERSION}-dev \
  clang-${LLVM_VERSION} \
  llvm-${LLVM_VERSION} \
  make \
  ninja-build \
  ocl-icd-libopencl1 \
  ocl-icd-dev \
  ocl-icd-opencl-dev \
  libhwloc-dev \
  zlib1g \
  zlib1g-dev \
  clinfo \
  dialog \
  apt-utils \
  libxml2-dev \
  libclang-cpp${LLVM_VERSION}-dev \
  libclang-cpp${LLVM_VERSION} \
  llvm-${LLVM_VERSION}-dev
