#!/usr/bin/env bash

rm -rf build
mkdir build
cd build

# Path to a compiled libjpeg-turbo library, for example:
LIBJPEG_TURBO_LIB="${HOME}/libjpeg-turbo/x86-64/lib/cmake/libjpeg-turbo"

# Path to onnxruntime library, for example:
ONNXRUNTIME_DIR="${HOME}/onnxruntime-linux-x64-gpu-1.15.0"
 
cmake -G Ninja \
    -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-14 \
    -DENABLE_REMOTE_SERVER=YES \
    -DENABLE_LATEST_CXX_STD=YES \
    -DENABLE_CUDA=NO \
    -DENABLE_TESTS=0 \
    -DENABLE_EXAMPLES=1 \
    -DENABLE_SPIR=0 \
    -DENABLE_SPIRV=0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_OPENCV_ONNX=ON \
    -DORT_INSTALL_DIR=${ONNXRUNTIME_DIR} \
    -DENABLE_TURBO_JPEG_DECOMPRESSION=ON \
    -Dlibjpeg-turbo_DIR=${LIBJPEG_TURBO_DIR}\
    -DENABLE_FFMPEG_DECODE=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
    -DTRACY_ENABLE=OFF \
    -DQUEUE_PROFILING=ON \
    -DENABLE_REMOTE_ADVERTISEMENT_DHT=YES\
    ..

ninja -j $(nproc)
