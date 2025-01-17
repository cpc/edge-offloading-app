#!/usr/bin/env bash

# Copy .onnx models to the expected /tmp directory. The yolov8n-seg.onnx (found
# in android/app/src/main/assets) is required. Other YOLOv8 models can be used 
# by generating the .onnx files from https://github.com/ultralytics/ultralytics
cp ../../android/app/src/main/assets/yolov8n-seg.onnx /tmp

# Path to the pocl project (this directory), for example:
POCL_DIR="${HOME}/pocl"

export CUDA_VERSION='12.1'
export CUDADIR="/usr/local/cuda-${CUDA_VERSION}"
export CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
export CUDA_INSTALL_PATH="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:/lib/x86_64-linux-gnu:${POCL_DIR}/build/lib/CL/devices/pthread"
unset OPENCV_LOG_LEVEL

OCL_ICD_VENDORS="${POCL_DIR}/build/ocl-vendors" \
POCL_DEVICES="pthread pthread" \
POCL_BUILDING=1 \
POCL_DEBUG=err,warn,remote \
POCL_ENABLE_UNINT=1 \
POCLD_ALLOW_CLIENT_RECONNECT=1 \
POCL_REMOTE_SERVER_NAME="c1cda2cf102ce70474972eb626517ec6" \
./build/pocld/pocld --log_filter error,warn,remote
