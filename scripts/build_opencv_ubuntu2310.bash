#!/usr/bin/env bash

## how to use said script
## 1. install the nvidia-cuda-toolkit should be higher than cuda 11.5 since that version leads to compilation errors in opencv
## 2. install nvidia-cudnn8
## 3. download the opencv repo and checkout the 4.7.0 release
## 4. download the opencv_contrib repo and checkout the 4.7.0 release
## 5. create and source a python virtual environment that has numpy installed
## 6. run script with the first argument set to the opencv repo path
##
## this script is losely based on the following tutorial and might be useful
## https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/


OPENCV_PATH="."

if [ "$#" -ne 1 ]; then
	echo "could not parse arguments, assuming script is in opencv dir"

else 
    OPENCV_PATH=$1
fi

cd "$OPENCV_PATH" || exit

rm -rf build
mkdir build
cd build || exit

cmake -D CMAKE_BUILD_TYPE=DEBUG \
	-D CMAKE_INSTALL_PREFIX=~/.local \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN="6.1 7.0 7.5 8.6 8.9"  \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE="$(which python)" \
	-D BUILD_EXAMPLES=ON ..

make -j"$(nproc)"
make install

cd - || exit
