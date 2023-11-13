#!/usr/bin/env bash

## how to use said script
## 1. add the nvidia ppa this can be done by following these instructions:
## https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
## NOTE: make sure to select network installer
## 2. apt install cuda-11-7 (note: cuda 11.5 that comes with ubuntu 22.04 is bugged and will break during compilation)
## 3. download the opencv repo and checkout the 4.7.0 release
## 4. download the opencv_contrib repo and checkout the 4.7.0 release
## 5. create a python virtual environment that has numpy installed
## 6. make sure LD_LIBRARY_PATH has '/lib/x86_64-linux-gnu'
## 7. set the CMAKE_INSTALL_PREFIX to your local directory
## 8. set the CUDA_ARCH_BIN to your GPU (check https://developer.nvidia.com/cuda-gpus which arch you have)
## 9. set OPENCV_EXTRA_MODULES_PATH to where the modules are in opencv_contrib repo you downloaded
## 10. point the PYTHON_EXECUTABLE  to your python virtual environment
## 11. set the number of threads for make to match your cpu thread count
## 12. make sure CUDA_VERSION, CUDADIR, CUDA_PATH AND CUDA_INSTALL_PATH are set to the right
## 12. copy this file to the opencv repo and execute it
##
## this script is losely based on the following tutorial and might be useful
## https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

if [ "$#" -ne 2 ]; then
	echo "not the correct number of arguments"
	# both return and exit for a smooth exit
	return -1 2>/dev/null
	exit -1 
fi

ARCH_BIN=$1
OPENCV_VERSION=$2

echo -e "ARCH_BIN=$ARCH_BIN\nOPENCV_VERSION=$OPENCV_VERSION"


mkdir opencv_build

cd opencv_build

git clone --branch $OPENCV_VERSION --depth 1 https://github.com/opencv/opencv.git

git clone --branch $OPENCV_VERSION --depth 1 https://github.com/opencv/opencv_contrib.git

cd opencv

mkdir build
cd build

cmake -G "Unix Makefiles" \
	-D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local/ \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D CUDA_ARCH_BIN=$ARCH_BIN \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D BUILD_EXAMPLES=ON ..

# leave one core to do other things
make -j $(($(nproc)-1))
make install

cd ../../../

rm -rf opencv_build/

