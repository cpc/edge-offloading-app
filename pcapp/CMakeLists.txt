# how to build:
# call cmake with the following parameters, for example:
# -DONNXRUNTIME_INCLUDE_DIRS=/usr/local/onnxruntime-linux-x64-gpu-1.15.0/include
# -DONNXRUNTIME_LIBRARIES=/usr/local/onnxruntime-linux-x64-gpu-1.15.0/lib/libonnxruntime.so
# -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7
# and run the app with the following env variables:
# CUDA_INSTALL_PATH=/usr/local/cuda-11.7
# CUDA_PATH=/usr/local/cuda-11.7
# CUDADIR=/usr/local/cuda-11.7
# LD_LIBRARY_PATH=/usr/local/onnxruntime-linux-x64-gpu-1.15.0/lib:/usr/local/cuda-11.7/lib64:/lib/x86_64-linux-gnu
# POCL_BUILDING=1
# POCL_DEBUG=warn,err,general
# POCL_DEVICES=basic
# POCL_ENABLE_UNINT=1

cmake_minimum_required(VERSION 3.22.1)
include(ExternalProject)
include(ProcessorCount)

project(pcapp LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "DEBUG")

set(POCL_ASAN 0)
if(ENABLE_ASAN)
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)
set(POCL_ASAN 1)
endif(ENABLE_ASAN)

ProcessorCount(CORES)
if(CORES EQUAL 0 )
set(EXTERNAL_PROJECT_CORES 8)
else(CORES EQUAL 0)
set(EXTERNAL_PROJECT_CORES  ${CORES})
endif(CORES EQUAL 0)
MESSAGE("EXTERNAL_PROJECT_CORES: ${EXTERNAL_PROJECT_CORES}")

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(EXTERNAL_DIR ${ROOT_DIR}/external)
set(APP_DIR ${ROOT_DIR}/android/app/src/main/cpp)

find_package(OpenCV REQUIRED COMPONENTS core dnn imgcodecs imgproc)

set(POCL_CMAKE_ARGS
        -DENABLE_ASAN=${POCL_ASAN}
        -DDISABLE_PTHREAD=1
        -DENABLE_LLVM=0
        -DENABLE_LOADABLE_DRIVERS=0
        -DENABLE_ICD=0
#        -DENABLE_PROXY_DEVICE=1
        -DENABLE_REMOTE_CLIENT=1
        -DENABLE_REMOTE_SERVER=0
        -DENABLE_HOST_CPU_DEVICES=1
        -DENABLE_TRAFFIC_MONITOR=1
        -DENABLE_HWLOC=0
        -DHOST_DEVICE_BUILD_HASH=00000000
        -DENABLE_POCLCC=0
        -DENABLE_TESTS=0
        -DENABLE_EXAMPLES=0
        -DBUILD_SHARED_LIBS=0
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pocl
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DVISIBILITY_HIDDEN=0
        -DOpenCV_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni
        -DENABLE_OPENCV_ONNX=YES
        -DONNXRUNTIME_INCLUDE_DIRS=${ONNXRUNTIME_INCLUDE_DIRS}
        -DONNXRUNTIME_LIBRARIES=${ONNXRUNTIME_LIBRARIES})

ExternalProject_Add(pocl
        SOURCE_DIR ${EXTERNAL_DIR}/pocl
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${POCL_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libOpenCL.a)

add_library(libpocl SHARED IMPORTED)
set_target_properties(libpocl
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libOpenCL.a)
 
set(LIBYUV_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/libyuv)

ExternalProject_Add(build_libyuv
        SOURCE_DIR ${EXTERNAL_DIR}/libyuv
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${LIBYUV_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libyuv/lib/libyuv.so)

add_library(libyuv SHARED IMPORTED)
set_target_properties(libyuv
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/libyuv/lib/libyuv.so)

add_executable(
  ${PROJECT_NAME}
  main.cpp ${APP_DIR}/poclImageProcessor.cpp ${APP_DIR}/poclImageProcessor.h
  ${APP_DIR}/sharedUtils.h ${APP_DIR}/event_logger.c ${APP_DIR}/event_logger.h)

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${EXTERNAL_DIR}/pocl/include
                         ${EXTERNAL_DIR}/libyuv/include ${APP_DIR})


add_dependencies(${PROJECT_NAME}
        pocl
        build_libyuv)

target_link_libraries(
    ${PROJECT_NAME}
    libpocl
    opencv_core
    opencv_dnn
    opencv_imgproc
    opencv_imgcodecs
    libyuv
    ${ONNXRUNTIME_LIBRARIES}
)

target_compile_definitions(
  ${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300
                          CL_HPP_TARGET_OPENCL_VERSION=300)

