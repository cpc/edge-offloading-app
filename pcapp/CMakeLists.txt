cmake_minimum_required(VERSION 3.1)

project(pcapp LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "DEBUG")

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
set(EXTERNAL_DIR ${ROOT_DIR}/external)
set(APP_DIR ${ROOT_DIR}/android/app/src/main/cpp)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn imgcodecs imgproc)

add_library(libyuv SHARED IMPORTED)
set_target_properties(
  libyuv PROPERTIES IMPORTED_LOCATION
                    ${ROOT_DIR}/external/libyuv/build/libyuv.so)

add_executable(
  ${PROJECT_NAME}
  main.cpp ${APP_DIR}/poclImageProcessor.cpp ${APP_DIR}/poclImageProcessor.h
  ${APP_DIR}/sharedUtils.h ${APP_DIR}/event_logger.c ${APP_DIR}/event_logger.h)

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${EXTERNAL_DIR}/pocl/include
                         ${EXTERNAL_DIR}/libyuv/include ${APP_DIR})

target_link_libraries(
  ${PROJECT_NAME}
  OpenCL::OpenCL
  opencv_core
  opencv_dnn
  opencv_imgproc
  opencv_imgcodecs
  libyuv)

target_compile_definitions(
  ${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300
                          CL_HPP_TARGET_OPENCL_VERSION=300)
