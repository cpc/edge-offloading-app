
cmake_minimum_required(VERSION 3.22.1)

add_executable(testRawImageReader
        testRawImageReader.cpp
        ${APP_DIR}/RawImageReader.cpp ${APP_DIR}/RawImageReader.hpp
        ${APP_DIR}/poclImageProcessorTypes.h
        ${APP_DIR}/poclImageProcessorUtils.cpp)

target_include_directories(testRawImageReader
        PRIVATE
        ${EXTERNAL_DIR}/pocl/include
        ${APP_DIR}
        ${EXTERNAL_DIR}/tracy/public/tracy)

target_link_libraries(testRawImageReader
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
)