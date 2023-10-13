# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

# set the number of cores with which to build external projects
set(EXTERNAL_PROJECT_CORES 8)

cmake_minimum_required(VERSION 3.22.1)
include(ExternalProject)

set(EXTERNAL_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../../external/)
# Declares and names the project.

project("poclaisademo")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        poclaisademo

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        vectorAddExample.cpp
        vectorAddExample.h
        )

target_include_directories(poclaisademo PRIVATE
        ${EXTERNAL_PATH}/libopencl-stub/include

        )


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library(android-lib android)

# add the libopencl-stub subdirectory with cmakelists.txt to link the opencl stubs to
add_subdirectory(${EXTERNAL_PATH}/libopencl-stub libopencl-stub-dir)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${EXTERNAL_PATH}/onnxruntime-android/jni/${ANDROID_ABI}/libonnxruntime.so)
add_library(onnxruntime4j_jni SHARED IMPORTED)
set_target_properties(onnxruntime4j_jni PROPERTIES IMPORTED_LOCATION ${EXTERNAL_PATH}/onnxruntime-android/jni/${ANDROID_ABI}/libonnxruntime4j_jni.so)
set(ONNXRUNTIME_LIBRARIES onnxruntime)
set(ONNXRUNTIME_INCLUDE_DIRS "${EXTERNAL_PATH}/onnxruntime-android/headers")

# build opencv.
set(OPENCV_CMAKE_ARGS
        -DCMAKE_MAKE_PROGRAM=${ANDROID_NDK}/prebuilt/linux-x86_64/bin/make
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DANDROID_NDK=${ANDROID_NDK}
        -DANDROID_SDK=${ANDROID_NDK}/../..
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DBUILD_SHARED_LIBS=1
        -DBUILD_TESTS=0
        -DBUILD_EXAMPLES=0
        -DOPENCV_DNN_OPENCL=1
        -DOPENCL_INCLUDE_DIR=${EXTERNAL_PATH}/libopencl-stub/include
        -DOPENCL_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/libopencl-stub-dir/libOpenCL.a
        -DBUILD_LIST=core,dnn,imgproc,imgcodecs
        -DBUILD_ANDROID_EXAMPLES=0
        -DBUILD_ANDROID_PROJECTS=0
        -DWITH_QUIRC=0
        -DBUILD_JAVA=0
        -DBUILD_FAT_JAVA_LIB=0
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/opencv
        )

ExternalProject_Add(build_opencv
        SOURCE_DIR ${EXTERNAL_PATH}/opencv
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${OPENCV_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_BYPRODUCTS
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_dnn.so
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgproc.so
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgcodecs.so
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_core.so
        )

add_library(opencv_dnn SHARED IMPORTED)
set_target_properties(opencv_dnn PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_dnn.so)
add_library(opencv_imgproc SHARED IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgproc.so)
add_library(opencv_imgcodecs SHARED IMPORTED)
set_target_properties(opencv_imgcodecs PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgcodecs.so)
add_library(opencv_core SHARED IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_core.so)

set(TURBO_JPEG_ARGS
        -DANDROID_ABI=${ANDROID_ABI}
#        -DANDROID_ARM_MODE=arm // set in the toolchain
#        -DANDROID_TOOLCHAIN=clang // set in the toolchain
        -DCMAKE_ASM_FLAGS="--target=aarch64-linux-android${ANDROID_VERSION}"
        -DANDROID_NDK=${ANDROID_NDK}
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/libturbojpeg
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DCMAKE_BUILD_TYPE=Release
        )

ExternalProject_Add(build_turbojpeg
        SOURCE_DIR ${EXTERNAL_PATH}/libjpeg-turbo
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_FLAGS ${TURBO_JPEG_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libturbojpeg/lib/libturbojpeg.a
        )

add_library(libturbojpeg SHARED IMPORTED)
set_target_properties(libturbojpeg PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libturbojpeg/lib/libturbojpeg.a)

# build pocl project
set(POCL_CMAKE_ARGS
        -DDISABLE_PTHREAD=1
        -DENABLE_LLVM=0
        -DENABLE_LOADABLE_DRIVERS=0
        -DENABLE_ICD=0
        -DENABLE_PROXY_DEVICE=1
        -DENABLE_REMOTE_CLIENT=1
        -DENABLE_REMOTE_SERVER=0
        -DENABLE_HOST_CPU_DEVICES=1
        -DENABLE_TRAFFIC_MONITOR=1
        -DENABLE_HWLOC=0
        -DHOST_DEVICE_BUILD_HASH=00000000
        -DENABLE_POCLCC=0
        -DENABLE_TESTS=0
        -DENABLE_EXAMPLES=0
        -DENABLE_LOCAL_INTEROP=0
        -DENABLE_EGL_LOCAL_INTEROP=0
        -DBUILD_SHARED_LIBS=0
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/pocl
        -DCMAKE_MAKE_PROGRAM=${ANDROID_NDK}/prebuilt/linux-x86_64/bin/make
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DANDROID_NDK=${ANDROID_NDK}
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DVISIBILITY_HIDDEN=0
        -DOpenCV_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni
        -DENABLE_OPENCV_ONNX=YES
        -DONNXRUNTIME_INCLUDE_DIRS=${ONNXRUNTIME_INCLUDE_DIRS}
        -DONNXRUNTIME_LIBRARIES=${ONNXRUNTIME_LIBRARIES}
        -DENABLE_TURBO_JPEG_COMPRESSION=YES
        -DTURBO_JPEG_LIBRARIES=libturbojpeg
        -DTURBO_JPEG_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/libturbojpeg/include
        )
#-DENABLE_ANDROID_COMPRESSION=YES

ExternalProject_Add(pocl
        SOURCE_DIR ${EXTERNAL_PATH}/pocl
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${POCL_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        )
# some interesting parameters set during init
#message(WARNING "-DANDROID_NDK=${ANDROID_NDK} ")
#message(WARNING "-DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}")
#message(WARNING "-DANDROID_ABI=${ANDROID_ABI}")
#message(WARNING "-DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}")

set(LIBYUV_CMAKE_ARGS
        -DCMAKE_MAKE_PROGRAM=${ANDROID_NDK}/prebuilt/linux-x86_64/bin/make
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DANDROID_NDK=${ANDROID_NDK}
        -DANDROID_SDK=${ANDROID_NDK}/../..
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/libyuv
        )

ExternalProject_Add(build_libyuv
        SOURCE_DIR ${EXTERNAL_PATH}/libyuv
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${LIBYUV_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/libyuv/lib/libyuv.so
        )

add_library(libyuv SHARED IMPORTED)
set_target_properties(libyuv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libyuv/lib/libyuv.so)

add_library(poclnative
        SHARED
        poclRemoteExample.cpp
        poclRemoteExample.h
        sharedUtils.h
        poclImageProcessor.cpp
        utils.cpp
        jniWrapper.cpp
        event_logger.c event_logger.h
        )

target_include_directories(poclnative
        PRIVATE
        ${EXTERNAL_PATH}/pocl/include
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni/include
        ${EXTERNAL_PATH}/libyuv/include
        )

target_link_libraries(
        poclnative
        ${android-lib}
        ${log-lib}
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        OpenCL
        opencv_dnn
        opencv_imgproc
        opencv_imgcodecs
        opencv_core
        ${ONNXRUNTIME_LIBRARIES}
        libyuv
        libturbojpeg
        jnigraphics
)


add_dependencies(pocl build_opencv build_turbojpeg)
add_dependencies(poclnative pocl build_libyuv)
add_dependencies(poclaisademo poclnative)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        poclaisademo

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}

        poclnative

        opencv_imgproc

        opencv_imgcodecs

        opencv_core

        libyuv
        )