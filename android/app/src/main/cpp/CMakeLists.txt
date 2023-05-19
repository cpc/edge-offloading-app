# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

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
        utils.cpp
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

# build opencv.
set (OPENCV_CMAKE_ARGS
        -DCMAKE_MAKE_PROGRAM=${ANDROID_NDK}/prebuilt/linux-x86_64/bin/make
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        -DANDROID_NDK=${ANDROID_NDK}
        -DANDROID_SDK=${ANDROID_NDK}/../..
        -DCMAKE_BUILD_TYPE=Debug
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DBUILD_SHARED_LIBS=1
        -DBUILD_TESTS=0
        -DBUILD_EXAMPLES=0
        -DOPENCV_DNN_OPENCL=1
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
        BUILD_BYPRODUCTS
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_dnn.so
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgproc.so
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_core.so
        )

add_library(opencv_dnn SHARED IMPORTED)
set_target_properties(opencv_dnn PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_dnn.so)
add_library(opencv_imgproc SHARED IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_imgproc.so)
add_library(opencv_core SHARED IMPORTED)
set_target_properties(opencv_core PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/libs/${ANDROID_ABI}/libopencv_core.so)

# build pocl project
set(POCL_CMAKE_ARGS
        -DENABLE_LLVM=0
        -DENABLE_LOADABLE_DRIVERS=0
        -DENABLE_ICD=0
        -DENABLE_PROXY_DEVICE=1
        -DENABLE_REMOTE_CLIENT=1
        -DENABLE_REMOTE_SERVER=0
        -DENABLE_HOST_CPU_DEVICES=1
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
        -DCMAKE_BUILD_TYPE=Debug
        -DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DVISIBILITY_HIDDEN=0
        -DOpenCV_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni
        )

ExternalProject_Add(pocl
        SOURCE_DIR ${EXTERNAL_PATH}/pocl
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${POCL_CMAKE_ARGS}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        )




add_library(poclnative
        SHARED
        poclRemoteExample.cpp
        poclRemoteExample.h
        sharedUtils.h
        poclImageProcessor.cpp
        )

target_include_directories(poclnative
        PRIVATE
        ${EXTERNAL_PATH}/pocl/include
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni/include
        )

target_link_libraries(
        poclnative
        ${android-lib}
        ${log-lib}
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        OpenCL
        opencv_dnn
        opencv_imgproc
        opencv_core
        )

add_dependencies(pocl build_opencv)
add_dependencies(poclnative pocl)
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
        )