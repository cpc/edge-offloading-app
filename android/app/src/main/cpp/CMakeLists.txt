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

# add the libopencl-stub subdirectory with cmakelists.txt to link the opencl stubs to
add_subdirectory(${EXTERNAL_PATH}/libopencl-stub libopencl-stub-dir)

# build pocl project
set(POCL_CMAKE_ARGS
        -DENABLE_LLVM=0
        -DENABLE_LOADABLE_DRIVERS=0
        -DENABLE_ICD=0
        -DENABLE_PROXY_DEVICE=1
        -DENABLE_REMOTE_CLIENT=1
        -DENABLE_REMOTE_SERVER=0
        -DENABLE_HOST_CPU_DEVICES=0
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
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}
        -DVISIBILITY_HIDDEN=0
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
        )

target_link_libraries(
        poclnative
        ${log-lib}
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        OpenCL
        )

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

        # this library comes from libopencl-stub
        OpenCL

        poclnative
        )