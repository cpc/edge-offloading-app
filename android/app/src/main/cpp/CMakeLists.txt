# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

# set the number of cores with which to build external projects
include(ProcessorCount)
ProcessorCount(CORES)
if(CORES EQUAL 0 )
	set(EXTERNAL_PROJECT_CORES 8)
else(CORES EQUAL 0)
	set(EXTERNAL_PROJECT_CORES  ${CORES})
endif(CORES EQUAL 0)

cmake_minimum_required(VERSION 3.22.1)
include(ExternalProject)

set(EXTERNAL_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../../external/)
# Declares and names the project.

project("poclaisademo")

# Set up Tracy and its options
# Note: For the options to take effect, delete cmake cache and reconfigure
option(TRACY_ENABLE "Enable profiling" OFF)
set(TRACY_STATIC OFF)
option(TRACY_NO_SAMPLING "enable tracing with stack traces" ON)
add_subdirectory(${EXTERNAL_PATH}/tracy/ tracy-build)
message(STATUS "TRACY_ENABLE: ${TRACY_ENABLE}")

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
		${EXTERNAL_PATH}/tracy/public/tracy
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

# while not directly used by the app, the app is linked
# to it so that it gets packaged into the apk
add_library(onnxruntime SHARED IMPORTED)
set(ONNXRUNTIME_SO_FILE ${EXTERNAL_PATH}/onnxruntime-android/jni/${ANDROID_ABI}/libonnxruntime.so)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRUNTIME_SO_FILE})

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
        )
# find the jpeg lib that was just built
set(libjpeg-turbo_DIR ${EXTERNAL_PATH}/libjpeg-turbo/libturbojpeg/armv8/lib/cmake/libjpeg-turbo)
find_package(libjpeg-turbo REQUIRED)

# build pocl project
set(POCL_CMAKE_ARGS
        #-DDISABLE_PTHREAD=1
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
		-DORT_INSTALL_DIR=${EXTERNAL_PATH}/onnxruntime-android
        -DENABLE_TURBO_JPEG_COMPRESSION=YES
		-Dlibjpeg-turbo_DIR=${CMAKE_CURRENT_BINARY_DIR}/libturbojpeg/lib/cmake/libjpeg-turbo/
        -DENABLE_ANDROID_MEDIACODEC=YES
        -DENABLE_LATEST_CXX_STD=YES
		-DTRACY_ENABLE=${TRACY_ENABLE}
        )

ExternalProject_Add(pocl
        SOURCE_DIR ${EXTERNAL_PATH}/pocl
        CMAKE_GENERATOR "Unix Makefiles"
        CMAKE_ARGS ${POCL_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target all -- -j ${EXTERNAL_PROJECT_CORES}
        BUILD_ALWAYS True
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/static/libpocl.a
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_add_i8.so
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_opencv_onnx.so
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_turbojpeg.so
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_mediacodec_encoder.so
        )
# some interesting parameters set during init
#message(WARNING "-DANDROID_NDK=${ANDROID_NDK} ")
#message(WARNING "-DANDROID_PLATFORM=${ANDROID_PLATFORM_LEVEL}")
#message(WARNING "-DANDROID_ABI=${ANDROID_ABI}")
#message(WARNING "-DANDROID_NATIVE_API_LEVEL=${ANDROID_PLATFORM_LEVEL}")

# create and configure a config file
configure_file("config.h.in.cmake" "config.h" ESCAPE_QUOTES)
include_directories("${CMAKE_BINARY_DIR}")

add_library(poclnative
        SHARED
        poclRemoteExample.cpp
        poclRemoteExample.h
        sharedUtils.h sharedUtils.c
#        poclImageProcessor.cpp
        jni_utils.cpp
#        jniWrapper.cpp
        event_logger.c event_logger.h
        yuv_compression.c yuv_compression.h
		dnn_stage.cpp dnn_stage.h
        jpeg_compression.c jpeg_compression.h
        hevc_compression.c hevc_compression.h
		eval.cpp eval.h
		eval_network_model.cpp eval_network_model.h
		testapps.c testapps.h
		poclImageProcessorTypes.h
		poclImageProcessorUtils.cpp poclImageProcessorUtils.h
		poclImageProcessorV2.cpp poclImageProcessorV2.h
		jniWrapperV2.cpp

        )


add_library(libpocl_pthread_add_i8 SHARED IMPORTED)
set_target_properties(libpocl_pthread_add_i8 PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_add_i8.so)

add_library(libpocl_pthread_opencv_onnx SHARED IMPORTED)
set_target_properties(libpocl_pthread_opencv_onnx PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_opencv_onnx.so)

add_library(libpocl_pthread_turbojpeg SHARED IMPORTED)
set_target_properties(libpocl_pthread_turbojpeg PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_turbojpeg.so)

add_library(libpocl_pthread_mediacodec_encoder SHARED IMPORTED)
set_target_properties(libpocl_pthread_mediacodec_encoder PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_BINARY_DIR}/pocl/lib/pocl/libpocl_pthread_mediacodec_encoder.so)


target_include_directories(poclnative
        PRIVATE
        ${EXTERNAL_PATH}/pocl/include
        ${CMAKE_CURRENT_BINARY_DIR}/opencv/sdk/native/jni/include
		${EXTERNAL_PATH}/tracy/public/tracy
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
		onnxruntime
		libjpeg-turbo::turbojpeg
        jnigraphics
		mediandk
        libpocl_pthread_add_i8
        libpocl_pthread_opencv_onnx
        libpocl_pthread_turbojpeg
        libpocl_pthread_mediacodec_encoder
		Tracy::TracyClient
)

add_dependencies(pocl build_opencv build_turbojpeg)
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

        opencv_imgproc

        opencv_imgcodecs

        opencv_core
        
        mediandk
		Tracy::TracyClient
        )
