#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

# To add a new BiK as a lib to pthread, add the relevant data to the
# builtin-kernels/metadata.h

# these are needed in order to have POCL messages working in the BiKs
set(LOGGING_SOURCES ../../pocl_debug.h ../../pocl_debug.c ../../pocl_threads.c ../../pocl_threads.h ../../pocl_timing.c ../../pocl_timing.h)

if(ANDROID)
    find_library(ANDROID_LOG_LIB log REQUIRED)
else(ANDROID)
    set(ANDROID_LOG_LIB "")
endif(ANDROID)

function(add_pocl_host_builtin_library name)
    add_library(${name} SHARED ${ARGN})
    # make the logging files private
    target_sources(${name} PRIVATE ${LOGGING_SOURCES})
    harden("${name}")
    set_target_properties(${name} PROPERTIES PREFIX "lib" SUFFIX ".so")
    if(TRACY_ENABLE)
        target_link_libraries(${name} PUBLIC Tracy::TracyClient)
    endif()
    target_link_libraries(${name} PUBLIC ${ANDROID_LOG_LIB})
    install(TARGETS ${name} LIBRARY DESTINATION "${POCL_INSTALL_PRIVATE_LIBDIR}" COMPONENT "lib")
endfunction()

### Vector addition test BiK
set(VECADD_BIK_LIB pocl_pthread_add_i8)
add_pocl_host_builtin_library(${VECADD_BIK_LIB}
        builtin-kernels/pocl_add_i8.cpp
        builtin-kernels/pocl_add_i8.h)
set_target_properties(${VECADD_BIK_LIB} PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET ${VECADD_BIK_LIB} PROPERTY CXX_STANDARD 14)

### ONNX inference and related kernels
# Requires running cmake with these options (example values):
# -DORT_INSTALL_DIR=/usr/local/onnxruntime-linux-x64-gpu-1.15.0/include
OPTION(ENABLE_OPENCV_ONNX "enable pthread device ONNX inference" OFF)
MESSAGE("-- pthread device OpenCV: ${ENABLE_OPENCV_ONNX}")
if (ENABLE_OPENCV_ONNX)

    if(NOT ANDROID)
        find_package(CUDA 11 REQUIRED)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    # OpenCV
    find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs)
    message("-- OpenCV: Include dirs ${OpenCV_INCLUDE_DIRS}")
    message("-- OpenCV: LINK LIBRARIES ${OpenCV_LIBS}")

    if (ANDROID)
        if (NOT ORT_INSTALL_DIR)
            message(FATAL_ERROR "OpenCV w. ONNX requires \"ORT_INSTALL_DIR\", but not provided")
        endif ()

        #assumes that ORT_INSTALL_DIR points to onnxruntime-android
        # find_library does not seem to work on android, so hardcode the path ourselves
        set(onnxruntime_LIBRARY "${ORT_INSTALL_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so")

        # create a library from the onnxruntime
        # since it doesn't contain a cmakefile.
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${onnxruntime_LIBRARY})
        set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ORT_INSTALL_DIR}/headers")

    elseif (ORT_INSTALL_DIR)
        # find the lib file
        find_library(onnxruntime_LIBRARY onnxruntime
                PATHS "${ORT_INSTALL_DIR}/lib")
        # create a library from the onnxruntime
        # since it doesn't contain a cmakefile.
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${onnxruntime_LIBRARY})
        set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ORT_INSTALL_DIR}/include")
    else ()
        # try finding the package with find_package
        find_package(onnxruntime REQUIRED)
    endif ()

    add_pocl_host_builtin_library(pocl_pthread_opencv_onnx
            builtin-kernels/opencv_onnx.cpp
            builtin-kernels/opencv_onnx.h)
    set_target_properties(pocl_pthread_opencv_onnx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(pocl_pthread_opencv_onnx PROPERTIES CXX_STANDARD 14)

    target_link_libraries(pocl_pthread_opencv_onnx PUBLIC
                        opencv_core
                        opencv_imgproc
                        opencv_dnn
                        opencv_imgcodecs
                        onnxruntime)

endif (ENABLE_OPENCV_ONNX)

### TurboJPEG compression / decompression
OPTION(ENABLE_TURBO_JPEG_COMPRESSION "use turbo jpeg to compress images on pthread devices" OFF)
OPTION(ENABLE_TURBO_JPEG_DECOMPRESSION "use turbo jpeg to decompress images on pthread devices" OFF)
if(ENABLE_TURBO_JPEG_COMPRESSION OR ENABLE_TURBO_JPEG_DECOMPRESSION)
    MESSAGE("-- pthread device turbo jpeg compression: ${ENABLE_TURBO_JPEG_COMPRESSION}")
    MESSAGE("-- pthread device turbo jpeg decompression: ${ENABLE_TURBO_JPEG_DECOMPRESSION}")

    add_pocl_host_builtin_library(pocl_pthread_turbojpeg
            builtin-kernels/turbo_jpeg.c
            builtin-kernels/turbo_jpeg.h)

    set_target_properties(pocl_pthread_turbojpeg PROPERTIES LINKER_LANGUAGE CXX)

    # if you install turbo jpeg in a non standard location,
    # set -Dlibjpeg-turbo_DIR to the cmake file of turbo jpeg
    find_package(libjpeg-turbo REQUIRED)
    target_link_libraries(pocl_pthread_turbojpeg PRIVATE libjpeg-turbo::turbojpeg)

endif() # ENABLE_TURBO_JPEG_COMPRESSION OR ENABLE_TURBO_JPEG_DECOMPRESSION

### ffmpeg/libav video decoder
OPTION(ENABLE_FFMPEG_DECODE "enable ffmpeg decoding options" OFF)
if(ENABLE_FFMPEG_DECODE)
    # apparently ffmpeg doesn't support find_package, so we need pkgconfig for this instead
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
            libavcodec
            libavutil
    )

    set(FFMPEG_BIK_LIB pocl_pthread_ffmpeg_decoder)

    add_pocl_host_builtin_library(${FFMPEG_BIK_LIB}
            builtin-kernels/pthread_ffmpeg.c
            builtin-kernels/pthread_ffmpeg.h)

    include_directories(SYSTEM /usr/include/x86_64-linux-gnu/)
    target_link_libraries(${FFMPEG_BIK_LIB} PUBLIC PkgConfig::LIBAV)

endif()

### Android mediacodec video encoder
OPTION(ENABLE_ANDROID_MEDIACODEC "enable the use of android mediacodec functions" OFF)
if(ENABLE_ANDROID_MEDIACODEC)
    set(MEDIACODEC_BIK_LIB pocl_pthread_mediacodec_encoder)

    add_pocl_host_builtin_library(${MEDIACODEC_BIK_LIB}
            builtin-kernels/android_media_codec.c
            builtin-kernels/android_media_codec.h)

    # enable this if you want to have function highlighting while developing on x86
#    target_include_directories(${MEDIACODEC_BIK_LIB} PRIVATE ~/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include)

    target_link_libraries(${MEDIACODEC_BIK_LIB} PUBLIC mediandk)

endif()

### Other pthread stuff
if (MSVC)
    set_source_files_properties(pocl-pthread.h pthread.c pthread_scheduler.c PROPERTIES LANGUAGE CXX)
endif (MSVC)

set(SOURCES pocl-pthread.h pthread.c pthread_scheduler.c builtin-kernels/metadata.h)
if (APPLE)
    list(APPEND SOURCES pthread_barrier.c)
endif (APPLE)
add_pocl_device_library(pocl-devices-pthread ${SOURCES})

if (ENABLE_LOADABLE_DRIVERS)
    target_link_libraries(pocl-devices-pthread PRIVATE pocl-devices-basic ${PTHREAD_LIBRARY})
endif ()

if(ENABLE_HOST_CPU_DEVICES_OPENMP)
    target_link_libraries(pocl-devices-pthread PRIVATE OpenMP::OpenMP_C)
endif()