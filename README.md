# AISA demo

This repo is the basis for the PoCL AISA demo and contains code to run an android application with OpenCL.


## Cloning Repo

clone the repo with its submodules with the following command:

```
git clone --recursive-submodules git@gitlab.tuni.fi:cs/cpc/aisa-demo.git
```

or initialize submodules after cloning with:

```
git submodule update --init --recursive
```

## Relevant Versions

Android constantly updates packages and libraries that can break with newer versions. Below is a list of relevent versions currently used:


| **What**                      | **Version**             | **where to check**                                                     |
|-------------------------------|-------------------------|------------------------------------------------------------------------|
| Android Studio                | Electric Eel (2022.1.1) | file > settings > appearance & behavior > system settings > updates    |
| JDK                           | 17                      | file > project structure > SDK Location > gradle settings > gradle JDK |
| Minimum SDK version           | 24                      | android/app/build.gradle > minSdk                                      |
| Targeted SDK version          | 33                      | android/app/build.gradle > targetSdk                                   |
| Gradle                        | 8.0                     | file > project structure > project > gradle version                    |
| Android Gradle plugin version | 8.0.2                   | file > project structure > project > android gradle plugin version     |
| CMake                         | 3.22.1                  | android/app/src/main/cpp/CMakeLists.txt > cmake_minimum_required       |
| NDK                           | 23.1.7779620            | (android layout in project tab) app/cpp/inlcudes/NDK                   |
| C++                           | 17                      | android/app/build.gradle > cppFlags '-std=c++\<version\>'              |


## Installation


## Check that phone exposes the OpenCL library
Some phones provide a OpenCL library that can be used in C. This library needs to be whitelisted by the vendor. This can be checked like so:
1. adb into the phone
2. run:
    ```
    cat /vendor/etc/public.libraries.txt  
    ```
    and check that `libOpenCL.so` is there


### Android Studio

1. install Android Studio
2. install the android sdk from `Tools > SDK Manager > SDK Platforms`
3. install the android Native Development Kit (NDK) from `Tools > SDK Manager > SDK Tools`
4. install CMake from `Tools > SDK Manager > SDK Tools`

### OpenCV

follow the instructions in the `scripts/build_opencv.bash` script

## Environment variables

In order to build and run OpenCV and BiK, it is recommended to have the following variables set:

`CUDA_VERSION='11.7'` \
`CUDADIR="/usr/local/cuda-11.7"` \
`CUDA_PATH="/usr/local/cuda-11.7"` \
`CUDA_INSTALL_PATH="/usr/local/cuda-11.7"` \
`LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/lib/x86_64-linux-gnu` 

Of these `LD_LIBRARY_PATH` is the most important.

## Development

While developing, there are a number of logging and debug variables that can be set. For `poclImageProcessor.ccp`, "PRINT_PROFILE_TIME" can be defined.
On the java side, the `DevelopmentVariables.java` contains variables that make the program more verbose


## Common trouble shootingshooting

### Can not find platform -1001 error

This is can happen after installing cuda-11-7. The ICD loader grabs the libopencl.so library that cuda-11.7 
provides instead of the one from `ocl-icd-libopencl1`. the easiest way to fix this is by moving the 
`libOpenCL.so*` files in `/usr/local/cuda-11/lib64` into a new directory so that they won't be found.

### clinfo can not find any platforms when using OCL_ICD_VENDORS

This is the same issue as the previous troubleshooting case.

### CL Error -1 (CL_DEVICE_NOT_FOUND) while setting `POCL_DEVICES="basic"`

Add `POCL_BUILDING=1` as a parameter as well.

### OpenCV error while loading model

If you get an error like this:
```
[ INFO:0@0.110] global onnx_importer.cpp:595 populateNet DNN/ONNX: loading ONNX v7 model produced by 'pytorch':2.0.0. Number of nodes = 297, inputs = 1, outputs = 2 OpenCV(4.7.0) Error: Assertion failed (blob_0.size == blob_1.size) in parseBias, file ./modules/dnn/src/onnx/onnx_importer.cpp, line 1067
```

make sure you built OpenCV and PoCL while having the `LD_LIBRARY_PATH` environment variable set to contain:

```
/usr/local/cuda-11.7/lib64:/lib/x86_64-linux-gnu
```
in that order

## Usage

### TO BE ADDED

## Development notes

* Use CPP files with `extern "C"` as the JNIEnv is different between C and CPP and the editer will have an issue with it otherwise.

## License

Apache 2.0 license
