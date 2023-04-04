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
| JDK                           | 11                      | file > project structure > SDK Location > gradle settings > gradle JDK |
| Minimum SDK version           | 24                      | android/app/build.gradle > minSdk                                      |
| Targeted SDK version          | 33                      | android/app/build.gradle > targetSdk                                   |
| Gradle                        | 7.5                     | file > project structure > project > gradle version                    |
| Android Gradle plugin version | 7.4.2                   | file > project structure > project > android gradle plugin version     |
| CMake                         | 3.22.1                  | android/app/src/main/cpp/CMakeLists.txt > cmake_minimum_required       |
| NDK                           | 23.1.7779620            | (android layout in project tab) app/cpp/inlcudes/NDK                   |
| C++                           | 17                      | android/app/build.gradle > cppFlags '-std=c++\<version\>'              |


## Installation


### Android Studio

1. install Android Studio
2. install the android sdk from `Tools > SDK Manager > SDK Platforms`
3. install the android Native Development Kit (NDK) from `Tools > SDK Manager > SDK Tools`
4. install CMake from `Tools > SDK Manager > SDK Tools`



## Usage

### TO BE ADDED

## Development notes

* Use CPP files with `extern "C"` as the JNIEnv is different between C and CPP and the editer will have an issue with it otherwise.

## License

### TO BE ADDED
