#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TJDIR="${SCRIPT_DIR}/libjpeg-turbo"

# Set these variables to suit your needs
NDK_PATH=/home/zadnik/Android/Sdk/ndk/25.1.8937393
TOOLCHAIN="clang"
ANDROID_VERSION="33"

cd $TJDIR
mkdir -p libturbojpeg

# Copy header
mkdir -p libturbojpeg/include
cp turbojpeg.h libturbojpeg/include

rm -rf build
mkdir build
cd build

# # Build armv8
# echo "=== BUILDING armv8 ==="
# cmake -G"Ninja" \
#   -DANDROID_ABI=arm64-v8a \
#   -DANDROID_ARM_MODE=arm \
#   -DANDROID_PLATFORM=android-${ANDROID_VERSION} \
#   -DANDROID_TOOLCHAIN=${TOOLCHAIN} \
#   -DCMAKE_ASM_FLAGS="--target=aarch64-linux-android${ANDROID_VERSION}" \
#   -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
#   ..

# ninja -j $(nproc)

# install_dir="${TJDIR}/libturbojpeg/arm64-v8a"
# mkdir -p $install_dir
# cp libturbojpeg.a $install_dir

# cd ..
# rm -rf build
# mkdir build
# cd build

# # Build armv7
# echo "=== BUILDING armv7 ==="
# cmake -G"Ninja" \
#   -DANDROID_ABI=armeabi-v7a \
#   -DANDROID_ARM_MODE=arm \
#   -DANDROID_PLATFORM=android-${ANDROID_VERSION} \
#   -DANDROID_TOOLCHAIN=${TOOLCHAIN} \
#   -DCMAKE_ASM_FLAGS="--target=arm-linux-androideabi${ANDROID_VERSION}" \
#   -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
#   ..

# ninja -j $(nproc)

# install_dir="${TJDIR}/libturbojpeg/armeabi-v7a"
# mkdir -p $install_dir
# cp libturbojpeg.a $install_dir

# cd ..
# rm -rf build
# mkdir build
# cd build

# Build x86-64
echo "=== BUILDING x86-64 ==="
cmake -G"Ninja" ..
ninja -j $(nproc)

install_dir="${TJDIR}/libturbojpeg/x86_64"
mkdir -p $install_dir
cp libturbojpeg.a $install_dir
