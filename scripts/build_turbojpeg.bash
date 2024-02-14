#!/usr/bin/env bash

# build turbo jpeg from source and install it to $TJDIR/libturbojpeg/<arch> directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TJDIR="${SCRIPT_DIR}/../external/libjpeg-turbo"
INSTALL_DIR_PREFIX="${TJDIR}/libturbojpeg"

# Set these variables to suit your needs (relevant only for Android
NDK_PATH="${HOME}/Android/Sdk/ndk/25.1.8937393"
TOOLCHAIN="clang"
ANDROID_VERSION="33"

cd "$TJDIR" || exit

# Build x86-64

BUILD_DIR="build-x86-64"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR || exit

echo "=== BUILDING x86-64 ==="
cmake -G"Ninja" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR_PREFIX}/x86-64" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  ..

ninja
ninja install

cd "$TJDIR" || exit

# Build armv8

BUILD_DIR="build-armv8"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR || exit

echo "=== BUILDING armv8 ==="
cmake -G"Ninja" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_ARM_MODE=arm \
  -DANDROID_PLATFORM=android-${ANDROID_VERSION} \
  -DANDROID_TOOLCHAIN=${TOOLCHAIN} \
  -DCMAKE_ASM_FLAGS="--target=aarch64-linux-android${ANDROID_VERSION}" \
  -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR_PREFIX}/armv8" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  ..

ninja
ninja install
