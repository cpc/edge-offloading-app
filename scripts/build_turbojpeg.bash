#!/usr/bin/env bash

# build turbo jpeg from source and install it to $TJDIR/libturbojpeg/<arch> directory

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TJDIR="${SCRIPT_DIR}/../external/libjpeg-turbo"
INSTALL_DIR_PREFIX="${TJDIR}/libturbojpeg"

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

cd "$SCRIPT_DIR" || exit
