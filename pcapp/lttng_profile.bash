#!/usr/bin/env bash

build_dir_name="cmake-build-debug"

# set the kernel parameters for perf
# sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
# sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'

script_path="$(dirname "$(readlink -f "${BASH_SOURCE[@]}")")"

executable_path="$script_path/$build_dir_name/"
echo "$executable_path"

pushd "${executable_path}" || exit

export CUDA_INSTALL_PATH="/usr/local/cuda-11.7"
export CUDA_PATH="/usr/local/cuda-11.7"
export CUDADIR="/usr/local/cuda-11.7"
export LD_LIBRARY_PATH="/home/rabijl/Projects/aisa-demo/external/libjpeg-turbo/test-libturbojpeg/x86_64/lib/:/home/rabijl/Projects/aisa-demo/pcapp/${build_dir_name}/pocl/lib/pocl:/usr/local/onnxruntime-linux-x64-gpu-1.15.0/lib:/usr/local/cuda-11.7/lib64:/lib/x86_64-linux-gnu"
export POCL_BUILDING=1
export POCL_DEBUG="warn,err"
export POCL_DEVICES="cpu cpu remote remote"
export POCL_ENABLE_UNINT=1
export POCL_REMOTE0_PARAMETERS="localhost:10998/0"
export POCL_REMOTE1_PARAMETERS="localhost:10998/1"
export POCL_TRACING=lttng
export POCL_STARTUP_DELAY=60

./pcapp

popd || exit

