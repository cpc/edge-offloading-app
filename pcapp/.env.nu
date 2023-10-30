# Environment setup for pocl.
# To load, use `overlay use .env.nu as pocl-env` in Nushell.

const ROOT_DIR = 'pcapp'

def ensure-dir [] {
    if ($env.PWD | path basename) != $ROOT_DIR {
        error make --unspanned {
            msg: $"Must run from ($ROOT_DIR) directory!"
        }
    }
}

export-env {
    ensure-dir

    for p in [
        ('../external/pocl/build/lib/CL' | path expand -s)
        '/lib/x86_64-linux-gnu'
    ] {
        if $p not-in $env.LD_LIBRARY_PATH {
            $env.LD_LIBRARY_PATH ++= [ $p ]
        }
    }

    load-env {
        CC: clang
        CXX: clang++
        OCL_ICD_VENDORS: ('../external/pocl/build/ocl-vendors' | path expand -s)
        POCL_BUILDING: 1
        POCL_DEBUG: 1 # error,warning,general,memory,llvm,events,cache,locking,refcounts,timing,hsa,tce,cuda,vulkan,proxy,all,1; 1 == error+warning+general
        POCL_DEVICES: 'basic' #pthread' # basic kernels cuda vulkan pthread ttasim almaif; space-separated list
        CUDA_VERSION: '11.7'
        CUDADIR: "/usr/local/cuda-11.7"
        CUDA_PATH: "/usr/local/cuda-11.7"
        CUDA_INSTALL_PATH: "/usr/local/cuda-11.7"
        # ASAN_OPTIONS: "halt_on_error=0"
        # OPENCV_LOG_LEVEL: "INFO"
    }
}

# Compile the project
#
# The --force flag forces CMake reconfiguration
export def compile [--force(-f)] {
    ensure-dir

    if $force {
        rm -rf build
    }

    mkdir build

    cd build
    cmake -G Ninja ..
    ninja -j (nproc)
}

# Run the example
export def run [--ld-debug] {
    ensure-dir

    if $ld_debug {
        LD_DEBUG=libs build/pcapp
    } else {
        build/pcapp
    }
}

# Compile pocl in ../external
export def build-pocl [--force(-f)] {
    ensure-dir
    cd ../external/pocl

    if $force {
        rm -rf build
    }

    mkdir build
    cd build

    let args = [
        -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-14
        -DENABLE_CUDA=NO
        -DENABLE_TESTS=0
        -DENABLE_EXAMPLES=0
        -DENABLE_SPIR=0
        -DENABLE_SPIRV=0
        -DENABLE_ICD=1
        -DCMAKE_BUILD_TYPE=Release
        -DENABLE_OPENCV_ONNX=YES
        -DONNXRUNTIME_INCLUDE_DIRS=/home/zadnik/.local/include
        -DONNXRUNTIME_LIBRARIES=/home/zadnik/.local/lib/libonnxruntime.so.1.15.0
    ]

    cmake -G Ninja $args ..
    ninja -j (nproc)
}

# Compile libyuv in ../external
export def build-libyuv [--force(-f)] {
    ensure-dir
    cd ../external/libyuv

    if $force {
        rm -rf build
    }

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE="Release" ..
    cmake --build . --config Release
}
