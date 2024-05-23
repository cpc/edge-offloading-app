
the prebuilt binaries on github are linked against older cuda versions.
here's how to build it from source:

1. clone git@github.com:microsoft/onnxruntime.git and checkout the `v1.7.3` tag
2. export the cuda variables:
```
export CUDA_HOME="/usr/lib/x86_64-linux-gnu/"
export CUDNN_HOME="/usr/lib/x86_64-linux-gnu/" 
export CUDACXX="$(which nvcc)" 
```
this should work if you installed `nvidia-cuda-toolkit` and `nvidia-cudnn` 
from the ubuntu repos.

3. run the build script
```
./build.sh --config=RelWithDebInfo --parallel --build_shared_lib --skip_tests --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}" --cuda_version=12.0
```
4. install the onnxruntime
```
cd build/Linux/RelWithDebInfo
sudo make install
```
this will install onnxruntime into `/usr/local`, hence the requirement to run it as `sudo`.

