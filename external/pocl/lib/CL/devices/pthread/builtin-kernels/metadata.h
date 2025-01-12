//
// Created by rabijl on 28.11.2023.
// This file contains metadata on builtin kernels that can be loaded as an so.
//

#ifndef POCL_METADATA_H
#define POCL_METADATA_H

#define NUM_PTHREAD_BUILTIN_HOST_KERNELS 21
static char *const kernel_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
        "pocl.add.i8",
        "pocl.dnn.detection.u8",
        "pocl.dnn.segmentation.postprocess.u8",
        "pocl.dnn.segmentation.reconstruct.u8",
        "pocl.dnn.eval.iou.f32",
        "pocl.compress.to.jpeg.yuv420nv21",
        "pocl.decompress.from.jpeg.rgb888",
        "pocl.encode.hevc.yuv420nv21",
        "pocl.decode.hevc.yuv420nv21",
        "pocl.configure.hevc.yuv420nv21",
        "pocl.encode.c2.android.hevc.yuv420nv21",
        "pocl.configure.c2.android.hevc.yuv420nv21",
        "pocl.init.decompress.jpeg.handle.rgb888",
        "pocl.decompress.from.jpeg.handle.rgb888",
        "pocl.destroy.decompress.jpeg.handle.rgb888",
        "pocl.dnn.ctx.init",
        "pocl.dnn.ctx.destroy",
        "pocl.dnn.ctx.detection.u8",
        "pocl.dnn.ctx.segmentation.postprocess.u8",
        "pocl.dnn.ctx.segmentation.reconstruct.u8",
        "pocl.dnn.ctx.eval.iou.f32",
};

// Make sure LD_LIBRARY_PATH is set to contain the .so files
static char *const dylib_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
        "libpocl_pthread_add_i8.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_turbojpeg.so",
        "libpocl_pthread_turbojpeg.so",
        "libpocl_pthread_mediacodec_encoder.so",
        "libpocl_pthread_ffmpeg_decoder.so",
        "libpocl_pthread_mediacodec_encoder.so",
        "libpocl_pthread_mediacodec_encoder.so",
        "libpocl_pthread_mediacodec_encoder.so",
        "libpocl_pthread_turbojpeg.so",
        "libpocl_pthread_turbojpeg.so",
        "libpocl_pthread_turbojpeg.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
        "libpocl_pthread_opencv_onnx.so",
};

static const char *const init_fn_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
        "init_pocl_add_i8",
        "",
        "",
        "",
        "",
        "init_turbo_jpeg",
        "init_turbo_jpeg",
        "init_mediacodec_encoder",
        "",
        "",
        "init_c2_android_hevc_encoder",
        "",
        "",
        "",
        "",
        "init_onnx_ctx",
        "init_onnx_ctx",
        "init_onnx_ctx",
        "init_onnx_ctx",
        "init_onnx_ctx",
        "init_onnx_ctx",
};

static const char *const free_fn_names[NUM_PTHREAD_BUILTIN_HOST_KERNELS] = {
        "free_pocl_add_i8",
        "",
        "",
        "",
        "",
        "destroy_turbo_jpeg",
        "destroy_turbo_jpeg",
        "destroy_mediacodec_encoder",
        "",
        "",
        "destroy_c2_android_hevc_encoder",
        "",
        "",
        "",
        "",
        "finish_onnx_ctx",
        "finish_onnx_ctx",
        "finish_onnx_ctx",
        "finish_onnx_ctx",
        "finish_onnx_ctx",
        "finish_onnx_ctx",
};

#endif //POCL_METADATA_H
