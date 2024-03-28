#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <assert.h>
//#include <file_descriptor_jni.h>
#include <jni.h>
#include "poclImageProcessor.h"
#include "poclImageProcessorV2.h"
#include "eval.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "jni_utils.h"

//
// Created by rabijl on 5.3.2024.
//

#ifdef __cplusplus
extern "C" {
#endif

pocl_image_processor_context *ctx = NULL;


JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_initPoclImageProcessorV2(JNIEnv *env,
                                                                                jclass clazz,
                                                                                jint config_flags,
                                                                                jobject j_asset_manager,
                                                                                jint width,
                                                                                jint height,
                                                                                jint fd,
                                                                                jint max_lanes) {

    bool file_there = put_asset_in_local_storage(env, j_asset_manager, "yolov8n-seg.onnx");
    assert(file_there);

    size_t src_size;
    char *codec_sources = read_file(env, j_asset_manager, "kernels/copy.cl", &src_size);
    jint status = create_pocl_image_processor_context(&ctx, max_lanes, width,
                                                      height, config_flags, codec_sources, src_size,
                                                      fd);

    free(codec_sources);
    return status;

}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_destroyPoclImageProcessorV2(JNIEnv *env,
                                                                                   jclass clazz) {

    return destroy_pocl_image_processor_context(&ctx);
}

JNIEXPORT jfloat JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclGetLastIouV2(JNIEnv *env, jclass clazz) {

    return get_last_iou(ctx);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclSubmitYUVImage(JNIEnv *env, jclass clazz,
                                                                          jint device_index,
                                                                          jint do_segment,
                                                                          jint do_compression,
                                                                          jint quality,
                                                                          jint rotation,
                                                                          jint do_algorithm,
                                                                          jobject plane0,
                                                                          jint row_stride0,
                                                                          jint pixel_stride0,
                                                                          jobject plane1,
                                                                          jint row_stride1,
                                                                          jint pixel_stride1,
                                                                          jobject plane2,
                                                                          jint row_stride2,
                                                                          jint pixel_stride2,
                                                                          jlong image_timestamp) {


    // TODO: implement poclSubmitYUVImage()
    image_data_t image_data;
    image_data.type = YUV_DATA_T;
    image_data.data.yuv.planes[0] = (uint8_t *) env->GetDirectBufferAddress(plane0);
    image_data.data.yuv.planes[1] = (uint8_t *) env->GetDirectBufferAddress(plane1);
    image_data.data.yuv.planes[2] = (uint8_t *) env->GetDirectBufferAddress(plane2);
    image_data.data.yuv.pixel_strides[0] = pixel_stride0;
    image_data.data.yuv.pixel_strides[1] = pixel_stride1;
    image_data.data.yuv.pixel_strides[2] = pixel_stride2;
    image_data.data.yuv.row_strides[0] = row_stride0;
    image_data.data.yuv.row_strides[1] = row_stride1;
    image_data.data.yuv.row_strides[2] = row_stride2;

    codec_config_t config;
    config.rotation = rotation;
    config.compression_type = (compression_t) do_compression;
    config.device_type = (device_type_enum) device_index;
    config.do_segment = do_segment;
    if (HEVC_COMPRESSION == do_compression ||
        SOFTWARE_HEVC_COMPRESSION == do_compression) {
        const int framerate = 5;
        config.config.hevc.framerate = framerate;
        config.config.hevc.i_frame_interval = 2;
        // heuristical map of the bitrate to quality parameter
        // (640 * 480 * (3 / 2) * 8 / (1 / framerate)) * (quality / 100)
        // equation can be simplied to equation below
        config.config.hevc.bitrate = 36864 * framerate * quality;
    } else if (JPEG_COMPRESSION == do_compression) {
        config.config.jpeg.quality = quality;
    }

    int status;
    // TODO: set eval and use do algorithm
    int is_eval_frame = 0;
    status = submit_image(ctx, config, image_data, is_eval_frame);

    return status;
}


JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_dequeue_1spot(JNIEnv *env, jclass clazz,
                                                                     jint timeout,
                                                                     jint dev_type) {

    return dequeue_spot(ctx, timeout, (device_type_enum) dev_type);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_waitImageAvailable(JNIEnv *env, jclass clazz,
                                                                          jint timeout) {
    return wait_image_available(ctx, timeout);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_receiveImage(JNIEnv *env, jclass clazz,
                                                                    jintArray detection_result,
                                                                    jbyteArray segmentation_result,
                                                                    jintArray do_segment_array,
                                                                    jfloat energy) {

    // todo: look into if iscopy=true works on android
    int32_t *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    // pocl returns segmentation result in uint8_t, but jbyte is int8_t
    uint8_t *segmentation_array = reinterpret_cast<uint8_t *>(env->GetByteArrayElements(
            segmentation_result, JNI_FALSE));
    int32_t *do_segment = env->GetIntArrayElements(do_segment_array, JNI_FALSE);

    int status;
    // todo: process this metadata
    eval_metadata_t metadata;
    status = receive_image(ctx, detection_array, segmentation_array, &metadata, do_segment);

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result,
                                  reinterpret_cast<jbyte *>(segmentation_array), JNI_FALSE);
    env->ReleaseIntArrayElements(do_segment_array, do_segment, JNI_FALSE);

    return status;

}

///**
// * Function to return what the quality algorithm decided on.
// * @param env
// * @param clazz
// * @return ButtonConfig that returns relevant config values
// */
//JNIEXPORT jobject JNICALL
//Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getButtonConfig(JNIEnv *env, jclass clazz) {
//
//    jclass button_config_class = env->FindClass(
//            "org/portablecl/poclaisademo/MainActivity$ButtonConfig");
//    assert(nullptr != button_config_class);
//
//    jmethodID button_config_constructor = env->GetMethodID(button_config_class, "<init>", "(III)V");
//    assert(nullptr != button_config_constructor);
//
//    return env->NewObject(button_config_class, button_config_constructor,
//                          (jint) config.compression_type, (jint) config.device_type,
//                          (jint) CUR_CODEC_ID);
//}


#ifdef __cplusplus
}
#endif