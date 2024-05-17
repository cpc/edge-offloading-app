#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <assert.h>
//#include <file_descriptor_jni.h>
#include <jni.h>
#include "poclImageProcessor.h"
#include "poclImageProcessorV2.h"
#include "codec_select.h"
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
selected_codec_t *selected_codec = NULL;
frame_stats_t *frame_stats = NULL;


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

    init_codec_select(&selected_codec, &frame_stats);

    return status;

}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_destroyPoclImageProcessorV2(JNIEnv *env,
                                                                                   jclass clazz) {

    destroy_codec_select(&selected_codec, &frame_stats);
    return destroy_pocl_image_processor_context(&ctx);
}

JNIEXPORT jfloat JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclGetLastIouV2(JNIEnv *env, jclass clazz) {

    return get_last_iou(ctx);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclSelectCodecAuto(JNIEnv *env,
                                                                           jclass clazz,
                                                                           jint do_segment,
                                                                           jint rotation) {
    select_codec_auto(do_segment, rotation, frame_stats, selected_codec);
    return 0;
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclSubmitYUVImage(JNIEnv *env, jclass clazz,
                                                                          jint device_index,
                                                                          jint do_segment,
                                                                          jint compression_type,
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

    int status;
    // TODO: set eval
    int is_eval_frame = 0;

    if (!do_algorithm) {
        select_codec_manual((device_type_enum) (device_index), do_segment,
                            (compression_t) (compression_type),
                            quality, rotation, selected_codec);
    }

    codec_config_t codec_config = get_codec_config(selected_codec);
    status = submit_image(ctx, codec_config, image_data, is_eval_frame);

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
                                                                    jlongArray metadataExchange,
                                                                    jfloat energy) {

    // todo: look into if iscopy=true works on android
    int32_t *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    // pocl returns segmentation result in uint8_t, but jbyte is int8_t
    uint8_t *segmentation_array = reinterpret_cast<uint8_t *>(env->GetByteArrayElements(
            segmentation_result, JNI_FALSE));
    int64_t *metadata_array = env->GetLongArrayElements(metadataExchange, JNI_FALSE);

    int status;
    // todo: process this metadata
    frame_metadata_t metadata;
    status = receive_image(ctx, detection_array, segmentation_array, &metadata,
                           (int32_t *) metadata_array);

    // narrow down to micro seconds
    metadata_array[1] = ((metadata.host_ts_ns.stop - metadata.host_ts_ns.start) / 1000);

    if (status == CL_SUCCESS) {
        // log statistics to codec selection data
        update_stats(&metadata, frame_stats);
    }

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result,
                                  reinterpret_cast<jbyte *>(segmentation_array), JNI_FALSE);
    env->ReleaseLongArrayElements(metadataExchange, metadata_array, JNI_FALSE);

    return status;

}

/**
 * Function to return what the quality algorithm decided on.
 * @param env
 * @param clazz
 * @return ButtonConfig that returns relevant config values
 */
JNIEXPORT jobject JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getCodecConfig(JNIEnv *env, jclass clazz) {
    jclass button_config_class = env->FindClass(
            "org/portablecl/poclaisademo/CodecConfig");
    assert(nullptr != button_config_class);

    jmethodID button_config_constructor = env->GetMethodID(button_config_class, "<init>", "(III)V");
    assert(nullptr != button_config_constructor);

    codec_config_t config = get_codec_config(selected_codec);
    int codec_id = get_codec_id(selected_codec);

    return env->NewObject(button_config_class, button_config_constructor,
                          (jint) config.compression_type, (jint) config.device_type,
                          (jint) codec_id);
}

#ifdef __cplusplus
}
#endif