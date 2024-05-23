#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <assert.h>
//#include <file_descriptor_jni.h>
#include <jni.h>
#include "poclImageProcessor.h"
#include "poclImageProcessorV2.h"
#include "codec_select.h"
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
codec_select_state_t *state = NULL;

/**
 * Set codec config from the codec selection state and input parameters from the UI (rotation, do_segment)
 */
static void set_codec_config(int rotation, int do_segment, codec_config_t *codec_config) {
    const codec_params_t params = get_codec_params(state);
    const int codec_id = get_codec_id(state);
    codec_config->compression_type = params.compression_type;
    codec_config->device_type = params.device_type;
    codec_config->rotation = rotation;
    codec_config->do_segment = do_segment;
    codec_config->id = codec_id;
    if (codec_config->compression_type == JPEG_COMPRESSION) {
        codec_config->config.jpeg = params.config.jpeg;
    } else if (codec_config->compression_type == HEVC_COMPRESSION ||
               codec_config->compression_type == SOFTWARE_HEVC_COMPRESSION) {
        codec_config->config.hevc = params.config.hevc;
    }
}

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

    init_codec_select(&state);

    return status;

}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_destroyPoclImageProcessorV2(JNIEnv *env,
                                                                                   jclass clazz) {

    destroy_codec_select(&state);
    return destroy_pocl_image_processor_context(&ctx);
}

JNIEXPORT jfloat JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclGetLastIouV2(JNIEnv *env, jclass clazz) {

    return get_last_iou(ctx);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclSelectCodecAuto(JNIEnv *env,
                                                                           jclass clazz) {
    select_codec_auto(state);
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

    codec_config_t codec_config;

    if (do_algorithm) {
        set_codec_config(rotation, do_segment, &codec_config);
    } else {
        select_codec_manual((device_type_enum) (device_index), do_segment,
                            (compression_t) (compression_type),
                            quality, rotation, &codec_config);
    }


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
        update_stats(&metadata, state);
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

    codec_params_t config = get_codec_params(state);
    int codec_id = get_codec_id(state);

    return env->NewObject(button_config_class, button_config_constructor,
                          (jint) config.compression_type, (jint) config.device_type,
                          (jint) codec_id);
}

JNIEXPORT jobject JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getStats(JNIEnv *env, jclass clazz) {
    jclass stats_class = env->FindClass("org/portablecl/poclaisademo/MainActivity$Stats");
    assert(nullptr != stats_class);

    jmethodID stats_constructor = env->GetMethodID(stats_class, "<init>", "(FF)V");
    assert(nullptr != stats_constructor);

    float ping_ms = 0.0f;
    float ping_ms_avg = 0.0f;

    if (state != NULL) {
        pthread_mutex_lock(&state->lock);
        ping_ms = state->stats.ping_ms;
        ping_ms_avg = state->stats.ping_ms_avg;
        pthread_mutex_unlock(&state->lock);
    }

    return env->NewObject(stats_class, stats_constructor, (jfloat) ping_ms, (jfloat) ping_ms_avg);
}

#ifdef __cplusplus
}
#endif