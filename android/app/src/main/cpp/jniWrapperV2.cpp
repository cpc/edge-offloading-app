#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <assert.h>
//#include <file_descriptor_jni.h>
#include <jni.h>
#include "poclImageProcessorV2.h"
#include "codec_select.h"
#include "eval.h"
#include "sharedUtils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "jni_utils.h"
#include "platform.h"
#include "RawImageReader.hpp"
#include "codec_select_wrapper.h"

//
// Created by rabijl on 5.3.2024.
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif


pocl_image_processor_context *ctx = NULL;
codec_select_state_t *state = NULL;
RawImageReader *rawReader = nullptr;

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_initPoclImageProcessorV2(JNIEnv *env,
                                                                                jclass clazz,
                                                                                jint config_flags,
                                                                                jobject j_asset_manager,
                                                                                jint width,
                                                                                jint height,
                                                                                jint fd,
                                                                                jint max_lanes,
                                                                                jint do_algorithm,
                                                                                jint runtime_eval,
                                                                                jint lock_codec,
                                                                                jstring service_name,
                                                                                jint calibrate_fd) {

    bool file_there = put_asset_in_local_storage(env, j_asset_manager, "yolov8n-seg.onnx");
    assert(file_there);

    size_t src_sizes[2];
    char *codec_sources[2] = {
            read_file(env, j_asset_manager, "kernels/copy.cl", &(src_sizes)[0]),
            read_file(env, j_asset_manager, "kernels/compress_seg.cl", &(src_sizes)[1])
    };

    char *_service_name = NULL;
    if (service_name != NULL)
        _service_name = (char *) env->GetStringUTFChars(service_name, 0);

    jint status = create_pocl_image_processor_context(&ctx, max_lanes, width, height, config_flags,
                                                      (const char **) codec_sources,
                                                      (const size_t *) src_sizes, fd, runtime_eval,
                                                      _service_name);

    free(codec_sources[0]);
    free(codec_sources[1]);

    bool has_video_input = calibrate_fd >= 0;

    // TODO: don't init codec select if not using it
    init_codec_select(config_flags, fd, do_algorithm, lock_codec, has_video_input, &state);

    if (service_name != NULL)
        env->ReleaseStringUTFChars(service_name, _service_name);

    if (has_video_input) {
        rawReader = new RawImageReader(width, height, calibrate_fd);
    }

    return status;

}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_destroyPoclImageProcessorV2(JNIEnv *env,
                                                                                   jclass clazz) {

    destroy_codec_select(&state);
    delete rawReader;
    return destroy_pocl_image_processor_context(&ctx);
}

JNIEXPORT jfloat JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclGetLastIouV2(JNIEnv *env, jclass clazz) {

    if (NULL == ctx) {
        LOGW("last iou was called on ctx that was not initialized");
        return -1.0f;
    }
    return get_last_iou(ctx);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclSelectCodecAuto(JNIEnv *env,
                                                                           jclass clazz) {

    assert(NULL != state);
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

    assert(NULL != ctx);

    image_data_t image_data;
    bool is_last_playback_frame = false;
    if (nullptr != rawReader) {
        const int NUM_LOCAL_PLAYBACK_FRAMES = 10;  // run local device for only a few frames
        const int NUM_REMOTE_PLAYBACK_FRAMES = 75; // set a very large number to play all

        const int NUM_PLAYBACK_FRAMES =
                device_index == LOCAL_DEVICE ? NUM_LOCAL_PLAYBACK_FRAMES
                                                         : NUM_REMOTE_PLAYBACK_FRAMES;

        rotation = 0;

        // substitute the frame with one from a file
        is_last_playback_frame = rawReader->readImage(&image_data);
        is_last_playback_frame |= rawReader->getCurrentFrameNum() >= NUM_PLAYBACK_FRAMES;

        if (is_last_playback_frame) {
            rawReader->reset();
            signal_last_frame(state);
        }
    } else {
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
    }

    int status = codec_select_submit_image(state, ctx, device_index, do_segment, compression_type,
                                           quality, rotation, do_algorithm, &image_data);

    if (state->enable_profiling) {
        log_frame_int(state->fd, ctx->frame_index_head - 1, "frame", "is_last_frame",
                      is_last_playback_frame ? 1 : 0);
    }

    return status;
}


JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_dequeue_1spot(JNIEnv *env, jclass clazz,
                                                                     jint timeout, jint dev_type) {
    assert(NULL != ctx);
    return dequeue_spot(ctx, timeout, (device_type_enum) dev_type);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_waitImageAvailable(JNIEnv *env, jclass clazz,
                                                                          jint timeout) {
    assert(NULL != ctx);
    return wait_image_available(ctx, timeout);
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_receiveImage(JNIEnv *env, jclass clazz,
                                                                    jintArray detection_result,
                                                                    jbyteArray segmentation_result,
                                                                    jlongArray metadataExchange,
                                                                    jfloat energy) {

    assert(NULL != ctx);

    // todo: look into if iscopy=true works on android
    int32_t *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    // pocl returns segmentation result in uint8_t, but jbyte is int8_t
    uint8_t *segmentation_array = reinterpret_cast<uint8_t *>(env->GetByteArrayElements(
            segmentation_result, JNI_FALSE));
    int64_t *metadata_array = env->GetLongArrayElements(metadataExchange, JNI_FALSE);

    int status;
    status = codec_select_receive_image(state, ctx, detection_array, segmentation_array, metadata_array);

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

    assert(NULL != state);

    jclass button_config_class = env->FindClass("org/portablecl/poclaisademo/CodecConfig");
    assert(nullptr != button_config_class);

    jmethodID button_config_constructor = env->GetMethodID(button_config_class, "<init>",
                                                           "(IIIII)V");
    assert(nullptr != button_config_constructor);

    codec_params_t config = get_codec_params(state);

    int is_calibrating = (int) (state->is_calibrating);
    int codec_id = get_codec_id(state);
    int codec_sort_id = get_codec_sort_id(state);

    return env->NewObject(button_config_class, button_config_constructor,
                          (jint) config.compression_type, (jint) config.device_type,
                          (jint) codec_id, (jint) codec_sort_id, (jint) is_calibrating);
}

// For reference, if we want to measure ping using fillbuffer again:
//JNIEXPORT jobject JNICALL
//Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getStats(JNIEnv *env, jclass clazz) {
//
//    assert(NULL != state);
//
//    jclass stats_class = env->FindClass("org/portablecl/poclaisademo/MainActivity$Stats");
//    assert(nullptr != stats_class);
//
//    jmethodID stats_constructor = env->GetMethodID(stats_class, "<init>", "(FF)V");
//    assert(nullptr != stats_constructor);
//
//    float ping_ms = 0.0f;
//    float ping_ms_avg = 0.0f;
//
//    if (state != NULL) {
//        pthread_mutex_lock(&state->lock);
//        ping_ms = state->stats.ping_ms;
//        ping_ms_avg = state->stats.ping_ms_avg;
//        pthread_mutex_unlock(&state->lock);
//    }
//
//    return env->NewObject(stats_class, stats_constructor, (jfloat) ping_ms, (jfloat) ping_ms_avg);
//}

JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_pushExternalPow(JNIEnv *env, jclass clazz,
                                                                       jlong timestamp, jint amp,
                                                                       jint volt) {
    if (state != NULL) {
        push_external_pow(state, timestamp, amp, volt);
    }
}

JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_pushExternalPing(JNIEnv *env, jclass clazz,
                                                                        jlong timestamp,
                                                                        jfloat ping_ms) {
    if (state != NULL) {
        push_external_ping(state, timestamp, ping_ms);
    }
}

#ifdef __cplusplus
}
#endif