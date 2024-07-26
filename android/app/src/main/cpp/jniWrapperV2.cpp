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
#include "sharedUtils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "jni_utils.h"
#include "platform.h"
#include "RawImageReader.hpp"

//
// Created by rabijl on 5.3.2024.
//

#ifdef __cplusplus
extern "C" {
#endif

pocl_image_processor_context *ctx = NULL;
codec_select_state_t *state = NULL;
RawImageReader *rawReader = nullptr;

/**
 * Set codec config from the codec selection state and input parameters from the UI (rotation, do_segment)
 */
static void get_codec_config(int rotation, int do_segment, codec_config_t *codec_config) {
    assert(NULL != state);
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
                                                                                jint max_lanes,
                                                                                jint do_algorithm,
                                                                                jint runtime_eval,
                                                                                jint lock_codec,
                                                                                jstring service_name,
                                                                                jint calibrate_fd) {

    bool file_there = put_asset_in_local_storage(env, j_asset_manager, "yolov8n-seg.onnx");
    assert(file_there);

    size_t src_size;
    char *codec_sources = read_file(env, j_asset_manager, "kernels/copy.cl", &src_size);

    char *_service_name = NULL;
    if (service_name != NULL)
        _service_name = (char *) env->GetStringUTFChars(service_name, 0);

    jint status = create_pocl_image_processor_context(&ctx, max_lanes, width, height, config_flags,
                                                      codec_sources, src_size, fd, runtime_eval,
                                                      _service_name);

    free(codec_sources);

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

    bool use_playback_input = nullptr != rawReader;

    if (use_playback_input) {
        rotation = 0;
    }

    // read the current config from the codec selection state
    codec_config_t codec_config;
    // TODO: put in else statement of do_algorithm
    get_codec_config(rotation, do_segment, &codec_config);

    image_data_t image_data;
    bool is_last_playback_frame = false;
    if (use_playback_input) {
        const int NUM_LOCAL_PLAYBACK_FRAMES = 10;  // run local device for only a few frames
        const int NUM_REMOTE_PLAYBACK_FRAMES = 75; // set a very large number to play all

        const int NUM_PLAYBACK_FRAMES =
                codec_config.device_type == LOCAL_DEVICE ? NUM_LOCAL_PLAYBACK_FRAMES
                                                         : NUM_REMOTE_PLAYBACK_FRAMES;

        // substitute the frame with one from a file
        is_last_playback_frame = rawReader->readImage(&image_data);
        is_last_playback_frame |= rawReader->getCurrentFrameNum() >= NUM_PLAYBACK_FRAMES;

        if (is_last_playback_frame) {
            rawReader->reset();
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

    if (is_last_playback_frame) {
        signal_last_frame(state);
    }

    int status;
    int is_eval_frame = 0;
    int frame_index = -1;

    if (!do_algorithm) {
        // override the codec config with whatever the user set in the UI
        select_codec_manual((device_type_enum) (device_index), do_segment,
                            (compression_t) (compression_type), quality, rotation, &codec_config);
    }

    // check if we need to submit image to the eval pipeline
    if (ctx->enable_eval || state->is_calibrating) {
        status = check_eval(ctx->eval_ctx, state, codec_config, &is_eval_frame);
        CHECK_AND_RETURN(status, "could not check and submit eval frame");
    }

    // submit the image for the actual encoding (needs to be submitted *before* the eval frame)
    bool codec_selected = drain_codec_selected(state);
    int64_t latency_offset_ms = get_latency_offset_ms(state);
    status = submit_image(ctx, codec_config, image_data, is_eval_frame, codec_selected,
                          latency_offset_ms, &frame_index);
    CHECK_AND_RETURN(status, "could not submit frame");

    if (state->enable_profiling) {
        log_frame_int(state->fd, frame_index, "frame", "is_last_frame",
                      is_last_playback_frame ? 1 : 0);
    }

    if (is_eval_frame) {
        // submit the eval frame if appropriate
        status = run_eval(ctx->eval_ctx, state, codec_config, frame_index, image_data);
        CHECK_AND_RETURN(status, "could not submit eval frame");
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
    frame_metadata_t metadata;
//    lane_state_t new_state;
    status = receive_image(ctx, detection_array, segmentation_array, &metadata,
                           (int32_t *) metadata_array, state->collected_events);

    if (status == CL_SUCCESS) {
        // log statistics to codec selection data
        update_stats(&metadata, ctx->eval_ctx, state);
    }

    // not strictly necessary, just easier to debug without having old values lying around
    reset_collected_events(state->collected_events);

    // narrow down to micro seconds
    metadata_array[1] = ((metadata.host_ts_ns.stop - metadata.host_ts_ns.start) / 1000);

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