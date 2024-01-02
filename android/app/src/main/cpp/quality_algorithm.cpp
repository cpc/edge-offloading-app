//
// Created by rabijl on 11/23/23.
//

#include <ctime>
#include <cstdint>
#include <string.h>

#include "hevc_compression.h"
#include "jpeg_compression.h"
#include "platform.h"
#include "quality_algorithm.h"

#define EVAL_INTERVAL_MS 2000
#define QUALITY_VERBOSITY 1

#define NUM_JPEG_CONFIGS 2
#define NUM_HEVC_CONFIGS 2
#define NUM_CODEC_POINTS (1 + NUM_JPEG_CONFIGS + NUM_HEVC_CONFIGS)

typedef struct {
    float enc_time_ms;
    float dec_time_ms;
    float dnn_time_ms;  // DNN + postprocess + reconstruct
    float host_time_ms; // TODO: this one is volatile, we should use the projected one!
    float size_kb;
    float iou;
    float pow_w;
} codec_stats_t;

static int64_t LAST_TIMESTAMP_NS = 0;
static int64_t SINCE_LAST_EVAL_MS = 0;

static compression_t CODEC_POINTS[NUM_CODEC_POINTS] = {NO_COMPRESSION, JPEG_COMPRESSION,
                                                       JPEG_COMPRESSION, HEVC_COMPRESSION,
                                                       HEVC_COMPRESSION};

static int CUR_CODEC_POINT = 0;
static int CUR_NSAMPLES[NUM_CODEC_POINTS] = {0};

static jpeg_config_t JPEG_CONFIGS[NUM_JPEG_CONFIGS] = {{.quality = 80},
                                                       {.quality = 50}};

static hevc_config_t HEVC_CONFIGS[NUM_HEVC_CONFIGS] = {{.quality = 0, .bitrate = 640 * 480},
                                                       {.quality = 0, .bitrate = 640 * 480 / 5}};

static int CODEC_CONFIGS[NUM_CODEC_POINTS] = {0, 0, 1, 0, 1};

// TODO: Fill in real values from measurements
static codec_stats_t CODEC_STATS[NUM_CODEC_POINTS] = {
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 20.0f, .host_time_ms = 900.0f, .size_kb = 460.8f, .iou = 1.0f, .pow_w = 4.0f},
        {.enc_time_ms = 4.0f, .dec_time_ms = 4.0f, .dnn_time_ms = 20.0f, .host_time_ms = 250.0f, .size_kb = 41.3f, .iou = 0.8f, .pow_w = 4.0f},
        {.enc_time_ms = 3.0f, .dec_time_ms = 3.0f, .dnn_time_ms = 20.0f, .host_time_ms = 250.0f, .size_kb = 30.0f, .iou = 0.6f, .pow_w = 4.0f},
        {.enc_time_ms = 2.0f, .dec_time_ms = 2.0f, .dnn_time_ms = 20.0f, .host_time_ms = 200.0f, .size_kb = 18.7f, .iou = 0.9f, .pow_w = 4.0f},
        {.enc_time_ms = 1.0f, .dec_time_ms = 1.0f, .dnn_time_ms = 20.0f, .host_time_ms = 200.0f, .size_kb = 10.0f, .iou = 0.7f, .pow_w = 4.0f},
};

static int find_event(const char *description, const event_array_t *event_array) {
    for (int i = 0; i < event_array->current_capacity; ++i) {
        if (strcmp(description, event_array->array[i].description) == 0) {
            return i;
        }
    }

    return -1;
}

static int get_event_time_ms(cl_event event, const char *description, int start_cmd, int end_cmd,
                             float *time_ms) {
    int status = CL_SUCCESS;

    cl_ulong start_time_ns, end_time_ns;
    status = clGetEventProfilingInfo(event, start_cmd, sizeof(cl_ulong), &start_time_ns, NULL);
    if (status != CL_SUCCESS) {
        LOGE("QUALITY | Could not get profiling info of start command %#04x of event %s: %d\n",
             start_cmd, description, status);
        return status;
    }

    status = clGetEventProfilingInfo(event, end_cmd, sizeof(cl_ulong), &end_time_ns, NULL);
    if (status != CL_SUCCESS) {
        LOGE("QUALITY | Could not get profiling info of end command %#04x of event %s: %d\n",
             end_cmd,
             description, status);
        return status;
    }

    *time_ms = (float) (end_time_ns - start_time_ns) / 1e6;

    return status;
}

int evaluate_parameters(int frame_index, float power, float iou, uint64_t size_bytes,
                        const event_array_t *event_array,
                        const event_array_t *eval_event_array, const host_ts_ns_t *host_ts_ns,
                        compression_t *compression) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t now_ns = now.tv_sec * 1e9 + now.tv_nsec;

    if (QUALITY_VERBOSITY >= 2) {
        LOGI("QUALITY | frame %d | start %5.2f ms\n", frame_index,
             (host_ts_ns->before_enc - host_ts_ns->start) / 1e6);
        LOGI("QUALITY | frame %d | enc   %5.2f ms\n", frame_index,
             (host_ts_ns->before_fill - host_ts_ns->before_enc) / 1e6);
        LOGI("QUALITY | frame %d | fill  %5.2f ms\n", frame_index,
             (host_ts_ns->before_dnn - host_ts_ns->before_fill) / 1e6);
        LOGI("QUALITY | frame %d | dnn   %5.2f ms\n", frame_index,
             (host_ts_ns->before_eval - host_ts_ns->before_dnn) / 1e6);
        LOGI("QUALITY | frame %d | eval  %5.2f ms\n", frame_index,
             (host_ts_ns->before_wait - host_ts_ns->before_eval) / 1e6);
        LOGI("QUALITY | frame %d | wait  %5.2f ms\n", frame_index,
             (host_ts_ns->after_wait - host_ts_ns->before_wait) / 1e6);
        LOGI("QUALITY | frame %d | end   %5.2f ms\n", frame_index,
             (host_ts_ns->stop - host_ts_ns->after_wait) / 1e6);
    }

    if (QUALITY_VERBOSITY >= 1) {
        LOGI("QUALITY | frame %d | pow %5.3f W, iou %5.4f, size %d B, compression %d (%s), cur_nsamples: %d\n",
             frame_index, power, iou, size_bytes, CUR_CODEC_POINT,
             get_compression_name(CODEC_POINTS[CUR_CODEC_POINT]), CUR_NSAMPLES[CUR_CODEC_POINT]);
    }

    if (LAST_TIMESTAMP_NS == 0) {
        LOGI("QUALITY | frame %d | skip first\n", frame_index);
        LAST_TIMESTAMP_NS = now_ns;
        return 0;
    }

    SINCE_LAST_EVAL_MS += (now_ns - LAST_TIMESTAMP_NS) / 1e6;
    LAST_TIMESTAMP_NS = now_ns;

    // collect statistics

    int status = CL_SUCCESS;

    float host_time_ms = (host_ts_ns->after_wait - host_ts_ns->before_enc) / 1e6;
    const char *event_names[5] = {"enc_event", "dec_event", "dnn_event", "postprocess_event",
                                  "reconstruct_event"};
    float event_times_ms[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 5; ++i) {
        int event_i = find_event(event_names[i], event_array);
        if (event_i < 0) {
            continue;
        }

        status = get_event_time_ms(event_array->array[event_i].event, event_names[i],
                                   CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END,
                                   &event_times_ms[i]);
        if (status != CL_SUCCESS) {
            LOGE("QUALITY | frame %d | Could not get event time, skipping\n", frame_index);
            return 0;
        }
    }

    // Incremental averaging https://blog.demofox.org/2016/08/23/incremental-averaging/
    CUR_NSAMPLES[CUR_CODEC_POINT] += 1;
    int nsamples = CUR_NSAMPLES[CUR_CODEC_POINT];
    codec_stats_t *cur_stats = &CODEC_STATS[CUR_CODEC_POINT];
    cur_stats->enc_time_ms += (event_times_ms[0] - cur_stats->enc_time_ms) / nsamples;
    cur_stats->dec_time_ms += (event_times_ms[1] - cur_stats->enc_time_ms) / nsamples;
    cur_stats->dnn_time_ms += ((event_times_ms[2] + event_times_ms[3] + event_times_ms[4]) -
                               cur_stats->dnn_time_ms) / nsamples;
    cur_stats->size_kb += (size_bytes / 1e3 - cur_stats->size_kb) / nsamples;
    cur_stats->host_time_ms += (host_time_ms - cur_stats->host_time_ms) / nsamples;
    if (iou >= 0.0f) {
        cur_stats->iou += (iou - cur_stats->iou) / nsamples; // TODO: Fix this
    }
    cur_stats->pow_w += (power - cur_stats->pow_w) / nsamples;

    if (QUALITY_VERBOSITY >= 1) {
        LOGI("QUALITY | frame %d |  enc time: %5.3f ms\n", frame_index, cur_stats->enc_time_ms);
        LOGI("QUALITY | frame %d |  dec time: %5.3f ms\n", frame_index, cur_stats->dec_time_ms);
        LOGI("QUALITY | frame %d |  dnn time: %5.3f ms\n", frame_index, cur_stats->dnn_time_ms);
        LOGI("QUALITY | frame %d | host time: %5.3f ms\n", frame_index, cur_stats->host_time_ms);
        LOGI("QUALITY | frame %d |      size: %5.0f kB\n", frame_index, cur_stats->size_kb);
        LOGI("QUALITY | frame %d |       iou: %5.4f\n", frame_index, cur_stats->iou);
        LOGI("QUALITY | frame %d |       pow: %5.3f W\n", frame_index, cur_stats->pow_w);
    }

    int is_eval_frame = 0;
    if (SINCE_LAST_EVAL_MS >= EVAL_INTERVAL_MS) {
        is_eval_frame = 1;
    }

    if (QUALITY_VERBOSITY >= 1) {
        LOGI("QUALITY | frame %d | since last eval: %ld ms, should eval: %d\n", frame_index,
             SINCE_LAST_EVAL_MS, is_eval_frame);
    }

    if (!is_eval_frame) {
        return is_eval_frame;
    }

    SINCE_LAST_EVAL_MS = 0;

    // perform codec selection

    float non_static_time =
            cur_stats->host_time_ms - cur_stats->enc_time_ms - cur_stats->dec_time_ms -
            cur_stats->dnn_time_ms;
    float byterate_kbps = cur_stats->size_kb / non_static_time;

    if (QUALITY_VERBOSITY >= 1) {
        LOGI("QUALITY | frame %d | non-static time: %5.2f ms, %.2f kBps\n", frame_index,
             non_static_time, byterate_kbps);
    }

    float projected_host_times[NUM_CODEC_POINTS] = {0.0f};
    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        float size_kb = CODEC_STATS[i].size_kb;
        projected_host_times[i] = CODEC_STATS[i].enc_time_ms + CODEC_STATS[i].dec_time_ms +
                                  CODEC_STATS[i].dnn_time_ms + size_kb / byterate_kbps;

        if (QUALITY_VERBOSITY >= 1) {
            LOGI("QUALITY | frame %d | projected host time %d (%s): %5.2f ms\n", frame_index, i,
                 get_compression_name(CODEC_POINTS[i]), projected_host_times[i]);
        }
    }

    float min_host_time_ms = projected_host_times[CUR_CODEC_POINT];
    int min_i = CUR_CODEC_POINT;
    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        if (i == CUR_CODEC_POINT) {
            continue;
        }

        float time_ms = projected_host_times[i];
        if (time_ms < min_host_time_ms) {
            min_host_time_ms = time_ms;
            min_i = i;
        }
    }

    CUR_CODEC_POINT = min_i;
    *compression = CODEC_POINTS[CUR_CODEC_POINT];

    if (QUALITY_VERBOSITY >= 1) {
        LOGI("QUALITY | frame %d | Selected codec: %d (%s)\n", frame_index, CUR_CODEC_POINT,
             get_compression_name(*compression));
    }

    return is_eval_frame;
}
