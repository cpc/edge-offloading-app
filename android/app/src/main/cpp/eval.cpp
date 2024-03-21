//
// Created by rabijl on 11/23/23.
//

#include <ctime>
#include <cstdint>
#include <string.h>
#include <cmath>

#include "eval.h"
#include "eval_network_model.h"

#include <Tracy.hpp>

#include "platform.h"
#include "sharedUtils.h"
#include "poclImageProcessorTypes.h"

#define EVAL_INTERVAL_MS 2000 // How often to decide which codec to run
#define QUALITY_VERBOSITY 2 // Verbosity of messages printed from this file (0 turns them off)

// Simple LOGI wrapper to reduce clutter
#define QLOGI(verbosity, ...) \
    if (verbosity <= QUALITY_VERBOSITY) { LOGI(__VA_ARGS__); }

// Number of configurations considered for each codec
#define NUM_JPEG_CONFIGS 2
#define NUM_HEVC_CONFIGS 2
#define NUM_CODEC_POINTS (2 + NUM_JPEG_CONFIGS + NUM_HEVC_CONFIGS)

// Statistics collected for each codec
typedef struct {
    float enc_time_ms;  // encoding time
    float dec_time_ms;  // decoding time
    float dnn_time_ms;  // DNN + postprocess + reconstruct
    float size_bytes;   // size of the encoded frame
    float iou;          // overlap between DNN output with and without compression (1.0 is the best)
    float pow_w;        // power drawn from battery
} codec_stats_t;

// Bookkeeping variables
static int64_t LAST_TIMESTAMP_NS = 0;
static int64_t SINCE_LAST_EVAL_MS = 0;
static int DID_CODEC_CHANGE = 1;

// currently running codec
int CUR_CODEC_ID = 0;

// number of samples collected for each codec
static int CUR_NSAMPLES[NUM_CODEC_POINTS] = {0};

// starting values zero ensure that the selection first goes through all of them and thus initializes the values
static codec_stats_t CODEC_STATS[NUM_CODEC_POINTS] = {
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
};

// ping statistics
static const float PING_ALPHA = 0.7f;
static float PING_MS = 0.0f;

int is_near_zero(float x) {
    const float EPSILON_F32 = 1e-6f;
    return fabs(x) <= EPSILON_F32;
}

static const network_model_name_t NEWTORK_MODEL_NAME = ILSF;
static network_model_t *NETWORK_MODEL = nullptr;

// Find an event in an array
static int find_event(const char *description, const event_array_t *event_array) {
    for (int i = 0; i < event_array->current_capacity; ++i) {
        if (strcmp(description, event_array->array[i].description) == 0) {
            return i;
        }
    }

    return -1;
}

// Calculate time in ms of event (start/end_cmd specify the event stage, i.e., queued, submitted, started, ended)
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

void init_eval() {
    QLOGI(1, "QUALITY | Initializing model %s\n", NETWORK_NAMES_STR[NEWTORK_MODEL_NAME]);
    NETWORK_MODEL = init_network_model(NEWTORK_MODEL_NAME);
}

void destroy_eval() {
    destroy_network_model(NETWORK_MODEL);
}

int evaluate_parameters(int frame_index, float power, float iou, uint64_t size_bytes,
                        int file_descriptor, int config_flags, const event_array_t *event_array,
                        const event_array_t *eval_event_array, const host_ts_ns_t *host_ts_ns,
                        codec_config_t *result_config) {
    struct timespec ts_start, ts_mid, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    int64_t start_ns = ts_start.tv_sec * 1000000000 + ts_start.tv_nsec;

    QLOGI(1,
          "QUALITY | frame %d | ========================================================================\n",
          frame_index);
    QLOGI(1,
          "QUALITY | frame %d | pow %5.3f W, iou %5.4f, size %ld + %d B, compression %d (%s), cur_nsamples: %d, device: %d, codec changed: %d\n",
          frame_index, power, iou, size_bytes, TOTAL_RX_SIZE, CUR_CODEC_ID,
          get_compression_name(CODEC_POINTS[CUR_CODEC_ID]), CUR_NSAMPLES[CUR_CODEC_ID],
          result_config->device_type, DID_CODEC_CHANGE);
    QLOGI(2, "QUALITY | frame %d | start %7.2f ms\n", frame_index,
          (host_ts_ns->before_fill - host_ts_ns->start) / 1e6);
    QLOGI(2, "QUALITY | frame %d | fill  %7.2f ms\n", frame_index,
          (host_ts_ns->before_enc - host_ts_ns->before_fill) / 1e6);
    QLOGI(2, "QUALITY | frame %d | enc   %7.2f ms\n", frame_index,
          (host_ts_ns->before_dnn - host_ts_ns->before_enc) / 1e6);
    QLOGI(2, "QUALITY | frame %d | dnn   %7.2f ms\n", frame_index,
          (host_ts_ns->before_eval - host_ts_ns->before_dnn) / 1e6);
    QLOGI(2, "QUALITY | frame %d | eval  %7.2f ms\n", frame_index,
          (host_ts_ns->before_wait - host_ts_ns->before_eval) / 1e6);
    QLOGI(2, "QUALITY | frame %d | wait  %7.2f ms\n", frame_index,
          (host_ts_ns->after_wait - host_ts_ns->before_wait) / 1e6);
    QLOGI(2, "QUALITY | frame %d | end   %7.2f ms\n", frame_index,
          (host_ts_ns->stop - host_ts_ns->after_wait) / 1e6);

    if (LAST_TIMESTAMP_NS == 0) {
        QLOGI(1, "QUALITY | frame %d | skip first\n", frame_index);
        LAST_TIMESTAMP_NS = start_ns;
        return 0;
    }

    float fill_ping_ms = (float)(host_ts_ns->fill_ping_duration) / 1e6;
    PING_MS = ewma(fill_ping_ms, PING_MS, PING_ALPHA);

    QLOGI(1, "QUALITY | frame %d | fill ping time: %8.3f (%8.3f) ms\n", frame_index, fill_ping_ms, PING_MS);

    if (ENABLE_PROFILING & config_flags) {
        dprintf(file_descriptor, "%d,quality,cur_codec_point,%d\n", frame_index, CUR_CODEC_ID);
        dprintf(file_descriptor, "%d,quality,avg_fill_ping_ms,%f\n", frame_index, PING_MS);
    }

    // update network model data
    // TODO: Adding the the MTU size distorts the model, but using pocl write/read size makes it smoother
    NETWORK_MODEL->add_sample(FILL_PING_SIZE, fill_ping_ms, &NETWORK_MODEL->data);

    size_bytes = size_bytes + TOTAL_RX_SIZE;

    SINCE_LAST_EVAL_MS += (start_ns - LAST_TIMESTAMP_NS) / 1e6;
    LAST_TIMESTAMP_NS = start_ns;

    // collect statistics

    int status = CL_SUCCESS;

    if (!DID_CODEC_CHANGE) {
        // TODO: First frame is skipped due to latency spikes, this should be fixed
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
        // Incremental variance: https://math.stackexchange.com/a/1379804
        CUR_NSAMPLES[CUR_CODEC_ID] += 1;
        float cur_nsamples = (float) (CUR_NSAMPLES[CUR_CODEC_ID]);
        codec_stats_t *cur_stats = &CODEC_STATS[CUR_CODEC_ID];
        float cur_size_bytes_f = (float) (size_bytes);
        float cur_dnn_time_ms = event_times_ms[2] + event_times_ms[3] + event_times_ms[4];
        float alpha = 0.7f;  // higher means more stable EWMA, lower means more agile

        if (cur_nsamples == 1) {
            // If it's the first sample, set the average to the current value
            alpha = 0.0f;
        }

        cur_stats->enc_time_ms = ewma(event_times_ms[0], cur_stats->enc_time_ms, alpha);
        cur_stats->dec_time_ms = ewma(event_times_ms[1], cur_stats->dec_time_ms, alpha);
        cur_stats->dnn_time_ms = ewma(cur_dnn_time_ms, cur_stats->dnn_time_ms, alpha);
        cur_stats->size_bytes = ewma(cur_size_bytes_f, cur_stats->size_bytes, alpha);
        if ((iou > 0.0f) || (is_near_zero(iou))) {
            // -2 means no detections on both reference and compressed images. This would mean IOU=1
            // but let's not count it to avoid distorting the statistics. The other negative values
            // are init/unimplemented, let's not count them.
            cur_stats->iou = ewma(iou, cur_stats->iou, alpha);
        }
        cur_stats->pow_w = ewma(power, cur_stats->pow_w, alpha);

        float network_ms = 0.0f;
        if (CODEC_DEVICES[CUR_CODEC_ID] != 0) {
            // only remote devices have network
            network_ms = host_time_ms - event_times_ms[0] - event_times_ms[1] - event_times_ms[2] -
                         event_times_ms[3] - event_times_ms[4];
        }

        if (CODEC_DEVICES[CUR_CODEC_ID] != 0) {
            // update network model data
            NETWORK_MODEL->add_sample(cur_size_bytes_f, network_ms, &NETWORK_MODEL->data);
        }

        QLOGI(1, "QUALITY | frame %d |      host time: %8.3f (     avg) ms\n", frame_index,
              host_time_ms);
        QLOGI(1, "QUALITY | frame %d |       enc time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[0], cur_stats->enc_time_ms);
        QLOGI(1, "QUALITY | frame %d |       dec time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[1], cur_stats->dec_time_ms);
        QLOGI(1, "QUALITY | frame %d |       dnn time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[2] + event_times_ms[3] + event_times_ms[4], cur_stats->dnn_time_ms);
        QLOGI(1, "QUALITY | frame %d |   network time: %8.3f ms\n", frame_index, network_ms);
        QLOGI(1, "QUALITY | frame %d |           size: %8.0f (%8.0f) B\n", frame_index,
              (float) (size_bytes), cur_stats->size_bytes);
        QLOGI(1, "QUALITY | frame %d |            iou: %8.3f (%8.3f)\n", frame_index, iou,
              cur_stats->iou);
        QLOGI(1, "QUALITY | frame %d |            pow: %8.3f (%8.3f) W\n", frame_index, power,
              cur_stats->pow_w);

        if (ENABLE_PROFILING & config_flags) {
            dprintf(file_descriptor, "%d,quality,host_time_ms,%f\n", frame_index, host_time_ms);
            dprintf(file_descriptor, "%d,quality,enc_time_ms,%f\n", frame_index, event_times_ms[0]);
            dprintf(file_descriptor, "%d,quality,dec_time_ms,%f\n", frame_index, event_times_ms[1]);
            dprintf(file_descriptor, "%d,quality,dnn_time_ms,%f\n", frame_index,
                    event_times_ms[2] + event_times_ms[3] + event_times_ms[4]);
            dprintf(file_descriptor, "%d,quality,network_ms,%f\n", frame_index, network_ms);
            for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
                dprintf(file_descriptor, "%d,quality,avg_size_%d,%f\n", frame_index, i, CODEC_STATS[i].size_bytes);
            }
        }
    } else {
        QLOGI(1, "QUALITY | frame %d | codec changed, skipped sample\n", frame_index);
    }

    int is_eval_frame = 0;
    if (!DID_CODEC_CHANGE && (SINCE_LAST_EVAL_MS >= EVAL_INTERVAL_MS)) {
        is_eval_frame = 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_mid);
    int64_t mid_ns = ts_mid.tv_sec * 1000000000 + ts_mid.tv_nsec;

    QLOGI(1,
          "QUALITY | frame %d | --- update took %.2f ms, since last eval: %ld ms, should eval: %d\n",
          frame_index, (mid_ns - start_ns) / 1e6, SINCE_LAST_EVAL_MS, is_eval_frame);

    if (ENABLE_PROFILING & config_flags) {
        dprintf(file_descriptor, "%d,quality,update_ms,%f\n", frame_index,
                (double) (mid_ns - start_ns) / 1e6);
        dprintf(file_descriptor, "%d,quality,since_last_eval_ms,%ld\n", frame_index,
                SINCE_LAST_EVAL_MS);
        dprintf(file_descriptor, "%d,quality,should_eval,%d\n", frame_index, is_eval_frame);
    }

    DID_CODEC_CHANGE = 0;

    if (!is_eval_frame) {
        return 0;
    }

    SINCE_LAST_EVAL_MS = 0;

    // determine which codec to select

    // TODO: The samples from fill ping collected when device == 0 distort the model significantly
    NETWORK_MODEL->update_params(&NETWORK_MODEL->data, &NETWORK_MODEL->params);
    NETWORK_MODEL->print_params(&NETWORK_MODEL->params, frame_index, 1);

    // Based on the current network transfer time, estimate what would be the network transfer time
    // for other codecs using the average encoded frame size of each codec.
    float projected_host_times_ms[NUM_CODEC_POINTS] = {0.0f};

    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        float size_bytes_f = CODEC_STATS[i].size_bytes;
        float size_bytes_f_log10 = 0.0f;
        if (size_bytes_f > 0.0f) {
            size_bytes_f_log10 = log10f(size_bytes_f);
        }

        float projected_network_time_ms = 0.0f;

        if (CODEC_DEVICES[i] != 0) {
            // only remote devices have non-zero network time
            projected_network_time_ms = NETWORK_MODEL->predict(&NETWORK_MODEL->params,
                                                               size_bytes_f_log10);
        }

        if (CODEC_DEVICES[i] == 0) {
            // do not project network time on local devices which don't have coding and network
            projected_host_times_ms[i] = CODEC_STATS[i].dnn_time_ms;
        } else {
            projected_host_times_ms[i] = CODEC_STATS[i].enc_time_ms + CODEC_STATS[i].dec_time_ms +
                                         CODEC_STATS[i].dnn_time_ms + projected_network_time_ms;
        }

        QLOGI(1,
              "QUALITY | frame %d | EVAL | avg size: %8.0f B, projected host time %d (%10s): %8.2f ms (network %8.2f ms, overhead %7.2f ms)\n",
              frame_index, size_bytes_f, i, get_compression_name(CODEC_POINTS[i]),
              projected_host_times_ms[i], projected_network_time_ms,
              projected_host_times_ms[i] - projected_network_time_ms);
    }

    float min_host_time_ms = projected_host_times_ms[CUR_CODEC_ID];
    int min_i = CUR_CODEC_ID;

    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        if (i == CUR_CODEC_ID) {
            continue;
        }

        float time_ms = projected_host_times_ms[i];
        if (time_ms < min_host_time_ms) {
            min_host_time_ms = time_ms;
            min_i = i;
        }
    }

    // switch codec

    int old_codec_point = CUR_CODEC_ID;
    CUR_CODEC_ID = min_i;
//    CUR_CODEC_ID = (CUR_CODEC_ID + 1) % NUM_CODEC_POINTS; // DEBUG
    result_config->compression_type = CODEC_POINTS[CUR_CODEC_ID];
    result_config->device_type = CODEC_DEVICES[CUR_CODEC_ID];
    const int config_id = CODEC_CONFIGS[CUR_CODEC_ID];

    switch (result_config->compression_type) {
        case JPEG_COMPRESSION:
            result_config->config.jpeg.quality = JPEG_CONFIGS[config_id].quality;
            break;
        case HEVC_COMPRESSION:
        case SOFTWARE_HEVC_COMPRESSION:
            result_config->config.hevc.i_frame_interval = HEVC_CONFIGS[config_id].i_frame_interval;
            result_config->config.hevc.framerate = HEVC_CONFIGS[config_id].framerate;
            result_config->config.hevc.bitrate = HEVC_CONFIGS[config_id].bitrate;
            break;
        case NO_COMPRESSION:
        default:
            break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    int64_t end_ns = ts_end.tv_sec * 1000000000 + ts_end.tv_nsec;

    QLOGI(1, "QUALITY | frame %d | EVAL | eval took %.2f ms\n", frame_index,
          (end_ns - start_ns) / 1e6);
    QLOGI(1, "QUALITY | frame %d | EVAL | >>> Selected codec: %d (%s) -> %d (%s) <<<\n",
          frame_index, old_codec_point, get_compression_name(CODEC_POINTS[old_codec_point]),
          CUR_CODEC_ID, get_compression_name(result_config->compression_type));

    if (ENABLE_PROFILING & config_flags) {
        dprintf(file_descriptor, "%d,quality_eval,eval_ms,%f\n", frame_index,
                (double) (end_ns - mid_ns) / 1e6);
        NETWORK_MODEL->log_params(&NETWORK_MODEL->params, frame_index, file_descriptor);
        for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
            dprintf(file_descriptor, "%d,quality_eval,projected_host_time_%d_ms,%f\n", frame_index,
                    i, projected_host_times_ms[i]);
        }
        dprintf(file_descriptor, "%d,quality_eval,selected_codec,%d\n", frame_index,
                CUR_CODEC_ID);
    }

    DID_CODEC_CHANGE = old_codec_point != CUR_CODEC_ID;

    return is_eval_frame;
}
