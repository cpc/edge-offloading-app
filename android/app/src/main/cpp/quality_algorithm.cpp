//
// Created by rabijl on 11/23/23.
//

#include <ctime>
#include <cstdint>
#include <string.h>
#include <cmath>

#include "platform.h"
#include "quality_algorithm.h"

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

// the below arrays are indexed by CUR_CODEC_ID
static const compression_t CODEC_POINTS[NUM_CODEC_POINTS] = {NO_COMPRESSION, NO_COMPRESSION,
                                                             JPEG_COMPRESSION, JPEG_COMPRESSION,
                                                             HEVC_COMPRESSION, HEVC_COMPRESSION};

static const int CODEC_DEVICES[NUM_CODEC_POINTS] = {0, 2, 2, 2, 2, 2};
static const int CODEC_CONFIGS[NUM_CODEC_POINTS] = {0, 0, 0, 1, 0, 1};
static int CUR_NSAMPLES[NUM_CODEC_POINTS] = {0};

// TODO: Fill in real values from measurements
static codec_stats_t CODEC_STATS[NUM_CODEC_POINTS] = {
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
        {.enc_time_ms = 0.0f, .dec_time_ms = 0.0f, .dnn_time_ms = 0.0f, .size_bytes = 0.0f, .iou = 0.0f, .pow_w = 0.0f},
};

// indexed by CODEC_CONFIGS[CUR_CODEC_ID]:
static const jpeg_config_t JPEG_CONFIGS[NUM_JPEG_CONFIGS] = {
        {.quality = 80},
        {.quality = 50},
};

// TODO: tune these parameters
static const hevc_config_t HEVC_CONFIGS[NUM_HEVC_CONFIGS] = {
        {.i_frame_interval = 2, .framerate = 5, .bitrate = 640 * 480},
        {.i_frame_interval = 2, .framerate = 5, .bitrate = 640 * 480 / 5},
};

static int is_near_zero(float x) {
    const float EPSILON_F32 = 1e-6f;
    return fabs(x) <= EPSILON_F32;
}

// General data structure for holding data required to perform incremental least squares fitting, see:
// https://blog.demofox.org/2016/12/22/incremental-least-squares-curve-fitting
typedef struct {
    float ATA[2][2];
    float ATY[2];
} least_squares_lin_t;

// We model our network as a line with x = log10(size_bytes), y = network_ms
static least_squares_lin_t NETWORK_MODEL = {
        .ATA = {{0.0f, 0.0f},
                {0.0f, 0.0f}},
        .ATY = {0.0f, 0.0f}
};

// Incrementaly add a sample to the network model
static void update_network_model(float x, float y, least_squares_lin_t *model) {
    model->ATA[0][0] += 1.0f;
    model->ATA[0][1] += x;
    model->ATA[1][0] += x;
    model->ATA[1][1] += x * x;

    model->ATY[0] += y;
    model->ATY[1] += y * x;
}

// Incrementally remove a sample from the network model
static void decrement_network_model(float x, float y, least_squares_lin_t *model) {
    model->ATA[0][0] -= 1.0f;
    model->ATA[0][1] -= x;
    model->ATA[1][0] -= x;
    model->ATA[1][1] -= x * x;

    model->ATY[0] -= y;
    model->ATY[1] -= y * x;
}

// Get the actual model parameters (e.g., slope and offset of a linear model)
static void eval_network_model(const least_squares_lin_t model, float *slope, float *offset) {
    float a = model.ATA[0][0]; // number of collected samples
    float b = model.ATA[0][1];
    float c = model.ATA[1][0];
    float d = model.ATA[1][1];
    float det = (a * d) - (b * c);

    *offset = 0.0f;
    *slope = 0.0f;

    if (is_near_zero(det)) {
        if (!is_near_zero(a)) {
            *offset = model.ATY[0] / a;
        }
    } else {
        float idet = 1.0f / det;
        float iATA[2][2] = {{d * idet,  -b * idet},
                            {-c * idet, a * idet}};

        *offset = iATA[0][0] * model.ATY[0] + iATA[0][1] * model.ATY[1];
        *slope = iATA[1][0] * model.ATY[0] + iATA[1][1] * model.ATY[1];
    }
}

// Size of the ringbuffer, enough for 10 seconds of samples running at 100 FPS
#define MAX_NETWORK_NSAMPLES 1024

// How old samples to keep in the network stats
// TODO: Current value effectively disables any samples dropping
static const int64_t MAX_SAMPLE_AGE_NS = INT64_MAX;

// Ringbuffer struct for holding the last X seconds worth of samples
typedef struct {
    float x[MAX_NETWORK_NSAMPLES];
    float y[MAX_NETWORK_NSAMPLES];
    int64_t timestamp_ns[MAX_NETWORK_NSAMPLES];  // needed to know which samples to remove
    int pos;
} ringbuffer_t;

static ringbuffer_t NETWORK_RINGBUFFER = {
        .x = {0.0f},
        .y = {0.0f},
        .timestamp_ns = {0},
        .pos = -1,
};

// Add a new sample to the ringbuffer
static void
add_sample(float x, float y, ringbuffer_t *ringbuffer, least_squares_lin_t *network_model) {
    // add the sample to the network model
    update_network_model(x, y, network_model);

    // add the sample to the ringbuffer
    ringbuffer->pos = (ringbuffer->pos + 1) % MAX_NETWORK_NSAMPLES;
    ringbuffer->x[ringbuffer->pos] = x;
    ringbuffer->y[ringbuffer->pos] = y;
    ringbuffer->timestamp_ns[ringbuffer->pos] = get_timestamp_ns();

    QLOGI(3, "QUALITY | DEBUG | added sample %d: %ld\n", ringbuffer->pos,
          ringbuffer->timestamp_ns[ringbuffer->pos]);
}

// Remove samples from the ringbuffer and the network model older than X ns
static int remove_old_samples(int64_t older_than_ns, ringbuffer_t *ringbuffer,
                              least_squares_lin_t *network_model) {
    const int cur_pos = ringbuffer->pos;
    const int64_t cur_timestamp_ns = ringbuffer->timestamp_ns[cur_pos];
    QLOGI(3, "QUALITY | DEBUG | cur_pos %d, cur_ts %ld, older than %ld\n", cur_pos,
          cur_timestamp_ns,
          older_than_ns);

    int nremoved = 0;
    int i = cur_pos - 1;
    while (ringbuffer->timestamp_ns[i] != 0) {
        if (i < 0) {
            i = MAX_NETWORK_NSAMPLES - 1;
        }

        QLOGI(3, "QUALITY | DEBUG | ts[%d] %ld, diff %ld\n", i, ringbuffer->timestamp_ns[i],
              cur_timestamp_ns - ringbuffer->timestamp_ns[i]);

        if (i == cur_pos) {
            // do not remove the latest sample
            break;
        }

        int64_t ts = ringbuffer->timestamp_ns[i];

        if (ts != 0 && (cur_timestamp_ns - ts) > older_than_ns) {
            const float x = ringbuffer->x[i];
            const float y = ringbuffer->y[i];

            QLOGI(3, "QUALITY | DEBUG | -- removed\n");

            // remove data from network model
            decrement_network_model(x, y, network_model);

            // remove the sample from ringbuffer
            ringbuffer->timestamp_ns[i] = 0;
            ringbuffer->x[i] = 0.0f;
            ringbuffer->y[i] = 0.0f;
            nremoved += 1;
        }

        i -= 1;
    }

    return nremoved;
}

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
          "QUALITY | frame %d | pow %5.3f W, iou %5.4f, size %ld B, compression %d (%s), cur_nsamples: %d, device: %d, codec changed: %d\n",
          frame_index, power, iou, size_bytes, CUR_CODEC_ID,
          get_compression_name(CODEC_POINTS[CUR_CODEC_ID]), CUR_NSAMPLES[CUR_CODEC_ID],
          result_config->device_index, DID_CODEC_CHANGE);

    QLOGI(2, "QUALITY | frame %d | start %7.2f ms\n", frame_index,
          (host_ts_ns->before_enc - host_ts_ns->start) / 1e6);
    QLOGI(2, "QUALITY | frame %d | enc   %7.2f ms\n", frame_index,
          (host_ts_ns->before_fill - host_ts_ns->before_enc) / 1e6);
    QLOGI(2, "QUALITY | frame %d | fill  %7.2f ms\n", frame_index,
          (host_ts_ns->before_dnn - host_ts_ns->before_fill) / 1e6);
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
        float nsamples = (float) (CUR_NSAMPLES[CUR_CODEC_ID]);
        codec_stats_t *cur_stats = &CODEC_STATS[CUR_CODEC_ID];
        float cur_size_bytes_f = (float) (size_bytes);

        cur_stats->enc_time_ms += (event_times_ms[0] - cur_stats->enc_time_ms) / nsamples;
        cur_stats->dec_time_ms += (event_times_ms[1] - cur_stats->dec_time_ms) / nsamples;
        cur_stats->dnn_time_ms += ((event_times_ms[2] + event_times_ms[3] + event_times_ms[4]) -
                                   cur_stats->dnn_time_ms) / nsamples;
        cur_stats->size_bytes += (cur_size_bytes_f - cur_stats->size_bytes) / nsamples;
        if (iou >= 0.0f) {
            cur_stats->iou += (iou - cur_stats->iou) / nsamples; // TODO: Fix this
        }
        cur_stats->pow_w += (power - cur_stats->pow_w) / nsamples;

        float network_ms = 0.0f;
        if (CODEC_DEVICES[CUR_CODEC_ID] != 0) {
            // only remote devices have network
            network_ms = host_time_ms - event_times_ms[0] - event_times_ms[1] - event_times_ms[2] -
                         event_times_ms[3] - event_times_ms[4];
        }

        // update incremental least squares data
        if (CODEC_DEVICES[CUR_CODEC_ID] != 0) {
            float size_log10 = log10f(cur_size_bytes_f);
            add_sample(size_log10, network_ms, &NETWORK_RINGBUFFER, &NETWORK_MODEL);
            int nremoved = remove_old_samples(MAX_SAMPLE_AGE_NS, &NETWORK_RINGBUFFER,
                                              &NETWORK_MODEL);

            QLOGI(1, "QUALITY | frame %d | ringbuf pos %d, removed %d samples\n", frame_index,
                  NETWORK_RINGBUFFER.pos, nremoved);
        }

        QLOGI(1, "QUALITY | frame %d |    host time: %8.3f (     avg) ms\n", frame_index,
              host_time_ms);
        QLOGI(1, "QUALITY | frame %d |     enc time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[0], cur_stats->enc_time_ms);
        QLOGI(1, "QUALITY | frame %d |     dec time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[1], cur_stats->dec_time_ms);
        QLOGI(1, "QUALITY | frame %d |     dnn time: %8.3f (%8.3f) ms\n", frame_index,
              event_times_ms[2] + event_times_ms[3] + event_times_ms[4], cur_stats->dnn_time_ms);
        QLOGI(1, "QUALITY | frame %d | network time: %8.3f ms\n", frame_index, network_ms);
        QLOGI(1, "QUALITY | frame %d |         size: %8.0f (%8.0f) B\n", frame_index,
              (float) (size_bytes), cur_stats->size_bytes);
        QLOGI(1, "QUALITY | frame %d |          iou: %8.3f (%8.3f)\n", frame_index, iou,
              cur_stats->iou);
        QLOGI(1, "QUALITY | frame %d |          pow: %8.3f (%8.3f) W\n", frame_index, power,
              cur_stats->pow_w);

        if (ENABLE_PROFILING & config_flags) {
            dprintf(file_descriptor, "%d,quality,cur_codec_point,%d\n", frame_index,
                    CUR_CODEC_ID);
            dprintf(file_descriptor, "%d,quality,host_time_ms,%f\n", frame_index, host_time_ms);
            dprintf(file_descriptor, "%d,quality,enc_time_ms,%f\n", frame_index, event_times_ms[0]);
            dprintf(file_descriptor, "%d,quality,dec_time_ms,%f\n", frame_index, event_times_ms[1]);
            dprintf(file_descriptor, "%d,quality,dnn_time_ms,%f\n", frame_index,
                    event_times_ms[2] + event_times_ms[3] + event_times_ms[4]);
            dprintf(file_descriptor, "%d,quality,network_ms,%f\n", frame_index, network_ms);
        }
    }

    int is_eval_frame = 0;
    if (SINCE_LAST_EVAL_MS >= EVAL_INTERVAL_MS) {
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

    float model_slope = 0.0f;
    float model_offset = 0.0f;

    if (CODEC_DEVICES[CUR_CODEC_ID] != 0) {
        eval_network_model(NETWORK_MODEL, &model_slope, &model_offset);
    }

    QLOGI(1, "QUALITY | frame %d | EVAL | LSE model_slope: %.3f, model_offset: %.3f\n", frame_index,
          model_slope, model_offset);

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

        if (CODEC_DEVICES[CUR_CODEC_ID] != 0 && CODEC_DEVICES[i] != 0) {
            // only remote devices have non-zero network time; also, when running locally, there are no meaningful network stats
            projected_network_time_ms = fmax(model_slope * size_bytes_f_log10 + model_offset, 0.0f);
        }

        if (CODEC_DEVICES[i] == 0) {
            // do not project network time on local devices which don't have coding and network
            projected_host_times_ms[i] = CODEC_STATS[i].dnn_time_ms;
        } else {
            projected_host_times_ms[i] = CODEC_STATS[i].enc_time_ms + CODEC_STATS[i].dec_time_ms +
                                         CODEC_STATS[i].dnn_time_ms + projected_network_time_ms;
        }

        QLOGI(1,
              "QUALITY | frame %d | EVAL | size: %8.0f B, projected host time %d (%s): %8.2f ms (network %8.2f ms, overhead %7.2f ms)\n",
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
    result_config->device_index = CODEC_DEVICES[CUR_CODEC_ID];
    const int config_id = CODEC_CONFIGS[CUR_CODEC_ID];

    switch (result_config->compression_type) {
        case JPEG_COMPRESSION:
            result_config->config.jpeg.quality = JPEG_CONFIGS[config_id].quality;
            break;
        case HEVC_COMPRESSION:
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
        dprintf(file_descriptor, "%d,quality_eval,model_slope,%f\n", frame_index, model_slope);
        dprintf(file_descriptor, "%d,quality_eval,model_offset,%f\n", frame_index, model_offset);
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
