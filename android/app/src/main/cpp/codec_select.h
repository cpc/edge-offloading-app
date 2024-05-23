/*
 * Code related to automatic selection of the codec and its parameters.
 */

#ifndef POCL_AISA_DEMO_CODEC_SELECT_H
#define POCL_AISA_DEMO_CODEC_SELECT_H

#include "poclImageProcessorV2.h"

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SELECT_VERBOSITY 0 // Verbosity of messages printed from this file (0 turns them off)

// Simple LOGI wrapper to reduce clutter
#define SLOGI(verbosity, ...) \
    if (verbosity <= SELECT_VERBOSITY) { LOGI(__VA_ARGS__); }

/**
 * Number of codec configs considered by the selection algorithm (should be >= 1 to always have at
 * least local execution).
 */
#define NUM_CONFIGS 6

/**
 * Codec parameters relevant for codec selection.
 *
 * This matches codec_config_t but without unnecessary fields.
 */
typedef struct {
    compression_t compression_type;
    device_type_enum device_type;
    union {
        jpeg_config_t jpeg;
        hevc_config_t hevc;
    } config;
} codec_params_t;

/**
 * Codec configs the selection algorithm is allowed to consider.
 *
 * The first one should always be local, the second one remote without compression.
 */
static const codec_params_t CONFIGS[NUM_CONFIGS] = {
        {.compression_type = NO_COMPRESSION, .device_type= LOCAL_DEVICE, .config = {NULL}},
        {.compression_type = NO_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {NULL}},
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 99}}},
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 90}}},
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 80}}},
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 10}}},
};

/**
 * Statistics collected for every frame, such as the processing time and inference accuracy
 */
typedef struct {
    int prev_id;  // codec ID of the last frame entering update_stats()
    float ping_ms;
    float ping_ms_avg;
    int init_nsamples[NUM_CONFIGS];
    int init_nruns[NUM_CONFIGS];
    float init_ping_ms_avg;
    float init_latency_ms[NUM_CONFIGS];
    int cur_nsamples;
    float cur_latency_ms;
} codec_stats_t;

/**
 * Stores all data required to perform codec selection
 */
typedef struct {
    pthread_mutex_t lock;
    codec_stats_t stats;
    int id;      // the currently active codec; points at CONFIGS
    int64_t since_last_select_ms;
    int64_t last_timestamp_ns;
    bool is_calibrating;
    float tgt_latency_ms;  // latency target to aim for
    int init_sorted_ids[NUM_CONFIGS];  // IDs of init_latency_ms sorted by latency
    bool is_allowed[NUM_CONFIGS];  // If the codec is allowed to be used or not
} codec_select_state_t;

void init_codec_select(codec_select_state_t **state);

void destroy_codec_select(codec_select_state_t **state);

void update_stats(const frame_metadata_t *frame_metadata, codec_select_state_t *state);

void
select_codec_manual(device_type_enum device_index, int do_segment, compression_t compression_type,
                    int quality, int rotation, codec_config_t *state);

void select_codec_auto(codec_select_state_t *state);

int get_codec_id(codec_select_state_t *state);
codec_params_t get_codec_params(codec_select_state_t *state);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_CODEC_SELECT_H
