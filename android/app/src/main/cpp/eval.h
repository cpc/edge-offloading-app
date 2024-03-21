//
// Created by rabijl on 11/23/23.
//

#ifndef POCL_AISA_DEMO_EVAL_H
#define POCL_AISA_DEMO_EVAL_H

#include "platform.h"
#include "event_logger.h"
#include "poclImageProcessor.h"

#define QUALITY_VERBOSITY 2 // Verbosity of messages printed from this file (0 turns them off)

// Simple LOGI wrapper to reduce clutter
#define QLOGI(verbosity, ...) \
    if (verbosity <= QUALITY_VERBOSITY) { LOGI(__VA_ARGS__); }

// Number of configurations considered for each codec
#define NUM_JPEG_CONFIGS 2
#define NUM_HEVC_CONFIGS 2
#define NUM_CODEC_POINTS (2 + NUM_JPEG_CONFIGS + NUM_HEVC_CONFIGS)

// ID of the currently selected codec
extern int CUR_CODEC_ID;

// the below arrays are indexed by CUR_CODEC_ID
static const compression_t CODEC_POINTS[NUM_CODEC_POINTS] = {NO_COMPRESSION, NO_COMPRESSION,
                                                             JPEG_COMPRESSION, JPEG_COMPRESSION,
                                                             HEVC_COMPRESSION, HEVC_COMPRESSION};

static const device_type_enum CODEC_DEVICES[NUM_CODEC_POINTS] = {LOCAL_DEVICE, REMOTE_DEVICE, REMOTE_DEVICE, REMOTE_DEVICE, REMOTE_DEVICE, REMOTE_DEVICE};
static const int CODEC_CONFIGS[NUM_CODEC_POINTS] = {0, 0, 0, 1, 0, 1};

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

void init_eval();
void destroy_eval();

// Collect statistics about the loop execution and when it's the right time, decide which codec to run next
int evaluate_parameters(int frame_index, float power, float iou, uint64_t size_bytes,
                        int file_descriptor, int config_flags, const event_array_t *event_array,
                        const event_array_t *eval_event_array, const host_ts_ns_t *host_ts_ns,
                        codec_config_t *config);

int is_near_zero(float x);

#endif //POCL_AISA_DEMO_EVAL_H
