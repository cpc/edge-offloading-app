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

/**
 * Statistics collected for every frame, such as the processing time and inference accuracy
 */
typedef struct {
    pthread_mutex_t lock;
    int test;
} frame_stats_t;

/**
 * Currently selected codec config with a lock and its ID
 */
typedef struct {
    pthread_mutex_t lock;
    codec_config_t config;
    int id;  // ID pointing at array of all available codec configs
} selected_codec_t;

void init_codec_select(selected_codec_t **codec, frame_stats_t **stats);

void destroy_codec_select(selected_codec_t **codec, frame_stats_t **stats);

void update_stats(const frame_metadata_t *frame_metadata, frame_stats_t *stats);

void
select_codec_manual(device_type_enum device_index, int do_segment, compression_t compression_type,
                    int quality, int rotation, selected_codec_t *codec);

void select_codec_auto(int do_segment, int rotation, frame_stats_t *stats, selected_codec_t *codec);

int get_codec_id(selected_codec_t *codec);
codec_config_t get_codec_config(selected_codec_t *codec);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_CODEC_SELECT_H
