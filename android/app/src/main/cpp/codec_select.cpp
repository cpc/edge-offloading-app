//
// Created by zadnik on 6.5.2024.
//

#include "codec_select.h"
#include "jpeg_compression.h"
#include "platform.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


void init_codec_select(selected_codec_t **codec, frame_stats_t **stats) {
    selected_codec_t *new_codec = (selected_codec_t *) calloc(1, sizeof(selected_codec_t));
    pthread_mutex_init(&new_codec->lock, NULL);
    new_codec->config.compression_type = NO_COMPRESSION;
    new_codec->config.device_type = LOCAL_DEVICE;
    new_codec->config.rotation = 90;
    new_codec->config.do_segment = 1;
    new_codec->config.config.jpeg.quality = 80;
    new_codec->id = 0;
    *codec = new_codec;

    frame_stats_t *new_stats = (frame_stats_t *) calloc(1, sizeof(frame_stats_t));
    pthread_mutex_init(&new_stats->lock, NULL);
    new_stats->test = 0;
    *stats = new_stats;
}

void destroy_codec_select(selected_codec_t **codec, frame_stats_t **stats) {
    if (*codec != NULL) {
        pthread_mutex_destroy(&(*codec)->lock);
        free(*codec);
        *codec = NULL;
    }

    if (*stats != NULL) {
        pthread_mutex_destroy(&(*stats)->lock);
        free(*stats);
        *stats = NULL;
    }
}

void update_stats(const frame_metadata_t *frame_metadata, frame_stats_t *stats) {
    pthread_mutex_lock(&stats->lock);
    stats->test = frame_metadata->frame_index;
    pthread_mutex_unlock(&stats->lock);
}

void
select_codec_manual(device_type_enum device_index, int do_segment, compression_t compression_type,
                    int quality, int rotation, selected_codec_t *codec) {
    pthread_mutex_lock(&codec->lock);

    codec->config.compression_type = compression_type;
    codec->config.device_type = device_index;
    codec->config.rotation = rotation;
    codec->config.do_segment = do_segment;

    if (HEVC_COMPRESSION == compression_type ||
        SOFTWARE_HEVC_COMPRESSION == compression_type) {
        const int framerate = 5;
        codec->config.config.hevc.framerate = framerate;
        codec->config.config.hevc.i_frame_interval = 2;
        // heuristical map of the bitrate to quality parameter
        // (640 * 480 * (3 / 2) * 8 / (1 / framerate)) * (quality / 100)
        // equation can be simplied to equation below
        codec->config.config.hevc.bitrate = 36864 * framerate * quality;
    } else if (JPEG_COMPRESSION == compression_type) {
        codec->config.config.jpeg.quality = quality;
    }

    pthread_mutex_unlock(&codec->lock);
}

void
select_codec_auto(int do_segment, int rotation, frame_stats_t *stats, selected_codec_t *codec) {
    static device_type_enum device_id = LOCAL_DEVICE;

    // reading from stats
    pthread_mutex_lock(&stats->lock);

    // Switch local <-> remote every 10 frames
    if (stats->test % 10 >= 9) {
        if (device_id == LOCAL_DEVICE) {
            device_id = REMOTE_DEVICE;
        } else {
            device_id = LOCAL_DEVICE;
        }
    }

    int codec_id = stats->test;
    pthread_mutex_unlock(&stats->lock);

    // writing to codec config
    pthread_mutex_lock(&codec->lock);
    codec->id = codec_id;

    codec->config.compression_type = NO_COMPRESSION;
    codec->config.device_type = device_id;
    codec->config.rotation = rotation;
    codec->config.do_segment = do_segment;

    pthread_mutex_unlock(&codec->lock);
}

int get_codec_id(selected_codec_t *codec) {
    pthread_mutex_lock(&codec->lock);
    int id = codec->id;
    pthread_mutex_unlock(&codec->lock);
    return id;
}

codec_config_t get_codec_config(selected_codec_t *codec) {
    pthread_mutex_lock(&codec->lock);
    codec_config_t config = codec->config;
    pthread_mutex_unlock(&codec->lock);
    return config;
}

#ifdef __cplusplus
}
#endif
