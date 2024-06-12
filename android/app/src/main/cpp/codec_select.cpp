//
// Created by zadnik on 6.5.2024.
//

#include "codec_select.h"
#include "jpeg_compression.h"
#include "platform.h"

#include <assert.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// How often to perform codec selection
static const int64_t SELECT_INTERVAL_MS = 3000;

// Smoothing factor (higher means smoother)
static const float PING_ALPHA = 0.9f;

// How many calibration runs to do in the beginning (>= 1)
static const int NUM_CALIB_RUNS = 3;

// How many samples need to be collected before doing codec selection decision
static const int MIN_NSAMPLES = 2;

// Exponentially weighted moving average
static float ewma(float new_x, float old_x, float alpha) {
    return alpha * old_x + (1.0f - alpha) * new_x;
}

// Used for sorting
typedef struct {
    float val;
    int idx;
} indexed_float_t;

// Used for sorting
static int cmp(const void *a, const void *b) {
    indexed_float_t *aa = (indexed_float_t *) a;
    indexed_float_t *bb = (indexed_float_t *) b;
    if ((*aa).val > (*bb).val) {
        return -1;
    } else if ((*aa).val < (*bb).val) {
        return 1;
    } else {
        return 0;
    }
}

void init_codec_select(int config_flags, codec_select_state_t **state) {
    codec_select_state_t *new_state = (codec_select_state_t *) calloc(1,
                                                                      sizeof(codec_select_state_t));

    pthread_mutex_init(&new_state->lock, NULL);
    new_state->local_only = (config_flags & LOCAL_ONLY) != 0;
    new_state->is_calibrating = true; // start by calibrating
    new_state->stats.prev_id = -1;
    new_state->is_allowed[0] = true;  // always allow local device
    // ... the rest of new_state should be zeros

    *state = new_state;
}

void destroy_codec_select(codec_select_state_t **state) {
    if (*state != NULL) {
        pthread_mutex_destroy(&(*state)->lock);
        free(*state);
        *state = NULL;
    }
}

void update_stats(const frame_metadata_t *frame_metadata, codec_select_state_t *state) {
    assert(NULL != state);
    const float latency_ms =
            (float) (frame_metadata->host_ts_ns.stop - frame_metadata->host_ts_ns.start) / 1e6f;

    // TODO: Get ping from Java PingMonitor
    float ping_ms;
    if (frame_metadata->host_ts_ns.fill_ping_duration < 0) {
        // fill ping didn't finish before the processing
        ping_ms = latency_ms;
    } else {
        ping_ms = (float) (frame_metadata->host_ts_ns.fill_ping_duration) / 1e6f;
    }

    // Codec ID used for encoding the frame; might be different from state->id
    const int frame_codec_id = frame_metadata->codec.id;

    SLOGI(2, "SELECT | Update | frame %3d, codec %d, ping: %6.1f ms, latency: %6.1f ms",
          frame_metadata->frame_index, frame_codec_id, ping_ms, latency_ms);

    pthread_mutex_lock(&state->lock);

    codec_stats_t *stats = &state->stats;

    if (stats->prev_id != frame_codec_id) {
        stats->cur_nsamples = 0;
        stats->cur_latency_ms = 0.0f;

        if (state->is_calibrating) {
            stats->init_nruns[stats->prev_id] += 1;
        }

        SLOGI(2, "SELECT | Update | Codec changed from %d, not updating stats.", stats->prev_id);
        goto cleanup;
    }

    if (ping_ms > 0.0f) {
        stats->ping_ms_avg = ewma(ping_ms, state->stats.ping_ms, PING_ALPHA);
        stats->ping_ms = ping_ms;
    }

    if (state->is_calibrating) {
        // incremental averaging
        stats->init_nsamples[frame_codec_id] += 1;
        const float old_latency = stats->init_latency_ms[frame_codec_id];
        stats->init_latency_ms[frame_codec_id] +=
                (latency_ms - old_latency) / stats->init_nsamples[frame_codec_id];

        // average ping over the whole calibration period
        int total_nsamples = 0;
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            total_nsamples += stats->init_nsamples[i];
        }
        stats->init_ping_ms_avg += (ping_ms - stats->init_ping_ms_avg) / total_nsamples;

        SLOGI(2,
              "SELECT | Update | calibrating, nsamples %4d (total %4d), init avg latency %6.1f ms, init avg ping %6.1f ms",
              stats->init_nsamples[frame_codec_id], total_nsamples,
              stats->init_latency_ms[frame_codec_id], stats->init_ping_ms_avg);
    } else {
        stats->cur_nsamples += 1;
        const float old_latency = stats->cur_latency_ms;
        stats->cur_latency_ms += (latency_ms - old_latency) / stats->cur_nsamples;
    }

    cleanup:
    stats->prev_id = frame_codec_id;
    pthread_mutex_unlock(&state->lock);
}

void
select_codec_manual(device_type_enum device_index, int do_segment, compression_t compression_type,
                    int quality, int rotation, codec_config_t *config) {
    config->compression_type = compression_type;
    config->device_type = device_index;
    config->rotation = rotation;
    config->do_segment = do_segment;
    config->id = 0;  // unused in manual selection

    if (HEVC_COMPRESSION == compression_type || SOFTWARE_HEVC_COMPRESSION == compression_type) {
        const int framerate = 5;
        config->config.hevc.framerate = framerate;
        config->config.hevc.i_frame_interval = 2;
        // heuristical map of the bitrate to quality parameter
        // (640 * 480 * (3 / 2) * 8 / (1 / framerate)) * (quality / 100)
        // equation can be simplied to equation below
        config->config.hevc.bitrate = 36864 * framerate * quality;
    } else if (JPEG_COMPRESSION == compression_type) {
        config->config.jpeg.quality = quality;
    }
}

void select_codec_auto(codec_select_state_t *state) {
    struct timespec ts_start, ts_mid, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    const int64_t start_ns = ts_start.tv_sec * 1000000000 + ts_start.tv_nsec;

    pthread_mutex_lock(&state->lock);

    const int old_id = state->id;
    const codec_stats_t *stats = &state->stats;

    if (state->last_timestamp_ns == 0) {
        // skipping first frame
        state->last_timestamp_ns = start_ns;
        goto cleanup;
    }

    state->since_last_select_ms += (start_ns - state->last_timestamp_ns) / 1000000;
    state->last_timestamp_ns = start_ns;

    if (state->since_last_select_ms < SELECT_INTERVAL_MS) {
        goto cleanup;
    }

    state->since_last_select_ms = 0.0f;

    if (state->is_calibrating) {
        bool should_still_calibrate = false;
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            if (stats->init_nruns[i] < NUM_CALIB_RUNS) {
                should_still_calibrate = true;
                break;
            }
        }

        if (should_still_calibrate) {
            state->id = (old_id + 1) % NUM_CONFIGS;
            SLOGI(1, "SELECT | Calibrating (%d -> %d) ", old_id, state->id);
        } else {
            SLOGI(1, "SELECT | Calibration end. Avg ping: %6.1f ms", stats->init_ping_ms_avg);
            state->is_calibrating = false;

            float tgt_latency_ms = 0.0f;
            int nlat = 0;
            // sort only the remote latencies
            indexed_float_t sorted_latencies[NUM_CONFIGS - 1] = {};
            for (int i = 1; i < NUM_CONFIGS; ++i) {
                sorted_latencies[i - 1] = {.val = stats->init_latency_ms[i], .idx=i};

                if (!state->local_only && (stats->init_latency_ms[i] <= stats->init_latency_ms[0])) {
                    // only consider remote devices with latency <= local
                    nlat += 1;
                    tgt_latency_ms += (stats->init_latency_ms[i] - tgt_latency_ms) / nlat;
                    state->is_allowed[i] = true;
                }
            }

            // estimate an achievable target latency value to start with
            state->tgt_latency_ms = tgt_latency_ms;
//            state->tgt_latency_ms = stats->init_latency_ms[1] + 50;
            SLOGI(1, "SELECT | Target latency: %6.1f ms", state->tgt_latency_ms);

            // sort codecs by latency
            qsort(sorted_latencies, NUM_CONFIGS, sizeof(sorted_latencies[0]), cmp);

            state->init_sorted_ids[0] = 0;
            for (int i = 1; i < NUM_CONFIGS; ++i) {
                state->init_sorted_ids[i] = sorted_latencies[i - 1].idx;
            }

            // select first codec and print
            float prev_lat_ms = 99999.9f;
            float lat_ms;
            for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
                // iterate by descending init latency
                int id = state->init_sorted_ids[sort_id];
                lat_ms = stats->init_latency_ms[id];
                SLOGI(1, "SELECT | Codec %d, nsamples: %3d, avg latency: %6.1f ms, allowed: %d", id,
                      stats->init_nsamples[id], lat_ms, state->is_allowed[id]);

                if (prev_lat_ms > state->tgt_latency_ms && lat_ms < state->tgt_latency_ms &&
                    state->is_allowed[id]) {
                    state->id = id;
                }

                prev_lat_ms = lat_ms;
            }

            SLOGI(1, "SELECT | Calibration end. (%d -> %d)", old_id, state->id);
        }

    } else {
        if (stats->cur_nsamples < MIN_NSAMPLES) {
            SLOGI(1, "SELECT | Got only %d/%d samples. Not enough, skipping", stats->cur_nsamples,
                  MIN_NSAMPLES);
            goto cleanup;
        }

//        if (stats->cur_latency_ms > stats->init_latency_ms[0]) {
//            // latency higher than local -> switch to local
//            state->id = 0;
//            SLOGI(1, "SELECT | Latency %7.1f ms more than local, switching to local (%d -> %d)",
//                  stats->cur_latency_ms, old_id, state->id);
//            goto cleanup;
//        }

        int sort_id = 0;
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            if (state->init_sorted_ids[i] == state->id) {
                sort_id = i;
                break;
            }
        }

        int new_id = state->id;

        do {
            if ((sort_id < NUM_CONFIGS - 1) && stats->cur_latency_ms > state->tgt_latency_ms) {
                // latency too high -> increase compression
                sort_id += 1;
            } else if (sort_id > 0 && stats->cur_latency_ms < state->tgt_latency_ms &&
                       stats->init_latency_ms[sort_id - 1] < state->tgt_latency_ms) {
                // latency too low -> decrease compression but don't go above the target latency
                sort_id -= 1;
            } else {
                break;
            }

            new_id = state->init_sorted_ids[sort_id];
        } while (!state->is_allowed[new_id]);

        if (state->is_allowed[new_id]) {
            state->id = new_id;
        }

        // just prints
        if (stats->cur_latency_ms > state->tgt_latency_ms) {
            SLOGI(1, "SELECT | Latency %7.1f ms too high, %d -> %d", stats->cur_latency_ms, old_id,
                  state->id);
        } else {
            SLOGI(1, "SELECT | Latency %7.1f ms too low,  %d -> %d", stats->cur_latency_ms, old_id,
                  state->id);
        }

//        float err = stats->cur_latency_ms - state->tgt_latency_ms;
    }

    cleanup:
    pthread_mutex_unlock(&state->lock);
}

int get_codec_id(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    const int id = state->id;
    pthread_mutex_unlock(&state->lock);
    return id;
}

codec_params_t get_codec_params(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    assert(state->id >= 0 && state->id < NUM_CONFIGS);
    codec_params_t params = CONFIGS[state->id];
    pthread_mutex_unlock(&state->lock);
    return params;
}

#ifdef __cplusplus
}
#endif
