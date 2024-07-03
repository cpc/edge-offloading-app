//
// Created by zadnik on 6.5.2024.
//

#include "sharedUtils.h"
#include "codec_select.h"
#include "jpeg_compression.h"
#include "platform.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// How often to perform codec selection
static const int64_t SELECT_INTERVAL_MS = 2000;

// Smoothing factors (higher means smoother)
static const float PING_ALPHA = 0.9f;
static const float IOU_ALPHA = 0.9f;
static const float EXTERNAL_ALPHA = 0.9f;

// How many calibration runs to do in the beginning (>= 1)
static const int NUM_CALIB_RUNS = 3;

// How many IoU eval samples per codec are required for finishing the calibration
static const int NUM_CALIB_IOU_SAMPLES = 2;

// How many samples need to be collected before doing codec selection decision
static const int MIN_NSAMPLES = 2;

// Empty initializers

static const kernel_times_ms_t EMPTY_KERNEL_TIMES = {.enc = 0.0f, .dec = 0.0f, .dnn = 0.0f, .postprocess = 0.0f, .reconstruct = 0.0f,};

static const latency_data_t EMPTY_LATENCY_DATA = {.kernel_times_ms = EMPTY_KERNEL_TIMES, .total_ms = 0.0f, .network_ms = 0.0f, .size_bytes = 0.0f, .size_bytes_log10 = 0.0f, .fps = 0.0f};

static const int EMPTY_EXT_POW = -1;
static const int EMPTY_EXT_PING = -1.0f;

// the worst value ever that is worse than every value
static const float WORST_VAL = -999999.999999f;

// Exponentially weighted moving average
static void ewma(float alpha, float new_x, float *old_x) {
    *old_x = alpha * (*old_x) + (1.0f - alpha) * new_x;
}

// Used for sorting metrics by their product
static int cmp_metrics(const void *a, const void *b) {
    indexed_metrics_t *aa = (indexed_metrics_t *) a;
    indexed_metrics_t *bb = (indexed_metrics_t *) b;

    if (aa->all_fit_constraints && !bb->all_fit_constraints) {
        return 1;
    } else if (!aa->all_fit_constraints && bb->all_fit_constraints) {
        return -1;
    } else {
        if (aa->product > bb->product) {
            return 1;
        } else if (aa->product < bb->product) {
            return -1;
        } else {
            return 0;
        }
    }
}

// incremental averaging
static void incr_avg(float new_val, float *avg, int *nsamples) {
    *nsamples += 1;

    if (*nsamples == 1) {
        // protect against a case when avg is non-zero for some reason (e.g., leftover previous value)
        *avg = new_val;
    } else {
        *avg += (new_val - *avg) / (float) (*nsamples);
    }
}

// incremental average without updating the sample count
static void incr_avg_noup(float new_val, float *avg, int nsamples) {
    nsamples += 1;

    if (nsamples == 1) {
        // protect against a case when avg is non-zero for some reason (e.g., leftover previous value)
        *avg = new_val;
    } else {
        *avg += (new_val - *avg) / (float) (nsamples);
    }
}

static bool is_within_constraint(float val, constraint_t constraint) {
    if (constraint.type == CONSTR_SOFT) {
        return true;
    }

    switch (constraint.optimization) {
        case OPT_MIN:
            return val <= constraint.limit;
        case OPT_MAX:
            return val >= constraint.limit;
    }
}

static metric_t to_metric(int i) {
    return (metric_t) (i);
}

static void
metrics_product_custom(const codec_select_state_t *const state,
                       const constraint_t constraints[NUM_CONSTRAINTS],
                       indexed_metrics_t *metrics) {
    const int codec_id = metrics->codec_id;

    if (!state->is_allowed[codec_id]) {
        metrics->product = 0.0f;
        metrics->all_fit_constraints = false;
        for (int i = 0; i < NUM_METRICS; ++i) {
            metrics->violated_constraint[i] = 0;
        }
        return;
    }

    float product = 1.0f;
    bool all_fit_constraints = true;
    int violated_constraint[NUM_METRICS];
    for (int i = 0; i < NUM_METRICS; ++i) {
        violated_constraint[i] = -1;
    }

    for (int i = 0; i < NUM_METRICS; ++i) {
        const metric_t metric = to_metric(i);
        const float val = metrics->vals[i];

        for (int j = 0; j < NUM_CONSTRAINTS; ++j) {
            const constraint_t constraint = constraints[j];

            if (constraint.metric != metric) {
                continue;
            }

            if (!is_within_constraint(val, constraint)) {
                all_fit_constraints = false;
                violated_constraint[i] = j;
            }

            if (constraint.type == CONSTR_SOFT) {
                const float prod = val * constraint.scale;

                switch (constraint.optimization) {
                    case OPT_MIN: {
                        if (prod == 0.0f) {
                            product = 0.0f;
                        } else {
                            product /= prod;
                        }
                        break;
                    }
                    case OPT_MAX: {
                        product *= prod;
                        break;
                    }
                }

                // avoid multiplying the product twice if more constraints are defined for one metric
                break;
            }
        }
    }

    for (int i = 0; i < NUM_METRICS; ++i) {
        metrics->violated_constraint[i] = violated_constraint[i];
    }
    metrics->all_fit_constraints = all_fit_constraints;
    metrics->product = product;
}

static void metrics_product(const codec_select_state_t *const state, indexed_metrics_t *metrics) {
    return metrics_product_custom(state, state->constraints, metrics);
}

static void populate_vals(float latency_ms, float size_bytes, float power_w, float iou,
                          float vals[NUM_METRICS]) {
    vals[METRIC_LATENCY_MS] = latency_ms;
    vals[METRIC_SIZE_BYTES] = size_bytes;
    vals[METRIC_SIZE_BYTES_LOG10] = log10f(size_bytes);
    vals[METRIC_POWER_W] = power_w;
    vals[METRIC_IOU] = iou;
}

static indexed_metrics_t
populate_metrics(const codec_select_state_t *const state, int codec_id, float latency_ms,
                 float size_bytes, float power_w, float iou) {
    indexed_metrics_t metrics;

    populate_vals(latency_ms, size_bytes, power_w, iou, metrics.vals);
    metrics.codec_id = codec_id;

    metrics_product(state, &metrics);

    return metrics;
}

static indexed_metrics_t
populate_init_metrics(const codec_select_state_t *const state, int codec_id) {
    const float latency_ms = state->stats.init_latency_data[codec_id].total_ms;
    const float size_bytes = state->stats.init_latency_data[codec_id].size_bytes;
    const float power_w = state->stats.external_data->init_pow_w[codec_id];
    const float iou = state->stats.eval_data->init_iou[codec_id];

    return populate_metrics(state, codec_id, latency_ms, size_bytes, power_w, iou);
}

static void sort_init_metrics(const codec_select_state_t *const state,
                              indexed_metrics_t sorted_metrics[NUM_CONFIGS]) {
    for (int id = 0; id < NUM_CONFIGS; ++id) {
        sorted_metrics[id] = populate_init_metrics(state, id);
    }

    qsort(sorted_metrics, NUM_CONFIGS, sizeof(sorted_metrics[0]), cmp_metrics);
}

static void
log_constraints(const codec_select_state_t *state, const constraint_t constraints[NUM_CONSTRAINTS],
                int frame_id) {
    if (state->enable_profiling) {
        for (int i = 0; i < NUM_CONSTRAINTS; ++i) {
            constraint_t constr = constraints[i];
            char tag[32];
            sprintf(tag, "cs_constr%02d", i);
            log_frame_str(state->fd, frame_id, tag, "metric", METRIC_NAMES[constr.metric]);
            log_frame_str(state->fd, frame_id, tag, "opt", OPT_NAMES[constr.optimization]);
            log_frame_str(state->fd, frame_id, tag, "type", CONSTR_TYPE_NAMES[constr.type]);
            log_frame_f(state->fd, frame_id, tag, "limit", constr.limit);
            log_frame_f(state->fd, frame_id, tag, "scale", constr.scale);
        }
    }
}

static void print_constraints(const constraint_t constraints[NUM_CONSTRAINTS]) {
    if (SELECT_VERBOSITY > 0) {
        for (int i = 0; i < NUM_CONSTRAINTS; ++i) {
            constraint_t constr = constraints[i];
            if (constr.type == CONSTR_HARD) {
                SLOGI(2, "SELECT | Constraints | %16s: %8.3f (scale %11.6f)",
                      METRIC_NAMES[constr.metric], constr.limit, constr.scale);
            }
        }
    }
}

// returns 1 if better, -1 if worse, 0 if same
static int cmp_vals(float cur_val, float best_val, optimization_t optimization) {
    if (best_val == WORST_VAL) {
        return 1;
    }

    if (cur_val == best_val) {
        return 0;
    }

    switch (optimization) {
        case OPT_MIN:
            return cur_val < best_val ? 1 : -1;
        case OPT_MAX:
            return cur_val > best_val ? 1 : -1;
    }
}

static int select_codec(const codec_select_state_t *const state,
                        const constraint_t constraints[NUM_CONSTRAINTS],
                        const indexed_metrics_t sorted_metrics[NUM_CONFIGS]) {
    int new_codec_id = LOCAL_CODEC_ID;  // default to local
//    int max_product_codec_id = LOCAL_CODEC_ID;
//    bool all_fit_constraints[NUM_CONFIGS];
    bool some_fit_constraints = false;
    int old_id = state->id;

    // print kernel time breakdown

    if (state->is_calibrating) {
        for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
            const indexed_metrics_t metrics = sorted_metrics[sort_id];
            const int id = metrics.codec_id;

            SLOGI(2,
                  "SELECT | Select | Codec %2d, size: %7.0f B, latency: total %8.3f, network %8.3f, enc %8.3f, dec %8.3f, dnn %8.3f, postprocess %8.3f, reconstruct %8.3f",
                  id, state->stats.init_latency_data[id].size_bytes,
                  state->stats.init_latency_data[id].total_ms,
                  state->stats.init_latency_data[id].network_ms,
                  state->stats.init_latency_data[id].kernel_times_ms.enc,
                  state->stats.init_latency_data[id].kernel_times_ms.dec,
                  state->stats.init_latency_data[id].kernel_times_ms.dnn,
                  state->stats.init_latency_data[id].kernel_times_ms.postprocess,
                  state->stats.init_latency_data[id].kernel_times_ms.reconstruct);
        }
    } else {
        SLOGI(2,
              "SELECT | Select | Codec %2d, size: %7.0f B, latency: total %8.3f, network %8.3f, enc %8.3f, dec %8.3f, dnn %8.3f, postprocess %8.3f, reconstruct %8.3f",
              old_id, state->stats.init_latency_data[old_id].size_bytes,
              state->stats.init_latency_data[old_id].total_ms,
              state->stats.init_latency_data[old_id].network_ms,
              state->stats.init_latency_data[old_id].kernel_times_ms.enc,
              state->stats.init_latency_data[old_id].kernel_times_ms.dec,
              state->stats.init_latency_data[old_id].kernel_times_ms.dnn,
              state->stats.init_latency_data[old_id].kernel_times_ms.postprocess,
              state->stats.init_latency_data[old_id].kernel_times_ms.reconstruct);
    }

//    float net_ms = 0.0f;
//    float bw_kBps = 0.0f;
//
//    if (old_id != LOCAL_CODEC_ID) {
//        net_ms = fmax(0.0f, state->stats.cur_latency_data.network_ms - state->stats.ping_ms_avg);
//        if (net_ms > 0.0f) {
//            bw_kBps = log10f(state->stats.cur_latency_data.size_bytes) / net_ms;
//        }
//    }

//    SLOGI(2, "SELECT | DEBUG | Old Codec: %2d, BW log: %7.1f kBps, ping %6.1f ms, net_ms %6.1f ms",
//          state->id, bw_kBps, state->stats.ping_ms_avg, net_ms);

    for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
        // iterate by ascending the init product
        const indexed_metrics_t metrics = sorted_metrics[sort_id];
        const int id = metrics.codec_id;
        const bool fits_constraints = metrics.all_fit_constraints;
//        all_fit_constraints[sort_id] = metrics.all_fit_constraints;

        some_fit_constraints |= fits_constraints;
//        max_product_codec_id = id;

//        float proj_lat_ms = 0.0f;
//        indexed_metrics_t proj_metrics = metrics;
//        if (bw_kBps > 0.0f) {
//            proj_lat_ms = log10f(state->stats.init_latency_data[id].size_bytes) / bw_kBps;
//        }
//        proj_metrics.vals[METRIC_LATENCY_MS] = proj_lat_ms;
//        metrics_product(state, &proj_metrics);

        if (state->is_calibrating) {
            SLOGI(2,
                  "SELECT | Select | Codec %2d, nsamples: %3d, size %7.0f B, avg fps %5.1f, latency %6.1f ms, iou %5.3f, pow %5.3f W, allowed: %d, fits: %d, prod %8.5f",
                  id, state->stats.init_nsamples[id], state->stats.init_latency_data[id].size_bytes,
                  state->stats.init_latency_data[id].fps, metrics.vals[METRIC_LATENCY_MS],
                  metrics.vals[METRIC_IOU], metrics.vals[METRIC_POWER_W],
                  state->is_allowed[metrics.codec_id], fits_constraints, metrics.product);
        } else {
            SLOGI(2,
                  "SELECT | Select | Codec %2d, size %7.0f B, avg fps %5.1f, latency %6.1f  ms, iou %5.3f, pow %5.3f W, fits: %d, prod %8.5f",
                  metrics.codec_id, state->stats.init_latency_data[id].size_bytes,
                  state->stats.init_latency_data[id].fps, metrics.vals[METRIC_LATENCY_MS],
                  metrics.vals[METRIC_IOU], metrics.vals[METRIC_POWER_W], fits_constraints,
                  metrics.product);
        }


        if (state->enable_profiling) {
            char tag[32];
            sprintf(tag, "cs_select_codec%02d", id);

            log_frame_int(state->fd, state->last_frame_id, tag, "rank", sort_id);
            log_frame_int(state->fd, state->last_frame_id, tag, "codec_id", id);
            log_frame_int(state->fd, state->last_frame_id, tag, "allowed",
                          state->is_allowed[id] ? 1 : 0);
            log_frame_int(state->fd, state->last_frame_id, tag, "fits", fits_constraints ? 1 : 0);

            for (int i = 0; i < NUM_METRICS; ++i) {
                log_frame_f(state->fd, state->last_frame_id, tag, METRIC_NAMES[i], metrics.vals[i]);
            }

            log_frame_f(state->fd, state->last_frame_id, tag, "product", metrics.product);
        }

        if (metrics.product == 0.0f || !fits_constraints) {
            continue;
        }

        new_codec_id = id;
    }

    if (!some_fit_constraints) {
        bool is_constr_violated[NUM_CONSTRAINTS];

        for (int metric_id = 0; metric_id < NUM_METRICS; ++metric_id) {
            bool all_codecs_violate_this_metrics_constraint = true;

            for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
                const indexed_metrics_t metrics = sorted_metrics[sort_id];

                if (metrics.violated_constraint[metric_id] < 0) {
                    all_codecs_violate_this_metrics_constraint = false;
                }
            }

            if (all_codecs_violate_this_metrics_constraint) {
                for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
                    if (constraints[constr_id].metric == metric_id) {
                        is_constr_violated[constr_id] = true;
                        break;
                    }
                }
            }
        }

        int violated_constr_ids[NUM_METRICS];  // -1 terminated array
        int nviolated = 0;

        for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
            if (is_constr_violated[constr_id]) {
                violated_constr_ids[nviolated] = constr_id;
            } else {
                violated_constr_ids[nviolated] = -1;
            }

            nviolated += 1;
        }

        assert(nviolated >= 1);

        // Select codec with the best value of the violated constraint metric
        int best_codec_id = 0;
        float best_val[NUM_METRICS];
        for (int i = 0; i < NUM_METRICS; ++i) {
            best_val[i] = WORST_VAL;
        }

        for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
            const indexed_metrics_t metrics = sorted_metrics[sort_id];

            int violated_constr_id = 0;
            constraint_t violated_constraint = constraints[violated_constr_id];
            float cur_val = metrics.vals[violated_constraint.metric];

            int cmp_res = cmp_vals(cur_val, best_val[violated_constraint.metric],
                                   violated_constraint.optimization);
            SLOGI(4,
                  "SELECT | Select | All Fail Cmp |   Codec %2d, metric %16s, curval %11.3f, bestval %13.3f, cmp %2d",
                  metrics.codec_id, METRIC_NAMES[violated_constraint.metric], cur_val,
                  best_val[violated_constraint.metric], cmp_res);

            while (cmp_res == 0 && violated_constr_id < nviolated - 1) {
                violated_constr_id += 1;
                violated_constraint = constraints[violated_constr_id];
                cur_val = metrics.vals[violated_constraint.metric];
                cmp_res = cmp_vals(cur_val, best_val[violated_constraint.metric],
                                   violated_constraint.optimization);

                SLOGI(4,
                      "SELECT | Select | All Fail Cmp | - Codec %2d, metric %16s, curval %11.3f, bestval %13.3f, cmp %2d",
                      metrics.codec_id, METRIC_NAMES[violated_constraint.metric], cur_val,
                      best_val[violated_constraint.metric], cmp_res);
            }

            if (cmp_res == 0) {
                SLOGI(2, "SELECT | Select | DEBUG | Codec %2d selecting by product",
                      metrics.codec_id);
                if (metrics.product > sorted_metrics[best_codec_id].product) {
                    best_codec_id = metrics.codec_id;
                    for (int i = 0; i < NUM_METRICS; ++i) {
                        best_val[i] = metrics.vals[i];
                    }
                }
            } else if (cmp_res > 0) {
                best_codec_id = metrics.codec_id;
                for (int i = 0; i < NUM_METRICS; ++i) {
                    best_val[i] = metrics.vals[i];
                }
            }
        }

        SLOGI(1, "SELECT | Select | No codec fits constraints, selecting %d.", best_codec_id);
        new_codec_id = best_codec_id;
    }

    if (state->enable_profiling) {
        log_frame_int(state->fd, state->last_frame_id, "cs_select", "new_codec_id", new_codec_id);
    }

    return new_codec_id;
}

static void
collect_latency(const codec_select_state_t *state, const frame_metadata_t *frame_metadata,
                int64_t received_frame_ts_ns, bool do_average, codec_stats_t *stats) {

    SLOGI(4, "SELECT | Host Times | Frame %d | start %7.2f ms\n", frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_enc - frame_metadata->host_ts_ns.start) / 1e6f);
    SLOGI(4, "SELECT | Host Times | Frame %d | enc   %7.2f ms\n", frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_dnn - frame_metadata->host_ts_ns.before_enc) / 1e6f);
    SLOGI(4, "SELECT | Host Times | Frame %d | dnn   %7.2f ms\n", frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_wait - frame_metadata->host_ts_ns.before_dnn) / 1e6f);
    SLOGI(4, "SELECT | Host Times | Frame %d | wait  %7.2f ms\n", frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.after_wait - frame_metadata->host_ts_ns.before_wait) / 1e6f);
    SLOGI(4, "SELECT | Host Times | Frame %d | end   %7.2f ms\n", frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.stop - frame_metadata->host_ts_ns.after_wait) / 1e6f);

    kernel_times_ms_t kernel_times_ms = EMPTY_KERNEL_TIMES;
    find_event_time("enc_event", state->collected_events, &kernel_times_ms.enc);
    find_event_time("dec_event", state->collected_events, &kernel_times_ms.dec);
    find_event_time("dnn_event", state->collected_events, &kernel_times_ms.dnn);
    find_event_time("postprocess_event", state->collected_events, &kernel_times_ms.postprocess);
    find_event_time("reconstruct_event", state->collected_events, &kernel_times_ms.reconstruct);

    const float total_ms =
            (float) (frame_metadata->host_ts_ns.stop - frame_metadata->host_ts_ns.start) / 1e6f;
    float network_ms = 0.0f;
    if (frame_metadata->codec.id != LOCAL_CODEC_ID) {
        network_ms = fmax(0.0f, total_ms - kernel_times_ms.enc - kernel_times_ms.dec -
                                kernel_times_ms.dnn - kernel_times_ms.postprocess -
                                kernel_times_ms.reconstruct);
    }

    const float size_bytes = (float) (frame_metadata->size_bytes_tx +
                                      frame_metadata->size_bytes_rx);
    const float fps = 1e9f / (float) (received_frame_ts_ns - stats->last_received_image_ts_ns);
    const int frame_index = frame_metadata->frame_index;

    if (state->enable_profiling) {
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_enc_ms",
                    kernel_times_ms.enc);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_dec_ms",
                    kernel_times_ms.dec);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_dnn_ms",
                    kernel_times_ms.dnn);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_postprocess_ms",
                    kernel_times_ms.postprocess);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_reconstruct_ms",
                    kernel_times_ms.reconstruct);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "total_ms", total_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "size_bytes", size_bytes);
    }

    if (do_average) {
        latency_data_t *data;
        int *nsamples;

        if (state->is_calibrating) {
            data = &stats->init_latency_data[frame_metadata->codec.id];
            nsamples = &stats->init_nsamples[frame_metadata->codec.id];
        } else {
            data = &stats->cur_latency_data;
            nsamples = &stats->cur_nsamples;
        }

        incr_avg_noup(kernel_times_ms.enc, &data->kernel_times_ms.enc, *nsamples);
        incr_avg_noup(kernel_times_ms.dec, &data->kernel_times_ms.dec, *nsamples);
        incr_avg_noup(kernel_times_ms.dnn, &data->kernel_times_ms.dnn, *nsamples);
        incr_avg_noup(kernel_times_ms.postprocess, &data->kernel_times_ms.postprocess, *nsamples);
        incr_avg_noup(kernel_times_ms.reconstruct, &data->kernel_times_ms.reconstruct, *nsamples);
        incr_avg_noup(total_ms, &data->total_ms, *nsamples);
        incr_avg_noup(network_ms, &data->network_ms, *nsamples);
        incr_avg_noup(size_bytes, &data->size_bytes, *nsamples);
        incr_avg_noup(fps, &data->fps, *nsamples);
        *nsamples += 1;

        if (state->enable_profiling) {
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_enc_ms",
                        data->kernel_times_ms.enc);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_dec_ms",
                        data->kernel_times_ms.dec);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_dnn_ms",
                        data->kernel_times_ms.dnn);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_postprocess_ms",
                        data->kernel_times_ms.postprocess);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_reconstruct_ms",
                        data->kernel_times_ms.reconstruct);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_total_ms",
                        data->total_ms);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_size_bytes",
                        data->size_bytes);
            log_frame_f(state->fd, frame_index, "cs_update_latency", "nsamples", *nsamples);
        }
    }
}

static void collect_external_data(const codec_select_state_t *state, int frame_codec_id,
                                  int64_t frame_stop_ts_ns, external_data_t *data) {
    SLOGI(4, "SELECT | Collect External | Frame codec %2d, stop ts: %ld, prev stop ts: %ld",
          frame_codec_id, frame_stop_ts_ns, data->prev_frame_stop_ts_ns);
    while (data->istart < data->inext) {
        const int istart = data->istart % NUM_EXTERNAL_SAMPLES;
        const int64_t ts_ns = data->ts_ns[istart];
        SLOGI(4, "SELECT | Collect External | -- istart %4d, ts %ld", istart, ts_ns);

        if (ts_ns != 0 && ts_ns >= data->prev_frame_stop_ts_ns && ts_ns <= frame_stop_ts_ns) {
            const int amp = data->ext_amp[istart];
            const int volt = data->ext_volt[istart];
            const float pow_w = (float) (-amp) * (float) (volt) / 1e6f;
            const float ping_ms = data->ext_ping_ms[istart];

            SLOGI(4,
                  "SELECT | Collect External | -- within; amp: %4d, volt: %4d, pow_w: %5.3f, ping_ms: %7.1f",
                  amp, volt, pow_w, ping_ms);

            float value;
            float *avg_value;
            int *nsamples;
            float alpha;

            if (ping_ms != EMPTY_EXT_PING) {
                value = ping_ms;
                alpha = PING_ALPHA;
                if (state->is_calibrating) {
                    avg_value = &data->init_ping_ms[frame_codec_id];
                    nsamples = &data->init_ping_ms_nsamples[frame_codec_id];
                } else {
                    avg_value = &data->ping_ms[frame_codec_id];
                    nsamples = &data->ping_ms_nsamples[frame_codec_id];
                }
            } else if (amp != EMPTY_EXT_POW && volt != EMPTY_EXT_POW) {
                value = pow_w;
                alpha = EXTERNAL_ALPHA;
                if (state->is_calibrating) {
                    avg_value = &data->init_pow_w[frame_codec_id];
                    nsamples = &data->init_pow_w_nsamples[frame_codec_id];
                } else {
                    avg_value = &data->pow_w[frame_codec_id];
                    nsamples = &data->pow_w_nsamples[frame_codec_id];
                }
            } else {
                LOGE("SELECT | Collect External | ERROR: Got completely empty value for ts %ld",
                     ts_ns);
                data->ts_ns[istart] = 0;
                data->istart += 1;
                continue;
            }

            incr_avg(value, avg_value, nsamples);

            if (state->enable_profiling) {
                char param_val[32];
                char param_avg_val[32];
                char param_nsamples[32];

                if (ping_ms != EMPTY_EXT_PING) {
                    strcpy(param_val, "ping_ms");
                    strcpy(param_avg_val, "avg_ping_ms");
                    strcpy(param_nsamples, "nsamples_ping_ms");
                } else {
                    strcpy(param_val, "pow_w");
                    strcpy(param_avg_val, "avg_pow_w");
                    strcpy(param_nsamples, "nsamples_pow_w");
                }

                log_frame_i64(state->fd, state->last_frame_id, "cs_update_external", "ts_ns",
                              ts_ns);
                log_frame_f(state->fd, state->last_frame_id, "cs_update_external", param_val,
                            value);
                log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                            param_avg_val, *avg_value);
                log_frame_int(state->fd, state->last_frame_id, "cs_update_external",
                              param_nsamples, *nsamples);
            }
        }

        data->ts_ns[istart] = 0;
        data->istart += 1;
    }
}

void init_codec_select(int config_flags, int fd, codec_select_state_t **state) {
    codec_select_state_t *new_state = (codec_select_state_t *) calloc(1,
                                                                      sizeof(codec_select_state_t));

    eval_data_t *eval_data = (eval_data_t *) calloc(1, sizeof(eval_data_t));
    external_data_t *external_data = (external_data_t *) calloc(1, sizeof(external_data_t));
    collected_events_t *collected_events = (collected_events_t *) calloc(1,
                                                                         sizeof(collected_events_t));

    srand(time(NULL));

    pthread_mutex_init(&new_state->lock, NULL);

    new_state->collected_events = collected_events;

    new_state->local_only = (config_flags & LOCAL_ONLY) != 0;
    for (int id = 0; id < NUM_CONFIGS; ++id) {
        if (id == LOCAL_CODEC_ID) {
            // always allow local device
            new_state->is_allowed[id] = true;
            continue;
        }

        // all remote codecs allowed, unless running local-only
        new_state->is_allowed[id] = !new_state->local_only;
    }
    new_state->is_calibrating = !new_state->local_only; // there is nothing to calibrate in local-only
    new_state->enable_profiling = ENABLE_PROFILING & config_flags;
    new_state->fd = fd;

    int nconstr = 0;
    new_state->constraints[nconstr++] = {.metric = METRIC_LATENCY_MS, .optimization = OPT_MIN, .type = CONSTR_HARD, .limit = 200, .scale = LATENCY_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_IOU, .optimization = OPT_MAX, .type = CONSTR_HARD, .limit = 0.5f, .scale = IOU_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_LATENCY_MS, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = LATENCY_SCALE};
//    new_state->constraints[nconstr++] = {.metric = METRIC_SIZE_BYTES, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = SIZE_SCALE};
//    new_state->constraints[nconstr++] = {.metric = METRIC_SIZE_BYTES_LOG10, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = SIZE_SCALE_LOG10};
    new_state->constraints[nconstr++] = {.metric = METRIC_POWER_W, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = POWER_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_IOU, .optimization = OPT_MAX, .type = CONSTR_SOFT, .scale = IOU_SCALE};
    assert(nconstr == NUM_CONSTRAINTS);

    new_state->stats.eval_data = eval_data;
    new_state->stats.external_data = external_data;
    new_state->stats.prev_id = -1;
    // local-only always max. quality
    new_state->stats.eval_data->init_iou[LOCAL_CODEC_ID] = 1.0f;
    new_state->stats.eval_data->iou[LOCAL_CODEC_ID] = 1.0f;
    // ... the rest of new_state should be zeros

    log_constraints(new_state, new_state->constraints, -1);

    *state = new_state;
}

void destroy_codec_select(codec_select_state_t **state) {
    if (*state != NULL) {
        pthread_mutex_destroy(&(*state)->lock);
        free((*state)->stats.eval_data);
        free((*state)->stats.external_data);
        free((*state)->collected_events);
        free(*state);
        *state = NULL;
    }
}

void update_stats(const frame_metadata_t *frame_metadata, const eval_pipeline_context_t *eval_ctx,
                  codec_select_state_t *state) {
    assert(NULL != state);
    const int64_t received_frame_ts_ns = get_timestamp_ns();
    const int64_t frame_stop_ts_ns = frame_metadata->host_ts_ns.stop;
    const int frame_index = frame_metadata->frame_index;

    // Codec ID used for encoding the frame; might be different from state->id
    const int frame_codec_id = frame_metadata->codec.id;

    pthread_mutex_lock(&state->lock);

    state->last_frame_id = frame_index;

    codec_stats_t *stats = &state->stats;
    const bool should_skip = stats->prev_id != frame_codec_id;

    if (state->enable_profiling) {
        log_frame_i64(state->fd, frame_index, "cs_update", "start_ns", received_frame_ts_ns);
        log_frame_int(state->fd, frame_index, "cs_update", "frame_codec_id", frame_codec_id);
        log_frame_int(state->fd, frame_index, "cs_update", "is_calibrating",
                      state->is_calibrating ? 1 : 0);
        log_frame_int(state->fd, frame_index, "cs_update", "skip", should_skip);
    }

    // Collect latency (do not update averages if codec just changed)
    collect_latency(state, frame_metadata, received_frame_ts_ns, !should_skip, stats);

    // Collect power
    collect_external_data(state, frame_codec_id, frame_stop_ts_ns, stats->external_data);

    if (should_skip) {
        stats->cur_latency_data = EMPTY_LATENCY_DATA;
        stats->cur_nsamples = 0;

        if (state->is_calibrating) {
            stats->init_nruns[stats->prev_id] += 1;
        }

        SLOGI(3, "SELECT | Update | Frame %d, Codec changed from %2d, not updating stats.",
              frame_index, stats->prev_id);
        goto cleanup;
    }

//    SLOGI(1, "SELECT | Update | DEBUG | Frame %d, Eval %d, Codec %2d", frame_index,
//          frame_metadata->is_eval_frame, frame_metadata->codec.id);
//    for (int i = 0; i < state->collected_events->num_events; ++i) {
//        SLOGI(1, "SELECT | EVENT LOGGER | Event %2d: %30s %8.3f ms", i,
//              state->collected_events->descriptions[i], state->collected_events->end_start_ms[i]);
//
//    }

    if (state->is_calibrating) {
        int total_nsamples = 0;
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            total_nsamples += stats->init_nsamples[i];
        }

        SLOGI(3,
              "SELECT | Update | Frame %3d, codec %2d, nsamples %4d (total %4d), init avg: latency %6.1f ms, ping %6.1f ms, pow %5.3f (%d sampl.)",
              frame_index, frame_codec_id, stats->init_nsamples[frame_codec_id],
              total_nsamples, stats->init_latency_data[frame_codec_id].total_ms,
              stats->external_data->init_ping_ms[frame_codec_id],
              stats->external_data->init_pow_w[frame_codec_id],
              stats->external_data->init_pow_w_nsamples[frame_codec_id]);
    } else {
        SLOGI(3,
              "SELECT | Update | Frame %3d, codec %2d, ping: %6.1f ms, latency: %6.1f ms, pow %5.3f W",
              frame_index, frame_codec_id, stats->external_data->ping_ms[frame_codec_id],
              stats->init_latency_data[frame_codec_id].total_ms,
              stats->external_data->pow_w[frame_codec_id]);
    }

    cleanup:
    state->stats.last_received_image_ts_ns = received_frame_ts_ns;
    stats->external_data->prev_frame_stop_ts_ns = frame_stop_ts_ns;
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
    config->id = LOCAL_CODEC_ID;  // unused in manual selection

    if (HEVC_COMPRESSION == compression_type || SOFTWARE_HEVC_COMPRESSION == compression_type) {
        const int framerate = 5;
        config->config.hevc.framerate = framerate;
        config->config.hevc.i_frame_interval = 2;
        // heuristic map of the bitrate to quality parameter
        // (640 * 480 * (3 / 2) * 8 / (1 / framerate)) * (quality / 100)
        // equation can be simplied to equation below
        config->config.hevc.bitrate = 36864 * framerate * quality;
    } else if (JPEG_COMPRESSION == compression_type) {
        config->config.jpeg.quality = quality;
    }
}

void select_codec_auto(codec_select_state_t *state) {
    int64_t end_ns;
    const int64_t start_ns = get_timestamp_ns();

    pthread_mutex_lock(&state->lock);

    const int old_id = state->id;
    codec_stats_t *stats = &state->stats;

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

        for (int id = 0; id < NUM_CONFIGS; ++id) {
            if (stats->init_nruns[id] < NUM_CALIB_RUNS) {
                should_still_calibrate = true;
                break;
            }

            if (id > 0 && stats->eval_data->init_iou_nsamples[id] < NUM_CALIB_IOU_SAMPLES) {
                should_still_calibrate = true;
                break;
            }
        }

        if (should_still_calibrate) {
            // TODO: Shuffle configs to ensure every run runs each codec once
//            state->id = rand() % NUM_CONFIGS;

            if (stats->init_nsamples[old_id] > stats->init_nsamples_prev[old_id]) {
                state->id = (old_id + 1) % NUM_CONFIGS;
            }

            stats->init_nsamples_prev[old_id] = stats->init_nsamples[old_id];

            // just prints
            indexed_metrics_t metrics = populate_init_metrics(state, old_id);

            SLOGI(1,
                  "SELECT | Calibrating | (%2d -> %2d), nsamples: %3d, size: %7.0f B, avg latency %6.1f ms, fps %6.2f, iou %5.3f, pow %5.3f W, allowed: %d, fits: %d, prod: %8.5f",
                  old_id, state->id, stats->init_nsamples[old_id],
                  stats->init_latency_data[old_id].size_bytes,
                  stats->init_latency_data[old_id].total_ms,
                  stats->init_latency_data[old_id].fps,
                  state->stats.eval_data->init_iou[old_id],
                  state->stats.external_data->init_pow_w[old_id], state->is_allowed[old_id],
                  metrics.all_fit_constraints, metrics.product);
        } else {
            SLOGI(1, "SELECT | Calibrating | End");

            indexed_metrics_t sorted_metrics[NUM_CONFIGS];
            sort_init_metrics(state, sorted_metrics);

            for (int id = 0; id < NUM_CONFIGS; ++id) {
                assert(sorted_metrics[id].codec_id >= 0);
                state->init_sorted_ids[id] = sorted_metrics[id].codec_id;
            }

            print_constraints(state->constraints);
            state->id = select_codec(state, state->constraints, sorted_metrics);

            // populate stats with init values
            for (int id = 0; id < NUM_CONFIGS; ++id) {
                stats->eval_data->iou_nsamples[id] = stats->eval_data->init_iou_nsamples[id];
                stats->eval_data->iou[id] = stats->eval_data->init_iou[id];
                stats->external_data->pow_w_nsamples[id] = stats->external_data->init_pow_w_nsamples[id];
                stats->external_data->pow_w[id] = stats->external_data->init_pow_w[id];
                stats->external_data->ping_ms_nsamples[id] = stats->external_data->init_ping_ms_nsamples[id];
                stats->external_data->ping_ms[id] = stats->external_data->init_ping_ms[id];
            }

            state->is_calibrating = false;
            SLOGI(1, "SELECT | Calibrating | End | (%2d -> %2d)", old_id, state->id);
        }
    } else {
        if (stats->cur_nsamples < MIN_NSAMPLES) {
            SLOGI(1, "SELECT | Select | Got only %d/%d samples. Not enough, skipping",
                  stats->cur_nsamples, MIN_NSAMPLES);
            goto cleanup;
        }

// vvv New code vvv
        indexed_metrics_t sorted_metrics[NUM_CONFIGS];
        float cur_vals[NUM_METRICS];
        float init_vals[NUM_METRICS];
        float offsets[NUM_METRICS];

        populate_vals(stats->cur_latency_data.total_ms, stats->cur_latency_data.size_bytes,
                      stats->external_data->pow_w[old_id], stats->eval_data->iou[old_id], cur_vals);
        populate_vals(stats->init_latency_data[old_id].total_ms,
                      stats->init_latency_data[old_id].size_bytes,
                      stats->external_data->init_pow_w[old_id], stats->eval_data->init_iou[old_id],
                      init_vals);

        for (int metric_id = 0; metric_id < NUM_METRICS; ++metric_id) {
            offsets[metric_id] = cur_vals[metric_id] - init_vals[metric_id];
            SLOGI(2, "SELECT | Select | Metric  %16s (%d): cur %11.3f, init %11.3f, off %+11.3f",
                  METRIC_NAMES[metric_id], metric_id, cur_vals[metric_id], init_vals[metric_id],
                  offsets[metric_id]);
        }

        for (int id = 0; id < NUM_CONFIGS; ++id) {
            indexed_metrics_t new_init_metrics;
            new_init_metrics.codec_id = id;
            populate_vals(stats->init_latency_data[id].total_ms,
                          stats->init_latency_data[id].size_bytes,
                          stats->external_data->init_pow_w[id], stats->eval_data->init_iou[id],
                          new_init_metrics.vals);

            for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
                const constraint_t constraint = state->constraints[constr_id];
                const metric_t metric = constraint.metric;

                if (constraint.type == CONSTR_HARD) {
                    // TODO: Abstract this away, putting min/max values of IoU to constraint_t
                    float new_val = 0.0f;

                    switch (constraint.optimization) {
                        case OPT_MIN:
                            new_val = new_init_metrics.vals[metric] + offsets[metric];
                            break;
                        case OPT_MAX:
                            new_val = new_init_metrics.vals[metric] - offsets[metric];
                            break;
                    }

                    if (metric == METRIC_IOU) {
                        new_val = fmax(fmin(1.0f, new_val), 0.0f);
                    }

                    SLOGI(3,
                          "SELECT | Select | Codec %2d: Metric %d, init %11.3f, new val %11.3f, off %+11.3f",
                          id, metric, new_init_metrics.vals[metric], new_val, offsets[metric]);
                    new_init_metrics.vals[metric] = new_val;
                }

            }

            metrics_product(state, &new_init_metrics);
            sorted_metrics[id] = new_init_metrics;
        }

        print_constraints(state->constraints);
// ^^^ New code ^^^

// vvv Old code vvv
//        indexed_metrics_t sorted_metrics[NUM_CONFIGS];
//
//        for (int id = 0; id < NUM_CONFIGS; ++id) {
//            if (id == old_id) {
//                sorted_metrics[id] = populate_metrics(state, old_id,
//                                                      stats->cur_latency_data.total_ms,
//                                                      stats->cur_latency_data.size_bytes,
//                                                      stats->external_data->pow_w[old_id],
//                                                      stats->eval_data->iou[old_id]);
//            } else {
//                sorted_metrics[id] = populate_init_metrics(state, id);
//            }
//        }
//
//        // If violating constraints, disqualify all configs whose init vals are worse than the init
//        // vals of the current config.
//        constraint_t custom_constraints[NUM_CONSTRAINTS];
//        int nconstr = 0;
//
//        for (int i = 0; i < NUM_CONSTRAINTS; ++i) {
//            constraint_t constraint = state->constraints[i];
//
//            if (constraint.type == CONSTR_HARD) {
//                float cur_val;
//                float new_limit;
//
//                switch (constraint.metric) {
//                    case METRIC_LATENCY_MS:
//                        cur_val = stats->cur_latency_data.total_ms;
//                        new_limit = stats->init_latency_data[old_id].total_ms;
//                        break;
//                    case METRIC_SIZE_BYTES:
//                        cur_val = stats->cur_latency_data.size_bytes;
//                        new_limit = stats->init_latency_data[old_id].size_bytes;
//                        break;
//                    case METRIC_SIZE_BYTES_LOG10:
//                        cur_val = stats->cur_latency_data.size_bytes_log10;
//                        new_limit = stats->init_latency_data[old_id].size_bytes_log10;
//                        break;
//                    case METRIC_POWER_W:
//                        cur_val = stats->external_data->pow_w[old_id];
//                        new_limit = stats->external_data->init_pow_w[old_id];
//                        break;
//                    case METRIC_IOU:
//                        cur_val = stats->eval_data->iou[old_id];
//                        new_limit = stats->eval_data->init_iou[old_id];
//                        break;
//                }
//
//                if (!is_within_constraint(cur_val, constraint) &&
//                    is_within_constraint(new_limit, constraint)) {
//                    SLOGI(2,
//                          "SELECT | Select | Codec %2d: Metric %d, val %5.3f, not within constraint limit of %5.3f, replacing with %5.3f",
//                          old_id, constraint.metric, cur_val, constraint.limit, new_limit);
//                    constraint.limit = new_limit;
//                }
//            }
//
//            custom_constraints[nconstr] = constraint;
//            nconstr += 1;
//        }

//        for (int id = 0; id < NUM_CONFIGS; ++id) {
//            metrics_product_custom(state, custom_constraints, &sorted_metrics[id]);
//        }
//
//        print_constraints(custom_constraints);
// ^^^ Old code ^^^

        qsort(sorted_metrics, NUM_CONFIGS, sizeof(sorted_metrics[0]), cmp_metrics);
        state->id = select_codec(state, state->constraints, sorted_metrics);

        // just prints
        SLOGI(1,
              "SELECT | Select | Codec %2d -> %2d; avg: ping %6.1f ms, fps %6.2f, latency %6.1f ms, iou %5.3f, pow %5.3f W",
              old_id, state->id, stats->external_data->ping_ms[old_id], stats->cur_latency_data.fps,
              stats->cur_latency_data.total_ms, stats->eval_data->iou[old_id],
              stats->external_data->pow_w[old_id]);

        // set up statistics for the next round
        for (int id = 0; id < NUM_CONFIGS; ++id) {
            stats->eval_data->iou_nsamples[id] = 1;
            stats->external_data->pow_w_nsamples[id] = 1;
            stats->external_data->ping_ms_nsamples[id] = 1;
        }
    }

    end_ns = get_timestamp_ns();
    SLOGI(4, "SELECT | Took %6.3f ms", (float) (end_ns - start_ns) / 1e6f);

    cleanup:
    pthread_mutex_unlock(&state->lock);
}

int get_codec_id(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    const int id = state->id;
    pthread_mutex_unlock(&state->lock);
    return id;
}

int get_codec_sort_id(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    const int id = state->id;
    int sort_id = 0;

    for (int i = 1; i < NUM_CONFIGS; ++i) {
        if (state->init_sorted_ids[i] == id) {
            sort_id = i;
        }
    }

    pthread_mutex_unlock(&state->lock);
    return sort_id;
}

codec_params_t get_codec_params(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    assert(state->id >= 0 && state->id < NUM_CONFIGS);
    codec_params_t params = CONFIGS[state->id];
    pthread_mutex_unlock(&state->lock);
    return params;
}

void signal_eval_start(codec_select_state_t *state, int frame_index, int codec_id) {
    pthread_mutex_lock(&state->lock);
    state->stats.eval_data->id = codec_id;
    state->stats.eval_data->frame_id = frame_index;
    SLOGI(3, "SELECT | Signal Eval | start | Frame %d, codec %d", frame_index, codec_id);
    pthread_mutex_unlock(&state->lock);
}

void signal_eval_finish(codec_select_state_t *state, float iou) {
    pthread_mutex_lock(&state->lock);

    eval_data_t *eval_data = state->stats.eval_data;
    int codec_id = eval_data->id;

    if (state->enable_profiling) {
        log_frame_f(state->fd, eval_data->frame_id, "cs_eval", "iou", iou);
    }

    int *iou_nsamples;
    float *iou_avg;

    if (state->is_calibrating) {
        iou_nsamples = &eval_data->init_iou_nsamples[codec_id];
        iou_avg = &eval_data->init_iou[codec_id];
    } else {
        if (iou == -2.0f) {
            // -2 means no detections both in reference, and compressed frame; not counted in calibration
            iou = 1.0f;
        }

        iou_nsamples = &eval_data->iou_nsamples[codec_id];
        iou_avg = &eval_data->iou[codec_id];
    }

    if (iou >= 0.0f) {
        incr_avg(iou, iou_avg, iou_nsamples);
    }

    SLOGI(3, "SELECT | Signal Eval | Finish | Codec %2d, iou %5.3f, avg iou %5.3f, %d samples",
          codec_id, iou, *iou_avg, *iou_nsamples);

    pthread_mutex_unlock(&state->lock);
}

void push_external_pow(codec_select_state_t *state, int64_t ts_ns, int amp, int volt) {
    SLOGI(3, "SELECT | Push External | ts %ld, %d amp, %d volt", ts_ns, amp, volt);
    pthread_mutex_lock(&state->lock);

    external_data_t *data = state->stats.external_data;
    const int inext = data->inext % NUM_EXTERNAL_SAMPLES;

    if (data->ts_ns[inext] != 0) {
        LOGW("SELECT | Push External | Overflow! Old ts: %ld", data->ts_ns[inext]);
    }

    data->ts_ns[inext] = ts_ns;
    data->ext_amp[inext] = amp;
    data->ext_volt[inext] = volt;
    data->ext_ping_ms[inext] = EMPTY_EXT_PING;
    data->inext += 1;

    pthread_mutex_unlock(&state->lock);
}

void push_external_ping(codec_select_state_t *state, int64_t ts_ns, float ping_ms) {
    SLOGI(3, "SELECT | Push External | Ping %6.1f ms", ping_ms);
    pthread_mutex_lock(&state->lock);

    external_data_t *data = state->stats.external_data;
    const int inext = data->inext % NUM_EXTERNAL_SAMPLES;

    if (data->ts_ns[inext] != 0) {
        LOGW("SELECT | Push External | Overflow! Old ts: %ld", data->ts_ns[inext]);
    }

    data->ts_ns[inext] = ts_ns;
    data->ext_amp[inext] = EMPTY_EXT_POW;
    data->ext_volt[inext] = EMPTY_EXT_POW;
    data->ext_ping_ms[inext] = ping_ms;
    data->inext += 1;

    pthread_mutex_unlock(&state->lock);
}

#ifdef __cplusplus
}
#endif
