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
static const int64_t CALIB_SELECT_INTERVAL_MS = 6500;

// Whether or not to run the first calibration run without collecting statistics
//static const bool DRY_RUN = true;

// Smoothing factors (higher means smoother)
static const float EXT_PING_ALPHA = 7.0f / 8.0f; // TCP RTT estimator = 0.875
static const float IOU_ALPHA = 0.6f;
static const float EXT_POW_ALPHA = 7.0f / 8.0f;

// How many calibration runs to do in the beginning (>= 1)
static const int NUM_CALIB_RUNS = 3;

// How many IoU eval samples per codec are required for finishing the calibration
static const int NUM_CALIB_IOU_SAMPLES = 5;

// How many latency samples need to be collected before doing codec selection decision
static const int MIN_NSAMPLES = 2;

// Empty initializers

static const kernel_times_ms_t EMPTY_KERNEL_TIMES = {.enc = 0.0f, .dec = 0.0f, .dnn = 0.0f, .postprocess = 0.0f, .seg_enc = 0.0f, .seg_dec = 0.0f, .reconstruct = 0.0f,};
static const latency_data_t EMPTY_LATENCY_DATA = {.kernel_times_ms = EMPTY_KERNEL_TIMES, .total_ms = 0.0f, .latency_ms = 0.0f, .network_ms = 0.0f, .size_bytes = 0.0f, .size_bytes_log10 = 0.0f, .fps = 0.0f};
static const int EMPTY_EXT_POW = -1;
static const int EMPTY_EXT_PING = -1.0f;

// the worst value ever that is worse than every value
static const float WORST_VAL = -999999.999999f;

// Limits for hard constraints -- could be configurable at runtime in principle
static const float LIMIT_LATENCY_MS = 300.0f;
static const float LIMIT_IOU = 0.5f;

// Used to artificially scale latency up/down to avoid going to a cellar
// TODO: For this to work, we'd need to precede the offsets by increasing ping sa well
#define NUM_LATENCY_OFFSETS 0
static const int64_t LATENCY_OFFSETS_MS[NUM_LATENCY_OFFSETS] = {}; // = {1500, 0};  // how much to add
static const int64_t LATENCY_OFFSET_TIMES_SEC[NUM_LATENCY_OFFSETS] = {}; // = {10, 30}; // when to add

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
        default:
            assert(0 && "unknown constraint optimization");
            return false;
    }
}

static metric_t to_metric(int i) {
    return (metric_t) (i);
}

static void metrics_product_custom(const codec_select_state_t *const state,
                                   const constraint_t constraints[NUM_CONSTRAINTS],
                                   indexed_metrics_t *metrics) {
    const int codec_id = metrics->codec_id;

    if (!state->is_allowed[codec_id]) {
        metrics->product = 0.0f;
        metrics->all_fit_constraints = false;
        for (int i = 0; i < NUM_METRICS; ++i) {
            metrics->violates_any_constr[i] = true;
        }
        return;
    }

    float product = 1.0f;
    bool all_fit_constraints = true;
    bool violates_any_constr[NUM_METRICS];
    for (int i = 0; i < NUM_METRICS; ++i) {
        violates_any_constr[i] = false;
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
                violates_any_constr[i] = true;
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

                // avoid multiplying the product twice if more soft constraints are defined for one metric
                // TODO: Should be an error
                break;
            }
        }
    }

    for (int i = 0; i < NUM_METRICS; ++i) {
        metrics->violates_any_constr[i] = violates_any_constr[i];
    }
    metrics->all_fit_constraints = all_fit_constraints;
    metrics->product = product;
}

static void metrics_product(const codec_select_state_t *const state, indexed_metrics_t *metrics) {
    return metrics_product_custom(state, state->constraints, metrics);
}

static float get_init_pow_variance(const external_data_t *data) {
    sum_data_t sum_data = data->init_pow_w_sum[data->min_pow_w_id];
    int nsamples = data->init_pow_w_nsamples[data->min_pow_w_id];

    float var = 0.0f;

    if (nsamples > 1) {
        var = (sum_data.sumsq - sum_data.sum * sum_data.sum / (float) (nsamples)) /
              ((float) (nsamples) - 1.0f);
    }

    return var;
}

static float get_init_rel_pow_thr(const external_data_t *data) {
    float min_pow_w = data->init_pow_w[data->min_pow_w_id];
    float variance = get_init_pow_variance(data);

    return min_pow_w + sqrtf(variance);
}

static float get_rel_pow(const external_data_t *data, float power_w) {
    const float thr = get_init_rel_pow_thr(data);

    float rel_pow = 1.0f;
    if (power_w > thr) {
        rel_pow = power_w / thr;
    }

    return rel_pow;
}

static void
populate_vals(float latency_ms, float size_bytes, float power_w, float iou, float rel_pow,
                          float vals[NUM_METRICS]) {
    vals[METRIC_LATENCY_MS] = latency_ms;
    vals[METRIC_SIZE_BYTES] = size_bytes;
    if (size_bytes == 0.0f) {
        vals[METRIC_SIZE_BYTES_LOG10] = 0.0f;
    } else {
        vals[METRIC_SIZE_BYTES_LOG10] = log10f(size_bytes);
    }
    vals[METRIC_POWER_W] = power_w;
    vals[METRIC_IOU] = iou;
    vals[METRIC_REL_POW] = rel_pow;
}

static indexed_metrics_t
populate_metrics(const codec_select_state_t *const state, int codec_id, float latency_ms,
                 float network_ms, float size_bytes, float power_w, float iou) {
    indexed_metrics_t metrics;

    float rel_pow = get_rel_pow(state->stats.external_data, power_w);

    populate_vals(latency_ms, size_bytes, power_w, iou, rel_pow, metrics.vals);
    populate_vals(network_ms, 0.0f, 0.0f, 0.0f, 0.0f, metrics.vol_vals);
    metrics.codec_id = codec_id;

    metrics_product(state, &metrics);

    return metrics;
}

static indexed_metrics_t
populate_init_metrics(const codec_select_state_t *const state, int codec_id) {
    const float latency_ms = state->stats.init_latency_data[codec_id].latency_ms;
    const float network_ms = state->stats.init_latency_data[codec_id].network_ms;
    const float size_bytes = state->stats.init_latency_data[codec_id].size_bytes;
    const float power_w = state->stats.external_data->init_pow_w[codec_id];
    const float iou = state->stats.eval_data->init_iou[codec_id];

    return populate_metrics(state, codec_id, latency_ms, network_ms, size_bytes, power_w, iou);
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
    if (SELECT_VERBOSITY & SLOG_CONSTR) {
        for (int i = 0; i < NUM_CONSTRAINTS; ++i) {
            constraint_t constr = constraints[i];
            if (constr.type != CONSTR_SOFT) {
                SLOGI(SLOG_CONSTR, "SELECT | Constraints | %16s: %8.3f (scale %11.6f)",
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
        default:
            assert(0 && "unknown constraint optimization");
            return false;
    }
}

static void set_min_pow_id(external_data_t *data) {
    int min_pow_w_id = 0;

    if (NUM_CONFIGS > 1) {
        float min_pow_w = data->init_pow_w[0];

        for (int id = 1; id < NUM_CONFIGS; ++id) {
            float pow = data->init_pow_w[id];
            if (pow < min_pow_w) {
                min_pow_w = pow;
                min_pow_w_id = id;
            }
        }
    }

    data->min_pow_w_id = min_pow_w_id;
}

bool is_calibrating(const codec_select_state_t *const state) {
    return state->stage != STAGE_RUNNING;
}

static int select_codec(const codec_select_state_t *const state,
                        const constraint_t constraints[NUM_CONSTRAINTS],
                        const indexed_metrics_t sorted_metrics[NUM_CONFIGS]) {
    int new_codec_id = LOCAL_CODEC_ID;  // default to local
    bool some_fit_constraints = false;
    int old_id = state->id;

    // print ping
    SLOGI(SLOG_SELECT_2,
          "SELECT | Select | Codec %2d, Ping init (fill): %8.2f ms, avg: %8.2f ms, remote avg %8.2f ms",
          old_id, state->stats.cur_init_avg_fill_ping_ms, state->stats.cur_avg_fill_ping_ms,
          state->stats.cur_remote_avg_fill_ping_ms);
    SLOGI(SLOG_SELECT_2,
          "SELECT | Select | Codec %2d, Ping init       : %8.2f ms, avg: %8.2f ms, remote avg %8.2f ms",
          old_id, state->stats.cur_init_avg_ping_ms, state->stats.cur_avg_ping_ms,
          state->stats.cur_remote_avg_ping_ms);

    // print kernel time breakdowns
    if (is_calibrating(state)) {
        for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
            const indexed_metrics_t metrics = sorted_metrics[sort_id];
            const int id = metrics.codec_id;

            SLOGI(SLOG_SELECT_2,
                  "SELECT | Select | Codec %2d, size: %7.0f B, latency: total %8.3f, network %8.3f, enc %8.3f, dec %8.3f, dnn %8.3f, postprocess %8.3f, seg_enc %8.3f, seg_dec %8.3f, reconstruct %8.3f",
                  id, state->stats.init_latency_data[id].size_bytes,
                  state->stats.init_latency_data[id].latency_ms,
                  state->stats.init_latency_data[id].network_ms,
                  state->stats.init_latency_data[id].kernel_times_ms.enc,
                  state->stats.init_latency_data[id].kernel_times_ms.dec,
                  state->stats.init_latency_data[id].kernel_times_ms.dnn,
                  state->stats.init_latency_data[id].kernel_times_ms.postprocess,
                  state->stats.init_latency_data[id].kernel_times_ms.seg_enc,
                  state->stats.init_latency_data[id].kernel_times_ms.seg_dec,
                  state->stats.init_latency_data[id].kernel_times_ms.reconstruct);
        }
    } else {
        SLOGI(SLOG_SELECT_2,
              "SELECT | Select | Codec %2d, size: %7.0f B, latency: total %8.3f, network %8.3f, enc %8.3f, dec %8.3f, dnn %8.3f, postprocess %8.3f, seg_enc %8.3f, seg_dec %8.3f, reconstruct %8.3f",
              old_id, state->stats.cur_latency_data.size_bytes,
              state->stats.cur_latency_data.latency_ms, state->stats.cur_latency_data.network_ms,
              state->stats.cur_latency_data.kernel_times_ms.enc,
              state->stats.cur_latency_data.kernel_times_ms.dec,
              state->stats.cur_latency_data.kernel_times_ms.dnn,
              state->stats.cur_latency_data.kernel_times_ms.postprocess,
              state->stats.cur_latency_data.kernel_times_ms.seg_enc,
              state->stats.cur_latency_data.kernel_times_ms.seg_dec,
              state->stats.cur_latency_data.kernel_times_ms.reconstruct);
    }

    // If running locally and the ping is still higher than before switching to local, keep local
    if ((old_id == LOCAL_CODEC_ID) &&
        (state->stats.cur_avg_fill_ping_ms > state->stats.cur_remote_avg_fill_ping_ms) &&
        (state->stats.cur_avg_fill_ping_ms > state->stats.cur_init_avg_fill_ping_ms)) {
        SLOGI(SLOG_SELECT_2, "SELECT | Select | Ping too high, staying with local");
        return LOCAL_CODEC_ID;
    }

    for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
        // iterate by ascending the init product
        const indexed_metrics_t metrics = sorted_metrics[sort_id];
        const int id = metrics.codec_id;
        const bool fits_constraints = metrics.all_fit_constraints;

        some_fit_constraints |= fits_constraints;

        if (is_calibrating(state)) {
            SLOGI(SLOG_SELECT_2,
                  "SELECT | Select | Codec %2d, nsamples: %3d, size %7.0f B, avg fps %5.1f, latency %6.1f ms, iou %5.3f, pow %5.3f W, allowed: %d, fits: %d, prod %8.5f",
                  id, state->stats.init_nsamples[id], state->stats.init_latency_data[id].size_bytes,
                  state->stats.init_latency_data[id].fps, metrics.vals[METRIC_LATENCY_MS],
                  metrics.vals[METRIC_IOU], metrics.vals[METRIC_POWER_W],
                  state->is_allowed[metrics.codec_id], fits_constraints, metrics.product);
        } else {
            SLOGI(SLOG_SELECT_2,
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
        for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
            is_constr_violated[constr_id] = false;
        }

        for (int metric_id = 0; metric_id < NUM_METRICS; ++metric_id) {
            bool any_codec_violates_this_metrics_constraint = false;

            for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
                const indexed_metrics_t metrics = sorted_metrics[sort_id];

                if (metrics.violates_any_constr[metric_id]) {
                    any_codec_violates_this_metrics_constraint = true;
                    break;
                }
            }

            if (any_codec_violates_this_metrics_constraint) {
                for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
                    if (constraints[constr_id].metric == metric_id) {
                        is_constr_violated[constr_id] = true;
                        break;
                    }
                }
            }
        }

        // Select codec with the best value of the violated constraint metric
        int best_codec_id = 0;
        float best_vals[NUM_METRICS];
        for (int i = 0; i < NUM_METRICS; ++i) {
            best_vals[i] = WORST_VAL;
        }

        for (int sort_id = 0; sort_id < NUM_CONFIGS; ++sort_id) {
            const indexed_metrics_t metrics = sorted_metrics[sort_id];
            int cmp_res = 0;

            for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
                if (!is_constr_violated[constr_id]) {
                    continue;
                }

                if (cmp_res != 0) {
                    break;
                }

                constraint_t violated_constraint = constraints[constr_id];
                float cur_val = metrics.vals[violated_constraint.metric];
                cmp_res = cmp_vals(cur_val, best_vals[violated_constraint.metric],
                                   violated_constraint.optimization);

                SLOGI(SLOG_SELECT_DBG,
                      "SELECT | Select | All Fail Cmp | Codec %2d, metric %16s, curval %11.3f, bestval %13.3f, cmp %2d",
                      metrics.codec_id, METRIC_NAMES[violated_constraint.metric], cur_val,
                      best_vals[violated_constraint.metric], cmp_res);
            }

            if (cmp_res == 0) {
                SLOGI(SLOG_SELECT_DBG,
                      "SELECT | Select | All Fail Cmp | Codec %2d is the new best (by product)",
                      metrics.codec_id);
                if (metrics.product > sorted_metrics[best_codec_id].product) {
                    best_codec_id = metrics.codec_id;
                    for (int i = 0; i < NUM_METRICS; ++i) {
                        best_vals[i] = metrics.vals[i];
                    }
                }
            } else if (cmp_res > 0) {
                SLOGI(SLOG_SELECT_DBG,
                      "SELECT | Select | All Fail Cmp | Codec %2d is the new best (by value)",
                      metrics.codec_id);
                best_codec_id = metrics.codec_id;
                for (int i = 0; i < NUM_METRICS; ++i) {
                    best_vals[i] = metrics.vals[i];
                }
            }
        }

        SLOGI(SLOG_SELECT, "SELECT | Select | No codec fits constraints, selecting %d.",
              best_codec_id);
        new_codec_id = best_codec_id;
    }

    if (state->enable_profiling) {
        float thr = get_init_rel_pow_thr(state->stats.external_data);
        float pow_var = get_init_pow_variance(state->stats.external_data);
        log_frame_f(state->fd, state->last_frame_id, "cs_select", "rel_pow_thr", thr);
        log_frame_f(state->fd, state->last_frame_id, "cs_select", "pow_var", thr);
        log_frame_int(state->fd, state->last_frame_id, "cs_select", "new_codec_id", new_codec_id);
    }

    return new_codec_id;
}

static void
collect_latency(const codec_select_state_t *state, const frame_metadata_t *frame_metadata,
                int64_t received_frame_ts_ns, bool should_skip, codec_stats_t *stats) {

    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | start %7.2f ms\n",
          frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_enc - frame_metadata->host_ts_ns.start) / 1e6f);
    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | enc   %7.2f ms\n",
          frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_dnn - frame_metadata->host_ts_ns.before_enc) / 1e6f);
    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | dnn   %7.2f ms\n",
          frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.before_wait - frame_metadata->host_ts_ns.before_dnn) / 1e6f);
    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | wait  %7.2f ms\n",
          frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.after_wait - frame_metadata->host_ts_ns.before_wait) / 1e6f);
    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | end   %7.2f ms\n",
          frame_metadata->frame_index,
          (frame_metadata->host_ts_ns.stop - frame_metadata->host_ts_ns.after_wait) / 1e6f);

    kernel_times_ms_t kernel_times_ms = EMPTY_KERNEL_TIMES;
    find_event_time("enc_event", state->collected_events, &kernel_times_ms.enc);
    find_event_time("dec_event", state->collected_events, &kernel_times_ms.dec);
    find_event_time("dnn_event", state->collected_events, &kernel_times_ms.dnn);
    find_event_time("postprocess_event", state->collected_events, &kernel_times_ms.postprocess);
    find_event_time("seg_enc_event", state->collected_events, &kernel_times_ms.seg_enc);
    find_event_time("seg_dec_event", state->collected_events, &kernel_times_ms.seg_dec);
    find_event_time("reconstruct_event", state->collected_events, &kernel_times_ms.reconstruct);

    // const
    float latency_ms =
            (float) (frame_metadata->host_ts_ns.stop - frame_metadata->host_ts_ns.start) / 1e6f;

    const int frame_index = frame_metadata->frame_index;

    // fill ping
    float fill_ping_ms = (float) (frame_metadata->host_ts_ns.fill_ping_duration_ms) / 1e6f;

    if (!should_skip && (state->stage != STAGE_CALIB_DRY) &&
        (state->stage != STAGE_CALIB_IOU_ONLY) &&
        (frame_metadata->host_ts_ns.fill_ping_duration_ms != -1)) {
        float *avg_fill_ping_ms;

        if (is_calibrating(state)) {
            avg_fill_ping_ms = &stats->cur_init_avg_fill_ping_ms;
        } else {
            avg_fill_ping_ms = &stats->cur_avg_fill_ping_ms;
        }

        if (*avg_fill_ping_ms == 0.0f) {
            *avg_fill_ping_ms = fill_ping_ms;
        } else if (fill_ping_ms != 0.0f) {
            ewma(EXT_PING_ALPHA, fill_ping_ms, avg_fill_ping_ms);
        }
    }

    SLOGI(SLOG_COLLECT_LATENCY_DBG, "SELECT | Host Times | Frame %d | fill  %7.2f ms\n",
          frame_metadata->frame_index, fill_ping_ms);
    if (frame_metadata->host_ts_ns.fill_ping_duration_ms == -1) {
        fill_ping_ms = latency_ms;
    }

    float network_ms = 0.0f;
    if (frame_metadata->codec.id != LOCAL_CODEC_ID) {
        network_ms = fmax(0.0f, latency_ms - kernel_times_ms.enc - kernel_times_ms.dec
                                - kernel_times_ms.dnn - kernel_times_ms.postprocess
                                - kernel_times_ms.seg_enc - kernel_times_ms.seg_dec
                                - kernel_times_ms.reconstruct);
    }

    const float size_bytes = (float) (frame_metadata->size_bytes_tx +
                                      frame_metadata->size_bytes_rx);

    const float frame_time_ms =
            (float) (received_frame_ts_ns - stats->last_received_image_ts_ns) / 1e6f;

    latency_data_t *data;
    int *nsamples;

    if (is_calibrating(state)) {
        data = &stats->init_latency_data[frame_metadata->codec.id];
        nsamples = &stats->init_nsamples[frame_metadata->codec.id];
    } else {
        data = &stats->cur_latency_data;
        nsamples = &stats->cur_nsamples;
    }

    if (!should_skip && (state->stage != STAGE_CALIB_DRY) &&
        (state->stage != STAGE_CALIB_IOU_ONLY)) {
        incr_avg_noup(kernel_times_ms.enc, &data->kernel_times_ms.enc, *nsamples);
        incr_avg_noup(kernel_times_ms.dec, &data->kernel_times_ms.dec, *nsamples);
        incr_avg_noup(kernel_times_ms.dnn, &data->kernel_times_ms.dnn, *nsamples);
        incr_avg_noup(kernel_times_ms.postprocess, &data->kernel_times_ms.postprocess, *nsamples);
        incr_avg_noup(kernel_times_ms.seg_enc, &data->kernel_times_ms.seg_enc, *nsamples);
        incr_avg_noup(kernel_times_ms.seg_dec, &data->kernel_times_ms.seg_dec, *nsamples);
        incr_avg_noup(kernel_times_ms.reconstruct, &data->kernel_times_ms.reconstruct, *nsamples);
        incr_avg_noup(latency_ms, &data->latency_ms, *nsamples);
        incr_avg_noup(network_ms, &data->network_ms, *nsamples);
        incr_avg_noup(size_bytes, &data->size_bytes, *nsamples);
        *nsamples += 1;

        data->total_ms += frame_time_ms;
        data->fps = (float) (*nsamples) / data->total_ms * 1e3f;
    }


    if (state->enable_profiling && (state->stage != STAGE_CALIB_DRY) &&
        (state->stage != STAGE_CALIB_IOU_ONLY)) {
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_enc_ms",
                    kernel_times_ms.enc);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_dec_ms",
                    kernel_times_ms.dec);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_dnn_ms",
                    kernel_times_ms.dnn);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_postprocess_ms",
                    kernel_times_ms.postprocess);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_enc_seg_ms",
                    kernel_times_ms.seg_enc);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_dec_seg_ms",
                    kernel_times_ms.seg_dec);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "kernel_reconstruct_ms",
                    kernel_times_ms.reconstruct);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "frame_time_ms", frame_time_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "latency_ms", latency_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "network_ms", network_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "size_bytes", size_bytes);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "fill_ping_ms", fill_ping_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "cur_init_avg_fill_ping_ms",
                    stats->cur_init_avg_fill_ping_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "cur_avg_fill_ping_ms",
                    stats->cur_avg_fill_ping_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "cur_remote_avg_fill_ping_ms",
                    stats->cur_remote_avg_fill_ping_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_enc_ms",
                    data->kernel_times_ms.enc);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_dec_ms",
                    data->kernel_times_ms.dec);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_dnn_ms",
                    data->kernel_times_ms.dnn);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_postprocess_ms",
                    data->kernel_times_ms.postprocess);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_enc_seg_ms",
                    data->kernel_times_ms.seg_enc);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_dec_seg_ms",
                    data->kernel_times_ms.seg_dec);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_kernel_reconstruct_ms",
                    data->kernel_times_ms.reconstruct);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "total_ms", data->total_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "fps", data->fps);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_latency_ms",
                    data->latency_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_network_ms",
                    data->network_ms);
        log_frame_f(state->fd, frame_index, "cs_update_latency", "avg_size_bytes",
                    data->size_bytes);
        log_frame_int(state->fd, frame_index, "cs_update_latency", "nsamples", *nsamples);
    }
}

static void collect_external_data(const codec_select_state_t *state, int frame_codec_id,
                                  int64_t frame_stop_ts_ns, codec_stats_t *stats) {
    external_data_t *data = stats->external_data;

    SLOGI(SLOG_COLLECT_EXTERNAL_DBG,
          "SELECT | Collect External | Frame codec %2d, stop ts: %ld, prev stop ts: %ld",
          frame_codec_id, frame_stop_ts_ns, data->prev_frame_stop_ts_ns);
    while (data->istart < data->inext) {
        const int istart = data->istart % NUM_EXTERNAL_SAMPLES;
        const int64_t ts_ns = data->ts_ns[istart];
        SLOGI(SLOG_COLLECT_EXTERNAL_DBG, "SELECT | Collect External | -- istart %4d, ts %ld",
              istart, ts_ns);

        if (ts_ns != 0) {
            const int amp = data->ext_amp[istart];
            const int volt = data->ext_volt[istart];
            const float pow_w = (float) (-amp) * (float) (volt) / 1e6f;
            const float ping_ms = data->ext_ping_ms[istart];

            SLOGI(SLOG_COLLECT_EXTERNAL_DBG,
                  "SELECT | Collect External | -- within; amp: %4d, volt: %4d, pow_w: %5.3f, ping_ms: %7.1f",
                  amp, volt, pow_w, ping_ms);

            float value;
            float *avg_codec_value;
            float *avg_value;
            int *codec_nsamples;
            float alpha;

            if (ping_ms != EMPTY_EXT_PING) {
                value = ping_ms;
                alpha = EXT_PING_ALPHA;
                if (is_calibrating(state)) {
                    avg_codec_value = &data->init_ping_ms[frame_codec_id];
                    codec_nsamples = &data->init_ping_ms_nsamples[frame_codec_id];
                    avg_value = &stats->cur_init_avg_ping_ms;
                } else {
                    avg_codec_value = &data->ping_ms[frame_codec_id];
                    codec_nsamples = &data->ping_ms_nsamples[frame_codec_id];
                    avg_value = &stats->cur_avg_ping_ms;
                }
            } else if (amp != EMPTY_EXT_POW || volt != EMPTY_EXT_POW) {
                value = pow_w;
                alpha = EXT_POW_ALPHA;
                if (is_calibrating(state)) {
                    avg_codec_value = &data->init_pow_w[frame_codec_id];
                    codec_nsamples = &data->init_pow_w_nsamples[frame_codec_id];
                    avg_value = &stats->cur_init_avg_pow_w;
                } else {
                    avg_codec_value = &data->pow_w[frame_codec_id];
                    codec_nsamples = &data->pow_w_nsamples[frame_codec_id];
                    avg_value = &stats->cur_avg_pow_w;
                }
            } else {
                LOGE("SELECT | Collect External | ERROR: Got completely empty value for ts %ld",
                     ts_ns);
                data->ts_ns[istart] = 0;
                data->istart += 1;
                continue;
            }

            if ((state->stage != STAGE_CALIB_DRY) && (state->stage != STAGE_CALIB_IOU_ONLY)) {
                if (ts_ns >= data->prev_frame_stop_ts_ns && ts_ns <= frame_stop_ts_ns) {
                    incr_avg(value, avg_codec_value, codec_nsamples);
                }

                if (amp != EMPTY_EXT_POW && volt != EMPTY_EXT_POW) {
                    sum_data_t *sum_data;

                    if (is_calibrating(state)) {
                        sum_data = &data->init_pow_w_sum[frame_codec_id];
                    } else {
                        sum_data = &data->pow_w_sum[frame_codec_id];
                    }

                    sum_data->sum += value;
                    sum_data->sumsq += value * value;
                }

                // track also global average ping/power
                if (*avg_value == 0.0f) {
                    *avg_value = value;
                } else if (value > 0.0f) {
                    ewma(alpha, value, avg_value);
                }
            }

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
                if ((state->stage != STAGE_CALIB_DRY) && (state->stage != STAGE_CALIB_IOU_ONLY)) {
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                param_avg_val, *avg_codec_value);
                    log_frame_int(state->fd, state->last_frame_id, "cs_update_external",
                                  param_nsamples, *codec_nsamples);
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                "cur_init_avg_ping_ms", stats->cur_init_avg_ping_ms);
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                "cur_avg_ping_ms", stats->cur_avg_ping_ms);
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                "cur_remote_avg_ping_ms", stats->cur_remote_avg_ping_ms);
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                "cur_init_avg_pow_w", stats->cur_init_avg_pow_w);
                    log_frame_f(state->fd, state->last_frame_id, "cs_update_external",
                                "cur_avg_pow_w", stats->cur_avg_pow_w);
                }
            }
        }

        data->ts_ns[istart] = 0;
        data->istart += 1;
    }
}

void
init_codec_select(int config_flags, int fd, int do_algorithm, bool lock_codec, bool sync_with_input,
                  codec_select_state_t **state) {
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
    if (do_algorithm == 0) {
        new_state->stage = STAGE_RUNNING;
        new_state->stage_id = NUM_STAGES - 1;
        new_state->lock_codec = true;
    } else {
        if (new_state->local_only) {
            // there is nothing to calibrate in local-only
            new_state->stage = STAGE_RUNNING;
            new_state->stage_id = NUM_STAGES - 1;
        } else {
            new_state->stage = STAGES[0];
            new_state->stage_id = 0;
        }
        new_state->lock_codec = lock_codec;
    }
    assert(new_state->stage_id < NUM_STAGES);
    new_state->sync_with_input = sync_with_input;
    new_state->enable_profiling = ENABLE_PROFILING & config_flags;
    new_state->fd = fd;

    int nconstr = 0;
    new_state->constraints[nconstr++] = {.metric = METRIC_LATENCY_MS, .optimization = OPT_MIN, .type = CONSTR_HARD, .limit = LIMIT_LATENCY_MS, .scale = LATENCY_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_IOU, .optimization = OPT_MAX, .type = CONSTR_HARD, .limit = LIMIT_IOU, .scale = IOU_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_LATENCY_MS, .optimization = OPT_MIN, .type = CONSTR_VOLATILE, .limit = LIMIT_LATENCY_MS, .scale = LATENCY_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_LATENCY_MS, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = LATENCY_SCALE};
//    new_state->constraints[nconstr++] = {.metric = METRIC_SIZE_BYTES, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = SIZE_SCALE};
//    new_state->constraints[nconstr++] = {.metric = METRIC_SIZE_BYTES_LOG10, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = SIZE_SCALE_LOG10};
//    new_state->constraints[nconstr++] = {.metric = METRIC_POWER_W, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = POWER_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_IOU, .optimization = OPT_MAX, .type = CONSTR_VOLATILE, .scale = IOU_SCALE};
    new_state->constraints[nconstr++] = {.metric = METRIC_IOU, .optimization = OPT_MAX, .type = CONSTR_SOFT, .scale = IOU_SCALE};
//    new_state->constraints[nconstr++] = {.metric = METRIC_REL_POW, .optimization = OPT_MIN, .type = CONSTR_SOFT, .scale = REL_POWER_SCALE};
    assert(nconstr == NUM_CONSTRAINTS);

    new_state->stats.eval_data = eval_data;
    new_state->stats.external_data = external_data;
    new_state->stats.prev_frame_codec_id = -1;
    // local-only always max. quality
    new_state->stats.eval_data->init_iou[LOCAL_CODEC_ID] = 1.0f;
    new_state->stats.eval_data->iou[LOCAL_CODEC_ID] = 1.0f;
    for (int id = 0; id < NUM_CONFIGS; ++id) {
        new_state->stats.init_nruns[id] = -1;
    }
    // ... the rest of new_state should be zeros

    log_constraints(new_state, new_state->constraints, -1);
    log_frame_int(new_state->fd, -1, "cs_init", "lock_codec", new_state->lock_codec);

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
    const int64_t update_start_ns = get_timestamp_ns();
    const int64_t frame_stop_ts_ns = frame_metadata->host_ts_ns.stop;
    const int frame_index = frame_metadata->frame_index;

    // Codec ID used for encoding the frame; might be different from state->id
    const int frame_codec_id = frame_metadata->codec.id;

    pthread_mutex_lock(&state->lock);

    state->last_frame_id = frame_index;

    codec_stats_t *stats = &state->stats;
    const bool should_skip = stats->prev_frame_codec_id != frame_codec_id;
    const bool is_eval_frame = frame_metadata->run_args.is_eval_frame;

    if (state->enable_profiling) {
        log_frame_int(state->fd, frame_index, "cs_update", "frame_codec_id", frame_codec_id);
        log_frame_i64(state->fd, frame_index, "cs_update", "start_ns", update_start_ns);
        log_frame_int(state->fd, frame_index, "cs_update", "is_calibrating",
                      state->stage == STAGE_RUNNING ? 0 : 1);
        log_frame_int(state->fd, frame_index, "cs_update", "is_dry_run",
                      state->stage == STAGE_CALIB_DRY ? 1 : 0);
        log_frame_str(state->fd, frame_index, "cs_update", "stage", STAGE_NAMES[state->stage]);
        log_frame_int(state->fd, frame_index, "cs_update", "skip", should_skip);
        log_frame_int(state->fd, frame_index, "cs_update", "codec_selected",
                      frame_metadata->run_args.codec_selected);
        log_frame_i64(state->fd, frame_index, "cs_update", "latency_offset_ms",
                      frame_metadata->run_args.latency_offset_ms);
    }

    if (frame_metadata->run_args.codec_selected) {
        // reset statistics because this frame is the first after a new codec selection
        for (int id = 0; id < NUM_CONFIGS; ++id) {
            stats->eval_data->iou_nsamples[id] = 1;
            stats->external_data->pow_w_nsamples[id] = 1;
            stats->external_data->pow_w_sum[id].sum = stats->external_data->pow_w[id];
            stats->external_data->pow_w_sum[id].sumsq =
                    stats->external_data->pow_w[id] * stats->external_data->pow_w[id];
            stats->external_data->ping_ms_nsamples[id] = 1;
        }

        // initialize latency data with the init values to prevent latency etc. being zero for codec selection
        stats->cur_latency_data = stats->init_latency_data[frame_codec_id];
        stats->cur_nsamples = 0;
    }

    // Collect latency (do not update averages if codec just changed)
    collect_latency(state, frame_metadata, update_start_ns, should_skip, stats);

    // Collect power
    collect_external_data(state, frame_codec_id, frame_stop_ts_ns, stats);

    if ((stats->prev_frame_codec_id != frame_codec_id) && is_calibrating(state)) {
        stats->init_nruns[stats->prev_frame_codec_id] += 1;
    }

//        SLOGI(SLOG_UPDATE_DBG,
//              "SELECT | Update | Frame %d, Codec changed from %2d, not updating stats.",
//              frame_index, stats->prev_frame_codec_id);
//        goto cleanup;
//    }

    if (is_calibrating(state)) {
        int total_nsamples = 0;
        for (int i = 0; i < NUM_CONFIGS; ++i) {
            total_nsamples += stats->init_nsamples[i];
        }

        SLOGI(SLOG_UPDATE_DBG,
              "SELECT | Update | Frame %3d, codec %2d, nsamples %4d (total %4d), init avg: latency %6.1f ms, ping %6.1f ms, pow %5.3f (%d sampl.)",
              frame_index, frame_codec_id, stats->init_nsamples[frame_codec_id], total_nsamples,
              stats->init_latency_data[frame_codec_id].latency_ms,
              stats->external_data->init_ping_ms[frame_codec_id],
              stats->external_data->init_pow_w[frame_codec_id],
              stats->external_data->init_pow_w_nsamples[frame_codec_id]);
    } else {
        SLOGI(SLOG_UPDATE_DBG,
              "SELECT | Update | Frame %3d, codec %2d, ping: %6.1f ms, latency: %6.1f ms, pow %5.3f W",
              frame_index, frame_codec_id, stats->external_data->ping_ms[frame_codec_id],
              stats->cur_latency_data.latency_ms,
              stats->external_data->pow_w[frame_codec_id]);
    }

    state->stats.last_received_image_ts_ns = update_start_ns;
    stats->external_data->prev_frame_stop_ts_ns = frame_stop_ts_ns;
    stats->prev_frame_codec_id = frame_codec_id;

    if (state->enable_profiling) {
        const float duration_ms = (float) (get_timestamp_ns() - update_start_ns) / 1e6f;
        log_frame_f(state->fd, frame_index, "cs_update", "duration_ms", duration_ms);
    }

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
    const int64_t start_ns = get_timestamp_ns();
    float duration_ms;
    bool codec_selected = false;

    pthread_mutex_lock(&state->lock);

    const int old_id = state->id;
    codec_stats_t *stats = &state->stats;

    int64_t select_interval_ms = SELECT_INTERVAL_MS;
    if (is_calibrating(state) && (state->stage != STAGE_CALIB_DRY)) {
        select_interval_ms = CALIB_SELECT_INTERVAL_MS;
    }

    if (state->last_timestamp_ns == 0) {
        // skipping first frame
        state->last_timestamp_ns = start_ns;
        goto cleanup;
    }

    state->since_last_select_ms += (start_ns - state->last_timestamp_ns) / 1000000;
    state->last_timestamp_ns = start_ns;

    if (is_calibrating(state) && state->sync_with_input) {
        if (state->got_last_frame) {
            SLOGI(SLOG_SELECT, "SELECT | Calibrating | Got last playback frame");
            state->got_last_frame = false;
        } else {
            goto cleanup;
        }
    } else {
        if (state->since_last_select_ms < select_interval_ms) {
            goto cleanup;
        }
    }


    state->since_last_select_ms = 0.0f;

    if (is_calibrating(state)) {
        // works on init data which are updated only during calibration
        set_min_pow_id(stats->external_data);

        bool stay_in_stage = false;
        for (int id = 0; id < NUM_CONFIGS; ++id) {
            if (stats->init_nruns[id] < state->stage_id) {
                stay_in_stage = true;
                break;
            }
        }

        if (!stay_in_stage) {
            state->stage_id += 1;
            assert(state->stage_id < NUM_STAGES);
            state->stage = STAGES[state->stage_id];
        }

        bool should_still_calibrate = state->stage != STAGE_RUNNING;

        if (should_still_calibrate) {
            // TODO: Shuffle configs to ensure every run runs each codec once
//            state->id = rand() % NUM_CONFIGS;

            if ((state->stage == STAGE_CALIB_DRY) ||
                (state->stage == STAGE_CALIB_IOU_ONLY) ||
                (stats->init_nsamples[old_id] > stats->init_nsamples_prev[old_id])) {
                state->id = (old_id + 1) % NUM_CONFIGS;
            }

            stats->init_nsamples_prev[old_id] = stats->init_nsamples[old_id];

            // just prints
            indexed_metrics_t metrics = populate_init_metrics(state, old_id);

            SLOGI(SLOG_SELECT,
                  "SELECT | Calibrating | (%2d -> %2d), nsamples: %3d, size: %7.0f B, avg latency %6.1f ms, fps %6.2f, iou %5.3f, pow %5.3f W, allowed: %d, fits: %d, prod: %8.5f",
                  old_id, state->id, stats->init_nsamples[old_id],
                  stats->init_latency_data[old_id].size_bytes,
                  stats->init_latency_data[old_id].latency_ms, stats->init_latency_data[old_id].fps,
                  state->stats.eval_data->init_iou[old_id],
                  state->stats.external_data->init_pow_w[old_id], state->is_allowed[old_id],
                  metrics.all_fit_constraints, metrics.product);
        } else {
            SLOGI(SLOG_SELECT, "SELECT | Calibrating | End");

            indexed_metrics_t sorted_metrics[NUM_CONFIGS];
            sort_init_metrics(state, sorted_metrics);

            for (int id = 0; id < NUM_CONFIGS; ++id) {
                assert(sorted_metrics[id].codec_id >= 0);
                state->init_sorted_ids[id] = sorted_metrics[id].codec_id;
            }

            print_constraints(state->constraints);
            state->id = select_codec(state, state->constraints, sorted_metrics);

            // populate stats with init values
            stats->cur_avg_fill_ping_ms = stats->cur_init_avg_fill_ping_ms;
            stats->cur_remote_avg_fill_ping_ms = stats->cur_init_avg_fill_ping_ms;
            stats->cur_avg_ping_ms = stats->cur_init_avg_ping_ms;
            stats->cur_remote_avg_ping_ms = stats->cur_init_avg_ping_ms;
            stats->cur_avg_pow_w = stats->cur_init_avg_pow_w;

            for (int id = 0; id < NUM_CONFIGS; ++id) {
                stats->eval_data->iou_nsamples[id] = stats->eval_data->init_iou_nsamples[id];
                stats->eval_data->iou[id] = stats->eval_data->init_iou[id];
                stats->external_data->pow_w_nsamples[id] = stats->external_data->init_pow_w_nsamples[id];
                stats->external_data->pow_w[id] = stats->external_data->init_pow_w[id];
                stats->external_data->pow_w_sum[id] = stats->external_data->init_pow_w_sum[id];
                stats->external_data->ping_ms_nsamples[id] = stats->external_data->init_ping_ms_nsamples[id];
                stats->external_data->ping_ms[id] = stats->external_data->init_ping_ms[id];
            }

            state->calib_end_ns = get_timestamp_ns();
            SLOGI(SLOG_SELECT, "SELECT | Calibrating | End | (%2d -> %2d)", old_id, state->id);
        }
    } else {
        // Whether to apply artificial latency offset
        if (NUM_LATENCY_OFFSETS > 0) {
            for (int i = NUM_LATENCY_OFFSETS - 1; i >= 0; --i) {
                int64_t offset_time_ns = LATENCY_OFFSET_TIMES_SEC[i] * 1000000000;

                if ((start_ns - state->calib_end_ns) > offset_time_ns) {
                    state->latency_offset_ms = LATENCY_OFFSETS_MS[i];
                    SLOGI(SLOG_SELECT_DBG, "SELECT | Select | Set offset %ld ms",
                          state->latency_offset_ms);
                    break;
                }
            }
        }

        if (stats->cur_nsamples < MIN_NSAMPLES) {
            SLOGI(SLOG_SELECT, "SELECT | Select | Got only %d/%d samples. Not enough, skipping",
                  stats->cur_nsamples, MIN_NSAMPLES);
            goto cleanup;
        }

        indexed_metrics_t sorted_metrics[NUM_CONFIGS];
        float cur_vals[NUM_METRICS];
        float cur_vol_vals[NUM_METRICS];
        float init_vol_vals[NUM_METRICS];
        float ratios[NUM_METRICS];

        float cur_rel_pow = get_rel_pow(stats->external_data, stats->external_data->pow_w[old_id]);
        float init_rel_pow = get_rel_pow(stats->external_data,
                                         stats->external_data->init_pow_w[old_id]);

        // TODO: Generalize ratio to a more general projection function
        populate_vals(stats->cur_latency_data.latency_ms, stats->cur_latency_data.size_bytes,
                      stats->external_data->pow_w[old_id], stats->eval_data->iou[old_id],
                      cur_rel_pow, cur_vals);
        populate_vals(stats->cur_latency_data.network_ms, stats->cur_latency_data.size_bytes,
                      stats->external_data->pow_w[old_id], stats->eval_data->iou[old_id],
                      cur_rel_pow, cur_vol_vals);
        populate_vals(stats->init_latency_data[old_id].network_ms,
                      stats->init_latency_data[old_id].size_bytes,
                      stats->external_data->init_pow_w[old_id], stats->eval_data->init_iou[old_id],
                      init_rel_pow, init_vol_vals);

        for (int metric_id = 0; metric_id < NUM_METRICS; ++metric_id) {
            const float init_vol_val = init_vol_vals[metric_id];

            if (init_vol_val == 0.0f) {
                // local device => do not scale vol value
                ratios[metric_id] = 1.0f;
            } else {
                ratios[metric_id] = cur_vol_vals[metric_id] / init_vol_val;
            }

            log_frame_f(state->fd, state->last_frame_id, "cs_select_ratio",
                        METRIC_NAMES[metric_id], ratios[metric_id]);
            SLOGI(SLOG_SELECT_3,
                  "SELECT | Select | Metric  %16s (%d): cur vol %11.3f, init vol %11.3f, ratio %11.3f",
                  METRIC_NAMES[metric_id], metric_id, cur_vol_vals[metric_id], init_vol_val,
                  ratios[metric_id]);
        }

        for (int id = 0; id < NUM_CONFIGS; ++id) {
            indexed_metrics_t proj_init_metrics;
            proj_init_metrics.codec_id = id;
            float rel_pow = get_rel_pow(stats->external_data, stats->external_data->init_pow_w[id]);

            populate_vals(stats->init_latency_data[id].latency_ms,
                          stats->init_latency_data[id].size_bytes,
                          stats->external_data->init_pow_w[id], stats->eval_data->init_iou[id],
                          rel_pow, proj_init_metrics.vals);
            populate_vals(stats->init_latency_data[id].network_ms,
                          stats->init_latency_data[id].size_bytes,
                          stats->external_data->init_pow_w[id], stats->eval_data->init_iou[id],
                          rel_pow, proj_init_metrics.vol_vals);

            for (int constr_id = 0; constr_id < NUM_CONSTRAINTS; ++constr_id) {
                const constraint_t constraint = state->constraints[constr_id];
                if (constraint.type == CONSTR_VOLATILE) {
                    const metric_t metric = constraint.metric;
                    const float init_vol_val = proj_init_metrics.vol_vals[metric];
                    float new_val = proj_init_metrics.vals[metric];
                    float new_vol_val = proj_init_metrics.vol_vals[metric];

                    if (id == old_id) {
                        new_val = cur_vals[metric];
                        new_vol_val = cur_vol_vals[metric];
                    }

                    const float static_val = new_val - new_vol_val;

                    if (id != LOCAL_CODEC_ID) {
                        // local-only mode doesn't have a network-volatile part
                        new_vol_val = init_vol_val * ratios[metric];
                    }

                    new_val = static_val + new_vol_val;

                    if (metric == METRIC_IOU) {
                        // TODO: Abstract this away, putting min/max values of IoU to constraint_t
                        new_val = fmax(fmin(1.0f, new_val), 0.0f);
                    }

                    SLOGI(SLOG_SELECT_3,
                          "SELECT | Select | Codec %2d: Metric %d, init %11.3f, new val %11.3f, vol val %11.3f, off %+11.3f",
                          id, metric, proj_init_metrics.vals[metric], new_val, new_vol_val,
                          new_val - proj_init_metrics.vals[metric]);

                    proj_init_metrics.vol_vals[metric] = new_vol_val;
                    proj_init_metrics.vals[metric] = new_val;
                }
            }

            metrics_product(state, &proj_init_metrics);
            sorted_metrics[id] = proj_init_metrics;
        }

        print_constraints(state->constraints);

        qsort(sorted_metrics, NUM_CONFIGS, sizeof(sorted_metrics[0]), cmp_metrics);
        int new_codec_id = select_codec(state, state->constraints, sorted_metrics);

        if (old_id != LOCAL_CODEC_ID && new_codec_id != LOCAL_CODEC_ID) {
            stats->cur_remote_avg_fill_ping_ms = stats->cur_avg_fill_ping_ms;
            stats->cur_remote_avg_ping_ms = stats->cur_avg_ping_ms;
        }

        if (!state->lock_codec) {
            state->id = new_codec_id;
        }

        // just prints
        SLOGI(SLOG_SELECT,
              "SELECT | Select | Codec %2d -> %2d; avg: ping %6.1f ms, fps %6.2f, latency %6.1f ms, iou %5.3f, pow %5.3f W",
              old_id, state->id, stats->external_data->ping_ms[old_id], stats->cur_latency_data.fps,
              stats->cur_latency_data.latency_ms, stats->eval_data->iou[old_id],
              stats->external_data->pow_w[old_id]);
    }

    // signal that we just performed a codec selection
    codec_selected = true;

    duration_ms = (float) (get_timestamp_ns() - start_ns) / 1e6f;
    SLOGI(SLOG_SELECT_DBG, "SELECT | Selecting codec took %6.3f ms", duration_ms);
    if (state->enable_profiling) {
        log_frame_f(state->fd, state->last_frame_id, "cs_select", "duration_ms", duration_ms);
    }

    cleanup:
    state->codec_selected = codec_selected;
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

bool drain_codec_selected(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    bool codec_selected = state->codec_selected;
    state->codec_selected = false;
    pthread_mutex_unlock(&state->lock);
    return codec_selected;
}

int64_t get_latency_offset_ms(codec_select_state_t *state) {
    if (NUM_CALIB_IOU_SAMPLES > 0) {
        pthread_mutex_lock(&state->lock);
        int64_t latency_offset_ms = state->latency_offset_ms;
        pthread_mutex_unlock(&state->lock);
        return latency_offset_ms;
    } else {
        return 0;
    }
}

void signal_eval_start(codec_select_state_t *state, int frame_index, int codec_id) {
    pthread_mutex_lock(&state->lock);
    state->stats.eval_data->id = codec_id;
    state->stats.eval_data->frame_id = frame_index;
    SLOGI(SLOG_EVAL, "SELECT | Signal Eval | start | Frame %d, codec %d", frame_index, codec_id);
    pthread_mutex_unlock(&state->lock);
}

void signal_eval_finish(codec_select_state_t *state, float iou) {
    pthread_mutex_lock(&state->lock);

    eval_data_t *eval_data = state->stats.eval_data;
    int codec_id = eval_data->id;

    int *iou_nsamples;
    float *iou_avg;

    if (state->stage != STAGE_CALIB_DRY && state->stage != STAGE_CALIB_NO_IOU) {
        if (is_calibrating(state)) {
            iou_nsamples = &eval_data->init_iou_nsamples[codec_id];
            iou_avg = &eval_data->init_iou[codec_id];
        } else {
            iou_nsamples = &eval_data->iou_nsamples[codec_id];
            iou_avg = &eval_data->iou[codec_id];
        }

        if (iou >= 0.0f) {
            if (is_calibrating(state)) {
                incr_avg(iou, iou_avg, iou_nsamples);
            } else {
                ewma(IOU_ALPHA, iou, iou_avg);
            }
        }

        SLOGI(SLOG_EVAL,
              "SELECT | Signal Eval | Finish | Codec %2d, iou %5.3f, avg iou %5.3f, %d samples",
              codec_id, iou, *iou_avg, *iou_nsamples);
    }

    if (state->enable_profiling) {
        const int64_t eval_ts_ns = get_timestamp_ns();
        log_frame_i64(state->fd, eval_data->frame_id, "cs_eval", "ts_ns", eval_ts_ns);
        log_frame_f(state->fd, eval_data->frame_id, "cs_eval", "iou", iou);
        if (state->stage != STAGE_CALIB_DRY && state->stage != STAGE_CALIB_NO_IOU) {
            log_frame_f(state->fd, eval_data->frame_id, "cs_eval", "avg_iou", *iou_avg);
        }
    }

    pthread_mutex_unlock(&state->lock);
}

void push_external_pow(codec_select_state_t *state, int64_t ts_ns, int amp, int volt) {
    SLOGI(SLOG_EXTERNAL_POW, "SELECT | Push External | ts %ld, %d amp, %d volt", ts_ns, amp, volt);
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
    SLOGI(SLOG_EXTERNAL_PING, "SELECT | Push External | Ping %6.1f ms", ping_ms);
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

void signal_last_frame(codec_select_state_t *state) {
    pthread_mutex_lock(&state->lock);
    state->got_last_frame = true;
    pthread_mutex_unlock(&state->lock);
}

#ifdef __cplusplus
}
#endif
