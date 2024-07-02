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
#define NUM_CONFIGS 10

/**
 * Size of external stats storage (should be set such that slow update_stats() doesn't cause buffer
 * overflows when pushing new results from Java)
 */
#define NUM_EXTERNAL_SAMPLES 128

/**
 * How many constraints we are placing on variables driving the codec selection.
 */
#define NUM_CONSTRAINTS 5

/**
 * How many metrics are considered per codec (should correspond to metric_t variants)
 */
#define NUM_METRICS 5

/**
 * ID of a local-only "codec" pointing at CONFIGS
 */
#define LOCAL_CODEC_ID 0

// scaling values to multiply metrics when calculating product (to bring the values into some normal range)
const float LATENCY_SCALE = 1.0f / 1e3f;
const float POWER_SCALE = 1.0f;
const float IOU_SCALE = 1.0f;
const float SIZE_SCALE = 1.0f / 1e6f;
const float SIZE_SCALE_LOG10 = 1.0f / 1e3f;

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
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {.i_frame_interval = 2, .framerate = 5, .bitrate =
        36864 * 90 * 5}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {.i_frame_interval = 2, .framerate = 5, .bitrate =
        36864 * 20 * 5}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {.i_frame_interval = 1, .framerate = 5, .bitrate =
        36864 * 20 * 5}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {.i_frame_interval = 1, .framerate = 5, .bitrate = 3000 }}},

};

/**
 * Which variables to track
 */
typedef enum {
    METRIC_LATENCY_MS, METRIC_SIZE_BYTES, METRIC_SIZE_BYTES_LOG10, METRIC_POWER_W, METRIC_IOU,
} metric_t;

/**
 * Whether the variable should be maximized (e.g., IoU), or minimized (e.g., power, latency)
 */
typedef enum {
    OPT_MIN, OPT_MAX,
} optimization_t;

/**
 * Type of the constraint
 */
typedef enum {
    CONSTR_HARD,  // Hard constraints need to fit below/over a certain limit, not counted into the metric product
    CONSTR_SOFT,  // Soft constraints do not have a limit, they are used to calculate the product to select the best codec
} constraint_type_t;

/**
 * Umbrella type for hard and soft constraints
 */
typedef struct {
    metric_t metric;
    optimization_t optimization;
    constraint_type_t type;
    float limit;  // limit for hard constraints, ignored with soft constraints
    float scale;  // scale for calculating the metric product to ensure it's in a reasonable range
} constraint_t;

/**
 * Managing data from the quality eval pipeline, modified via callbacks from the eval
 */
typedef struct {
    int id; // Codec ID for which eval pipeline is running. Since only one eval pipeline can run
    // at a time, this correctly reflects the one stored in eval_ctx. To be used only in
    // signal_eval_finish().
    int init_iou_nsamples[NUM_CONFIGS];
    float init_iou[NUM_CONFIGS];
    float iou[NUM_CONFIGS];
} eval_data_t;

/**
 * Data coming from the Java side, such as power
 */
typedef struct {
    int64_t prev_frame_stop_ts_ns;
    int inext;
    int istart;
    int64_t ts_ns[NUM_EXTERNAL_SAMPLES];
    int amp[NUM_EXTERNAL_SAMPLES];
    int volt[NUM_EXTERNAL_SAMPLES];
    int init_pow_w_nsamples[NUM_CONFIGS];
    float init_pow_w[NUM_CONFIGS];
    float pow_w[NUM_CONFIGS];
} external_data_t;

/**
 * Runtime breakdown of different kernels
 */
typedef struct {
    float enc;
    float dec;
    float dnn;
    float postprocess;
    float reconstruct;
} kernel_times_ms_t;

/**
 * Data related to latency
 */
typedef struct {
    kernel_times_ms_t kernel_times_ms;
    float total_ms;          // total end-to-end latency
    float network_ms;        // total latency without kernel times
    float size_bytes;        // size of the encoded frame
    float size_bytes_log10;  // log10 of size_bytes
    float fps;               // frequency of receiving frames
} latency_data_t;

/**
 * Statistics collected for every frame, such as the processing time and inference accuracy
 */
typedef struct {
    int prev_id;  // codec ID of the last frame entering update_stats()
    int64_t last_ping_ts_ns;
    float ping_ms;
    float ping_ms_avg;
    int init_nsamples_prev[NUM_CONFIGS];
    int init_nsamples[NUM_CONFIGS];
    int init_nruns[NUM_CONFIGS];
    int init_ping_nsamples;
    float init_ping_ms_avg;
//    float init_latency_ms[NUM_CONFIGS];
    int64_t last_received_image_ts_ns;  // TODO: Could be probably merged with state.last_timestamp_ns
    latency_data_t init_latency_data[NUM_CONFIGS];
    int cur_nsamples;
    latency_data_t cur_latency_data;
//    float cur_latency_ms;
//    float latency_ms_avg[NUM_CONFIGS];
    eval_data_t *eval_data;
    external_data_t *external_data;
} codec_stats_t;

/**
 * Stores all data required to perform codec selection
 */
typedef struct {
    pthread_mutex_t lock;
    collected_events_t *collected_events;
    codec_stats_t stats;
    bool local_only;
    int id;  // the currently active codec; points at CONFIGS
    int64_t since_last_select_ms;
    int64_t last_timestamp_ns;
    bool is_calibrating;
    float tgt_latency_ms;  // latency target to aim for
    int init_sorted_ids[NUM_CONFIGS];  // IDs of init_latency_ms sorted by latency
    bool is_allowed[NUM_CONFIGS];  // If the codec is allowed to be used or not (based on init devices)
    constraint_t constraints[NUM_CONSTRAINTS]; // Constraints considered for the codec selection
} codec_select_state_t;

/**
 * Values for each metric indexed by codec ID, used for sorting.
 *
 * Additinally contains other information, like the product of the metrics and whether it fits the
 * constraints.
 */
typedef struct {
    int codec_id;
    float vals[NUM_METRICS];
    float product;
    bool fits_constraints;
} indexed_metrics_t;

void init_codec_select(int config_flags, codec_select_state_t **state);

void destroy_codec_select(codec_select_state_t **state);

void update_stats(const frame_metadata_t *frame_metadata, const eval_pipeline_context_t *eval_ctx,
                  codec_select_state_t *state);

void
select_codec_manual(device_type_enum device_index, int do_segment, compression_t compression_type,
                    int quality, int rotation, codec_config_t *state);

void select_codec_auto(codec_select_state_t *state);

int get_codec_id(codec_select_state_t *state);
int get_codec_sort_id(codec_select_state_t *state);
codec_params_t get_codec_params(codec_select_state_t *state);

void signal_eval_start(codec_select_state_t *state, int codec_id);
void signal_eval_finish(codec_select_state_t *state, float iou);

void push_external_stats(codec_select_state_t *state, int64_t ts_ns, int amp, int volt);
void push_external_ping(codec_select_state_t *state, float ping_ms);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_CODEC_SELECT_H
