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

/** Different logging levels */
#define SLOG_SELECT                (1 << 0)
#define SLOG_SELECT_2              (1 << 1)
#define SLOG_SELECT_DBG            (1 << 2)
#define SLOG_CONSTR                (1 << 3)
#define SLOG_UPDATE_DBG            (1 << 4)
#define SLOG_COLLECT_LATENCY_DBG   (1 << 5)
#define SLOG_COLLECT_EXTERNAL_DBG  (1 << 6)
#define SLOG_EVAL                  (1 << 7)
#define SLOG_EXTERNAL_POW          (1 << 8)
#define SLOG_EXTERNAL_PING         (1 << 9)
#define SLOG_DBG                   (1 << 31)

/** Verbosity of messages printed from this file (0 turns them off) */
#define SELECT_VERBOSITY (0)

/** Simple LOGI wrapper to reduce clutter */
#define SLOGI(verbosity, ...) do { if (SELECT_VERBOSITY & (verbosity)) { LOGI(__VA_ARGS__); } } while (0)

/**
 * Number of codec configs considered by the selection algorithm (should be >= 1 to always have at
 * least local execution).
 */
#define NUM_CONFIGS 8

/**
 * Size of external stats storage (should be set such that slow update_stats() doesn't cause buffer
 * overflows when pushing new results from Java)
 */
#define NUM_EXTERNAL_SAMPLES 256

/**
 * How many constraints we are placing on variables driving the codec selection.
 */
#define NUM_CONSTRAINTS 5

/**
 * How many metrics are considered per codec (should correspond to metric_t variants)
 */
#define NUM_METRICS 6

/**
 * ID of a local-only "codec" pointing at CONFIGS
 */
#define LOCAL_CODEC_ID 0

// scaling values to multiply metrics when calculating product (to bring the values into some normal range)
const float LATENCY_SCALE = 1.0f / 1e3f;
const float POWER_SCALE = 1.0f;
const float IOU_SCALE = 1.0f;
const float REL_POWER_SCALE = 1.0f;
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
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 80}}},
        {.compression_type = JPEG_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.jpeg = {.quality = 20}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {
                .i_frame_interval = 1, .framerate = 1, .bitrate = 5000000}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {
                .i_frame_interval = 1, .framerate = 1, .bitrate = 250000}}},
        {.compression_type = HEVC_COMPRESSION, .device_type= REMOTE_DEVICE, .config = {.hevc = {
                .i_frame_interval = 1, .framerate = 1, .bitrate = 10000}}},
};

/**
 * Which variables to track
 */
typedef enum {
    METRIC_LATENCY_MS,
    METRIC_SIZE_BYTES,
    METRIC_SIZE_BYTES_LOG10,
    METRIC_POWER_W,
    METRIC_IOU,
    METRIC_REL_POW,
} metric_t;

const char *const METRIC_NAMES[NUM_METRICS] = {"latency_ms", "size_bytes", "size_bytes_log10",
                                               "power_w", "iou", "rel_pow"};

/**
 * Whether the variable should be maximized (e.g., IoU), or minimized (e.g., power, latency)
 */
typedef enum {
    OPT_MIN, OPT_MAX,
} optimization_t;

const char *const OPT_NAMES[2] = {"min", "max"};

/**
 * Type of the constraint
 */
typedef enum {
    CONSTR_HARD,  // Hard constraints need to fit below/over a certain limit, not counted into the metric product
    CONSTR_HARD_VOLATILE,  // Same as hard + has a volatile, network-dependent part (e.g. latency)
    CONSTR_SOFT,  // Soft constraints do not have a limit, they are used to calculate the product to select the best codec
} constraint_type_t;

const char *const CONSTR_TYPE_NAMES[3] = {"hard", "hard_volatile", "soft"};

/**
 * Umbrella type for hard and soft constraints
 * TODO: This could be a union to prevent repeating redundant fields
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
    int frame_id; // Frame index of a frame that started the eval pipeline
    // at a time, this correctly reflects the one stored in eval_ctx. To be used only in
    // signal_eval_finish().
    int init_iou_nsamples[NUM_CONFIGS];
    float init_iou[NUM_CONFIGS];
    int iou_nsamples[NUM_CONFIGS];
    float iou[NUM_CONFIGS];
} eval_data_t;

typedef struct {
    float sum;
    float sumsq;
} sum_data_t;

/**
 * Data coming from the Java side, such as power or ping
 */
typedef struct {
    int64_t prev_frame_stop_ts_ns;
    int inext;
    int istart;
    // data coming from external source:
    int64_t ts_ns[NUM_EXTERNAL_SAMPLES];
    int ext_amp[NUM_EXTERNAL_SAMPLES];
    int ext_volt[NUM_EXTERNAL_SAMPLES];
    float ext_ping_ms[NUM_EXTERNAL_SAMPLES];
    // collected statistics:
    int init_pow_w_nsamples[NUM_CONFIGS];
    float init_pow_w[NUM_CONFIGS];
    sum_data_t init_pow_w_sum[NUM_CONFIGS];
    int pow_w_nsamples[NUM_CONFIGS];
    float pow_w[NUM_CONFIGS];
    sum_data_t pow_w_sum[NUM_CONFIGS];
    int init_ping_ms_nsamples[NUM_CONFIGS];
    float init_ping_ms[NUM_CONFIGS];
    int ping_ms_nsamples[NUM_CONFIGS];
    float ping_ms[NUM_CONFIGS];
    int min_pow_w_id; // ID of codec with minimum mean power; Updated per select_codec()
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
    float total_ms;          // total run time of this codec since the last change
    float latency_ms;        // total end-to-end latency
    float network_ms;        // total latency without kernel times
    float size_bytes;        // size of the encoded frame
    float size_bytes_log10;  // log10 of size_bytes
    float fps;               // frequency of receiving frames
} latency_data_t;

/**
 * Statistics collected for every frame, such as the processing time and inference accuracy
 */
typedef struct {
    int prev_frame_codec_id;  // codec ID of the last frame entering update_stats()
    int init_nsamples_prev[NUM_CONFIGS];
    int init_nsamples[NUM_CONFIGS];
    int init_nruns[NUM_CONFIGS];
    int64_t last_received_image_ts_ns;  // TODO: Could be probably merged with state.last_timestamp_ns
    latency_data_t init_latency_data[NUM_CONFIGS];
    int cur_nsamples;
    latency_data_t cur_latency_data;
    eval_data_t *eval_data;
    external_data_t *external_data;
    float cur_init_avg_ping_ms;         // average ping during calibration
    float cur_avg_ping_ms;              // average ping
    float cur_remote_avg_ping_ms;       // average ping only calculated when offloading (i.e., not local)
    float cur_init_avg_fill_ping_ms;    // average fillbuffer ping during calibration
    float cur_avg_fill_ping_ms;         // average fillbuffer ping
    float cur_remote_avg_fill_ping_ms;  // average fillbuffer ping only calculated when offloading (i.e., not local)
    float cur_init_avg_pow_w;           // average power during calibration
    float cur_avg_pow_w;                // average power
} codec_stats_t;

/**
 * Stores all data required to perform codec selection
 */
typedef struct {
    pthread_mutex_t lock;
    collected_events_t *collected_events;
    codec_stats_t stats;
    bool local_only;
    bool is_calibrating;
    bool is_dry_run;
    bool enable_profiling;
    bool lock_codec;  // after calibration, lock the selected codec and never change it
    bool sync_with_input;
    bool got_last_frame;
    int fd;  // file descriptor of a log file
    int last_frame_id;  // Frame index of the frame that is being or was last logged into the stats
    int id;  // the currently active codec; points at CONFIGS
    bool codec_selected; // signals that codec selection was performed (the ID could stay the same)
    int64_t latency_offset_ms; // signals latency offset to spend sleeping at the end of the frame
    int64_t since_last_select_ms;
    int64_t last_timestamp_ns;
    int64_t calib_end_ns;  // timestamp of the end of calibration
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
    float vals[NUM_METRICS];
    float vol_vals[NUM_METRICS];  // part of val which is volatile, e.g., due to network
    bool violates_any_constr[NUM_METRICS];  // whether each metric in `vals` violates a constraint
    bool all_fit_constraints;  // all metrics fit the constraints
    int codec_id;
    float product;
} indexed_metrics_t;

void
init_codec_select(int config_flags, int fd, int do_algorithm, bool lock_codec, bool sync_with_input,
                  codec_select_state_t **state);

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
bool drain_codec_selected(codec_select_state_t *state);
int64_t get_latency_offset_ms(codec_select_state_t *state);

void signal_eval_start(codec_select_state_t *state, int frame_index, int codec_id);
void signal_eval_finish(codec_select_state_t *state, float iou);

void push_external_pow(codec_select_state_t *state, int64_t ts_ns, int amp, int volt);
void push_external_ping(codec_select_state_t *state, int64_t ts_ns, float ping_ms);

void signal_last_frame(codec_select_state_t *state);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_CODEC_SELECT_H
