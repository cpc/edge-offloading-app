/** Network model API
 */

#ifndef POCL_AISA_DEMO_EVAL_NETWORK_MODEL_H
#define POCL_AISA_DEMO_EVAL_NETWORK_MODEL_H

#include "eval.h"

/**************************************************************************************************/

// Helper functions

float ewma(float new_x, float old_x, float alpha);

// Ringbuffer-related definitions

// Size of the ringbuffer, enough for 10 seconds of samples running at 100 FPS
#define MAX_NETWORK_NSAMPLES 1024

// How old samples to keep in the network stats
static const int64_t MAX_SAMPLE_AGE_NS = 5000000000;  // 5 seconds

// Ringbuffer struct for holding the last X seconds worth of samples
typedef struct {
    float x[MAX_NETWORK_NSAMPLES];
    float y[MAX_NETWORK_NSAMPLES];
    int64_t timestamp_ns[MAX_NETWORK_NSAMPLES];  // needed to know which samples to remove
    int pos;
} ringbuffer_t;

/**************************************************************************************************/

// Incremental least squares fitting (ILSF) data structures
// https://blog.demofox.org/2016/12/22/incremental-least-squares-curve-fitting

typedef struct {
    // generic data for 2-dimensional ILSF
    float ATA[2][2];
    float ATY[2];
    // other data
    ringbuffer_t *ringbuffer;
} ilsf_data_t;

typedef struct {
    float slope;
    float offset;
} ilsf_params_t;

// Log(size)-Linear model interpolating per-codec exponentially-weighted moving average (EWMA)

typedef struct {
    float mean_rtt_ms[NUM_CODEC_POINTS];
    float mean_rtt_prev_ms[NUM_CODEC_POINTS];
    float mean_size_bytes[NUM_CODEC_POINTS];
    float nsamples[NUM_CODEC_POINTS];
    float alpha_agile;
    float alpha_stable;
    int last_codec_id;
} llewma_data_t;

typedef struct {
    float slope;
    float offset;
} llewma_params_t;

// General API data structures

typedef union {
    ilsf_data_t ilsf;
    llewma_data_t llewma;
} network_model_data_t;

typedef union {
    ilsf_params_t ilsf;
    llewma_params_t llewma;
} network_model_params_t;

/**************************************************************************************************/

// ILSF

void ilsf_add_sample(float x, float y, network_model_data_t *data);

void ilsf_update_params(network_model_data_t *data, network_model_params_t *params);

float ilsf_predict(const network_model_params_t *params, float x);

void ilsf_print_params(const network_model_params_t *params, int frame_index, int loglevel);

void ilsf_log_params(const network_model_params_t *params, int frame_index, int fd);

// LLEWMA

void llewma_add_sample(float x, float y, network_model_data_t *data);

void llewma_update_params(network_model_data_t *data, network_model_params_t *params);

float llewma_predict(const network_model_params_t *params, float x);

void llewma_print_params(const network_model_params_t *params, int frame_index, int loglevel);

void llewma_log_params(const network_model_params_t *params, int frame_index, int fd);

// General API

const char NETWORK_NAMES_STR[2][16] = {"ILSF", "LLEWMA"};

typedef enum {
    ILSF,
    LLEWMA,
} network_model_name_t;

typedef struct {
    network_model_name_t name;
    network_model_data_t data;
    network_model_params_t params;

    void (*add_sample)(float x, float y, network_model_data_t *data);

    void (*update_params)(network_model_data_t *data, network_model_params_t *params);

    float (*predict)(const network_model_params_t *params, float x);

    void (*print_params)(const network_model_params_t *params, int frame_index, int loglevel);

    void (*log_params)(const network_model_params_t *params, int frame_index, int fd);
} network_model_t;

network_model_t *init_network_model(network_model_name_t chosen_model);

void destroy_network_model(network_model_t *model);

#endif //POCL_AISA_DEMO_EVAL_NETWORK_MODEL_H
