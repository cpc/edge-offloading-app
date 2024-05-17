#include <cmath>

#include "eval_network_model.h"
#include "sharedUtils.h"

float ewma(float new_x, float old_x, float alpha) {
    return alpha * old_x + (1.0f - alpha) * new_x;
}

// ILSF

void ilsf_add_sample(float x, float y, network_model_data_t *data) {
    x = log10f(x);

    // add the sample to the network model
    data->ilsf.ATA[0][0] += 1.0f;
    data->ilsf.ATA[0][1] += x;
    data->ilsf.ATA[1][0] += x;
    data->ilsf.ATA[1][1] += x * x;

    data->ilsf.ATY[0] += y;
    data->ilsf.ATY[1] += y * x;

    // add the sample to the ringbuffer
    data->ilsf.ringbuffer->pos = (data->ilsf.ringbuffer->pos + 1) % MAX_NETWORK_NSAMPLES;
    data->ilsf.ringbuffer->x[data->ilsf.ringbuffer->pos] = x;
    data->ilsf.ringbuffer->y[data->ilsf.ringbuffer->pos] = y;
    data->ilsf.ringbuffer->timestamp_ns[data->ilsf.ringbuffer->pos] = get_timestamp_ns();

    QLOGI(3, "QUALITY | DEBUG | added sample %8.0f B, %8.2f ms, pos %d, ts %ld\n",
          powf(10.0f, x), y, data->ilsf.ringbuffer->pos,
          data->ilsf.ringbuffer->timestamp_ns[data->ilsf.ringbuffer->pos]);

    // Remove samples from the ringbuffer and the network model older than X ns
    const int cur_pos = data->ilsf.ringbuffer->pos;
    const int64_t cur_timestamp_ns = data->ilsf.ringbuffer->timestamp_ns[cur_pos];
    QLOGI(3, "QUALITY | DEBUG | cur_pos %d, cur_ts %ld, older than %ld\n", cur_pos,
          cur_timestamp_ns,
          MAX_SAMPLE_AGE_NS);

    int nremoved = 0;
    int i = cur_pos - 1;
    while (data->ilsf.ringbuffer->timestamp_ns[i] != 0) {
        if (i < 0) {
            i = MAX_NETWORK_NSAMPLES - 1;
        }

        QLOGI(3, "QUALITY | DEBUG | ts[%d] %ld, diff %ld\n", i,
              data->ilsf.ringbuffer->timestamp_ns[i],
              cur_timestamp_ns - data->ilsf.ringbuffer->timestamp_ns[i]);

        if (i == cur_pos) {
            // do not remove the latest sample
            break;
        }

        int64_t ts = data->ilsf.ringbuffer->timestamp_ns[i];

        if (ts != 0 && (cur_timestamp_ns - ts) > MAX_SAMPLE_AGE_NS) {
            const float x = data->ilsf.ringbuffer->x[i];
            const float y = data->ilsf.ringbuffer->y[i];

            QLOGI(3, "QUALITY | DEBUG | -- removed\n");

            // remove data from network model
            data->ilsf.ATA[0][0] -= 1.0f;
            data->ilsf.ATA[0][1] -= x;
            data->ilsf.ATA[1][0] -= x;
            data->ilsf.ATA[1][1] -= x * x;

            data->ilsf.ATY[0] -= y;
            data->ilsf.ATY[1] -= y * x;

            // remove the sample from ringbuffer
            data->ilsf.ringbuffer->timestamp_ns[i] = 0;
            data->ilsf.ringbuffer->x[i] = 0.0f;
            data->ilsf.ringbuffer->y[i] = 0.0f;
            nremoved += 1;
        }

        i -= 1;
    }

    QLOGI(1, "QUALITY | ringbuf pos %d, removed %d samples\n", data->ilsf.ringbuffer->pos,
          nremoved);
}

void ilsf_update_params(network_model_data_t *data, network_model_params_t *params) {
    float a = data->ilsf.ATA[0][0]; // number of collected samples
    float b = data->ilsf.ATA[0][1];
    float c = data->ilsf.ATA[1][0];
    float d = data->ilsf.ATA[1][1];
    float det = (a * d) - (b * c);

    params->ilsf.offset = 0.0f;
    params->ilsf.slope = 0.0f;

    if (is_near_zero(det)) {
        if (!is_near_zero(a)) {
            params->ilsf.offset = data->ilsf.ATY[0] / a;
        }
    } else {
        float idet = 1.0f / det;
        float iATA[2][2] = {{d * idet,  -b * idet},
                            {-c * idet, a * idet}};

        params->ilsf.offset = iATA[0][0] * data->ilsf.ATY[0] + iATA[0][1] * data->ilsf.ATY[1];
        params->ilsf.slope = iATA[1][0] * data->ilsf.ATY[0] + iATA[1][1] * data->ilsf.ATY[1];
    }
}

float ilsf_predict(const network_model_params_t *params, float x) {
    return fmax(params->ilsf.slope * x + params->ilsf.offset, 0.0f);
}

void ilsf_print_params(const network_model_params_t *params, int frame_index, int loglevel) {
    QLOGI(loglevel, "QUALITY | frame %d | EVAL | ILSF model_slope: %.3f, model_offset: %.3f\n",
          frame_index, params->ilsf.slope, params->ilsf.offset);
}

void ilsf_log_params(const network_model_params_t *params, int frame_index, int fd) {
    dprintf(fd, "%d,quality_eval,model_name,ilsf\n", frame_index);
    dprintf(fd, "%d,quality_eval,model_slope,%f\n", frame_index, params->ilsf.slope);
    dprintf(fd, "%d,quality_eval,model_offset,%f\n", frame_index, params->ilsf.offset);
}

// LLEWMA

void llewma_add_sample(float x, float y, network_model_data_t *data) {
    data->llewma.nsamples[CUR_CODEC_ID] += 1;

    data->llewma.mean_size_bytes[CUR_CODEC_ID] = ewma(x, data->llewma.mean_size_bytes[CUR_CODEC_ID],
                                                  data->llewma.alpha_stable);

    data->llewma.mean_rtt_ms[CUR_CODEC_ID] = ewma(y, data->llewma.mean_rtt_ms[CUR_CODEC_ID],
                                              data->llewma.alpha_agile);

    if (data->llewma.last_codec_id != CUR_CODEC_ID) {
        data->llewma.last_codec_id = CUR_CODEC_ID;
    }

    QLOGI(3,
          "QUALITY | DEBUG | codec %d | add sample, size: %8.0f bytes (ewma %8.0f), rtt: %8.2f ms (ewma %8.2f)\n",
          CUR_CODEC_ID, x, data->llewma.mean_size_bytes[CUR_CODEC_ID], y,
          data->llewma.mean_rtt_ms[CUR_CODEC_ID]);
}

void llewma_update_params(network_model_data_t *data, network_model_params_t *params) {
    float rtt_ms = data->llewma.mean_rtt_ms[CUR_CODEC_ID];
    float prev_rtt_ms = data->llewma.mean_rtt_prev_ms[CUR_CODEC_ID];

    if (data->llewma.nsamples[CUR_CODEC_ID] <= 0) {
        // codec hasn't been ran yet and prev value thus contains bogus value => use the current estimate
        prev_rtt_ms = rtt_ms;
    }

    float diff = rtt_ms - prev_rtt_ms;

    QLOGI(3, "QUALITY | DEBUG | codec %d | prev rtt:     %8.2f ms\n", CUR_CODEC_ID, prev_rtt_ms);
    QLOGI(3, "QUALITY | DEBUG | codec %d | current rtt:  %8.2f ms\n", CUR_CODEC_ID, rtt_ms);
    QLOGI(3, "QUALITY | DEBUG | codec %d | rtt diff:     %8.2f ms\n", CUR_CODEC_ID, diff);

    float sizes_log10[NUM_CODEC_POINTS] = {0.0f};
    float adjusted_rtt_ms[NUM_CODEC_POINTS] = {0.0f};

    int n = 0;
    float rtt_ms_mean = 0.0f;
    float size_log10_mean = 0.0f;

    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        if ((data->llewma.nsamples[i] <= 0) || (CODEC_DEVICES[i] == 0)) {
            adjusted_rtt_ms[i] = rtt_ms;
            data->llewma.mean_rtt_prev_ms[i] = rtt_ms;
        } else {
            sizes_log10[i] = log10f(data->llewma.mean_size_bytes[i]);
            adjusted_rtt_ms[i] = data->llewma.mean_rtt_prev_ms[i] + diff;

            n += 1;
            rtt_ms_mean += (adjusted_rtt_ms[i] - rtt_ms_mean) / (float) (n);
            size_log10_mean += (sizes_log10[i] - size_log10_mean) / (float) (n);
        }
    }

    for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
        QLOGI(3,
              "QUALITY | DEBUG | codec %2d: size %8.0f bytes, rtt prev %8.2f ms (adjusted %8.3f)\n",
              i, powf(10.0f, sizes_log10[i]), data->llewma.mean_rtt_prev_ms[i], adjusted_rtt_ms[i]);
    }

    float slope = 0.0f;
    float offset = 0.0f;

    if ((n > 0) && !is_near_zero(rtt_ms_mean) && !is_near_zero(size_log10_mean)) {
        float dxdy = 0.0f;
        float dx_sq = 0.0f;

        for (int i = 0; i < NUM_CODEC_POINTS; ++i) {
            if (CODEC_DEVICES[i] == 0) {
                continue;
            }

            float dx = sizes_log10[i] - size_log10_mean;
            dxdy += dx * (adjusted_rtt_ms[i] - rtt_ms_mean);
            dx_sq += dx * dx;
        }

        slope = dxdy / dx_sq;
        offset = rtt_ms_mean - slope * size_log10_mean;
    }

    data->llewma.mean_rtt_prev_ms[CUR_CODEC_ID] = rtt_ms;

    params->llewma.slope = slope;
    params->llewma.offset = offset;
}

float llewma_predict(const network_model_params_t *params, float x) {
    return fmax(params->llewma.slope * x + params->llewma.offset, 0.0f);
}

void llewma_print_params(const network_model_params_t *params, int frame_index, int loglevel) {
    QLOGI(loglevel, "QUALITY | frame %d | EVAL | LLEWMA model_slope: %.3f, model_offset: %.3f\n",
          frame_index, params->llewma.slope, params->llewma.offset);
}

void llewma_log_params(const network_model_params_t *params, int frame_index, int fd) {
    dprintf(fd, "%d,quality_eval,model_name,llewma\n", frame_index);
    dprintf(fd, "%d,quality_eval,model_slope,%f\n", frame_index, params->llewma.slope);
    dprintf(fd, "%d,quality_eval,model_offset,%f\n", frame_index, params->llewma.offset);
}

network_model_t *init_network_model(network_model_name_t chosen_model) {
    network_model_t *model = (network_model_t *) (malloc(sizeof(network_model_t)));
    model->name = chosen_model;

    switch (chosen_model) {
        case ILSF: {
            ringbuffer_t *ringbuffer = (ringbuffer_t *) (malloc(sizeof(ringbuffer_t)));
            for (int i = 0; i < MAX_NETWORK_NSAMPLES; ++i) {
                ringbuffer->x[i] = 0.0f;
                ringbuffer->y[i] = 0.0f;
                ringbuffer->timestamp_ns[i] = 0;
            }
            ringbuffer->pos = -1;

            network_model_data_t init_ilsf_data = {
                    .ilsf = {
                            .ATA = {{0.0f, 0.0f},
                                    {0.0f, 0.0f}},
                            .ATY = {0.0f, 0.0f},
                            .ringbuffer = ringbuffer,
                    }
            };

            network_model_params_t init_ilsf_params = {.ilsf = {.slope = 0.0f, .offset = 0.0f}};

            model->data = init_ilsf_data;
            model->params = init_ilsf_params;
            model->add_sample = ilsf_add_sample;
            model->update_params = ilsf_update_params;
            model->predict = ilsf_predict;
            model->print_params = ilsf_print_params;
            model->log_params = ilsf_log_params;

            break;
        }
        case LLEWMA: {
            network_model_data_t init_llewma_data = {
                    .llewma = {
                            .mean_rtt_ms = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                            .mean_rtt_prev_ms = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                            .mean_size_bytes = {480061, 480061, 71000, 50000, 27000,
                                                21000}, // rough averages
                            .nsamples = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                            .alpha_agile = 0.2f,
                            .alpha_stable = 0.8f,
                            .last_codec_id = -1,
                    }
            };

            network_model_params_t init_llewma_params = {.llewma = {.slope = 0.0f, .offset = 0.0f}};

            model->data = init_llewma_data;
            model->params = init_llewma_params;
            model->add_sample = llewma_add_sample;
            model->update_params = llewma_update_params;
            model->predict = llewma_predict;
            model->print_params = llewma_print_params;
            model->log_params = llewma_log_params;
            LOGI("done");
            break;
        }
        default:
            LOGE("Tried to initialize unknown network model: %s\n",
                 NETWORK_NAMES_STR[chosen_model]);
            break;
    }

    return model;
}

void destroy_network_model(network_model_t *model) {
    if (model == NULL) {
        return;
    }

    switch (model->name) {
        case ILSF:
            free(model->data.ilsf.ringbuffer);
            break;
        case LLEWMA:
            break;
        default:
            LOGE("Tried to destroy unknown network model: %s\n", NETWORK_NAMES_STR[model->name]);
    }

    free(model);
}
