//
// Created by rabijl on 11/23/23.
//

#ifndef POCL_AISA_DEMO_QUALITY_ALGORITHM_H
#define POCL_AISA_DEMO_QUALITY_ALGORITHM_H

#include "event_logger.h"
#include "poclImageProcessor.h"

// ID of the currently selected codec
extern int CUR_CODEC_ID;

// Collect statistics about the loop execution and when it's the right time, decide which codec to run next
int evaluate_parameters(int frame_index, float power, float iou, uint64_t size_bytes,
                        int file_descriptor, int config_flags, const event_array_t *event_array,
                        const event_array_t *eval_event_array, const host_ts_ns_t *host_ts_ns,
                        codec_config_t *config);

#endif //POCL_AISA_DEMO_QUALITY_ALGORITHM_H
