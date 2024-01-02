//
// Created by rabijl on 11/23/23.
//

#ifndef POCL_AISA_DEMO_QUALITY_ALGORITHM_H
#define POCL_AISA_DEMO_QUALITY_ALGORITHM_H

#include "event_logger.h"
#include "poclImageProcessor.h"

int evaluate_parameters(int frame_index, float power, float iou, uint64_t size_bytes, const event_array_t *event_array,
                        const event_array_t *eval_event_array, const host_ts_ns_t *host_ts_ns,
                        compression_t *compression);

#endif //POCL_AISA_DEMO_QUALITY_ALGORITHM_H
