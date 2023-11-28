//
// Created by rabijl on 11/23/23.
//

#ifndef POCL_AISA_DEMO_QUALITY_ALGORITHM_H
#define POCL_AISA_DEMO_QUALITY_ALGORITHM_H

#include "event_logger.h"
#include "poclImageProcessor.h"

int
evaluate_parameters(float energy, event_array_t *array, event_array_t *eval_array, compression_t *compression);

#endif //POCL_AISA_DEMO_QUALITY_ALGORITHM_H
