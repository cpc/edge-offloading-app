//
// Created by rabijl on 8.3.2024.
//

#include "sharedUtils.h"
#include <time.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

int64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

#ifdef __cplusplus
}
#endif