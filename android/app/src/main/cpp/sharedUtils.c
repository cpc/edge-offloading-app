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

/**
 * helper function to add nanoseconds to a timespec
 * @param ts pointer to timespec to increment
 * @param increment in nanoseconds
 */
void
add_ns_to_time(struct timespec *const ts, const int increment) {
    ts->tv_nsec += increment;
    // nsec can not be larger than a second
    if (ts->tv_nsec >= 1000000000) {
        ts->tv_sec += 1;
        ts->tv_nsec -= 1000000000;
    }
}

#ifdef __cplusplus
}
#endif