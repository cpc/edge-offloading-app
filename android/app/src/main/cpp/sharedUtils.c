//
// Created by rabijl on 8.3.2024.
//

#include "sharedUtils.h"
#include <time.h>
#include <stdint.h>

#define NS_PER_S 1000000000

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
add_ns_to_time(struct timespec *const ts, const long increment) {

  int64_t time = ts->tv_nsec + increment;
  ts->tv_nsec = time % NS_PER_S;
  ts->tv_sec += time / NS_PER_S;
}

#ifdef __cplusplus
}
#endif