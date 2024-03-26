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

/**
 * compare two timespecs. If ts_a is larger, this returns a positive number.
 * else a negative number. If both are equal, zero is returned.
 * @param ts_a 
 * @param ts_b
 * @return
 */
long
compare_timespec(struct timespec *const ts_a, struct timespec *const ts_b) {
    long diff = ts_a->tv_sec - ts_b->tv_sec;
    if (0 == diff) {
        return ts_a->tv_nsec - ts_b->tv_nsec;
    }
    return diff;

}

#ifdef __cplusplus
}
#endif