//
// Created by rabijl on 8.3.2024.
//

#include "sharedUtils.h"
#include <time.h>
#include <stdint.h>
#include <stdio.h>

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
compare_timespec(const struct timespec *const ts_a, const struct timespec *const ts_b) {
    long diff = ts_a->tv_sec - ts_b->tv_sec;
    if (0 == diff) {
        return ts_a->tv_nsec - ts_b->tv_nsec;
    }
    return diff;

}

/**
 * get the difference between two timespecs, substract the first from the second
 * @param start the spec to substract with
 * @param stop the spec to substract from
 * @return signed difference
 */
int64_t get_diff_timespec(const struct timespec *const start,
                          const struct timespec *const stop) {
  int64_t diff = stop->tv_sec - start->tv_sec;
  int64_t diff_ns = stop->tv_nsec - start->tv_nsec;

  return diff * 1000000000 + diff_ns;
}

void log_frame_int(int fd, int frame_index, const char *tag, const char *parameter, int value) {
    dprintf(fd, "%d,%s,%s,%d\n", frame_index, tag, parameter, value);
}

void log_frame_i64(int fd, int frame_index, const char *tag, const char *parameter, int64_t value) {
    dprintf(fd, "%d,%s,%s,%ld\n", frame_index, tag, parameter, value);
}

void log_frame_f(int fd, int frame_index, const char *tag, const char *parameter, float value) {
    dprintf(fd, "%d,%s,%s,%f\n", frame_index, tag, parameter, value);
}

void
log_frame_str(int fd, int frame_index, const char *tag, const char *parameter, const char *value) {
    dprintf(fd, "%d,%s,%s,%s\n", frame_index, tag, parameter, value);
}

#ifdef __cplusplus
}
#endif