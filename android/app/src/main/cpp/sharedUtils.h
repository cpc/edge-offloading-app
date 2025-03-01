//
// Created by rabijl on 21.4.2023.
//

#ifndef POCL_AISA_DEMO_SHAREDUTILS_H
#define POCL_AISA_DEMO_SHAREDUTILS_H

#include "platform.h"
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// enable this to print timing to logs
//#define PRINT_PROFILE_TIME

/**
 * when using this, don't forget to define LOGTAG before hand
 */
#define CHECK_AND_RETURN(ret, msg)                                          \
    if(ret != CL_SUCCESS) {                                                 \
        LOGE("ERROR: %s at %s:%d returned with %d\n",                       \
             msg, __FILE__, __LINE__, ret);                                 \
        return ret;                                                         \
    }

/**
 * when using this, don't forget to define LOGTAG before hand
 */
#define CHECK_AND_RETURN_NULL(status, ret, msg)                             \
    do{                                                                     \
    if(status != CL_SUCCESS) {                                              \
        LOGE("ERROR: %s at %s:%d returned with %d\n",                       \
             msg, __FILE__, __LINE__, status);                              \
        *ret = status;                                                      \
        return NULL;                                                        \
    }                                                                       \
}while(0)

#define  PRINT_DIFF(diff, msg)                                              \
    LOGE("%s took this long: %lu ms, %lu ns\n",                           \
        msg, ((diff) / 1000000), (diff) % 1000000);                         \

#define VAR_NAME(var) #var

#define COND_REL_MEM(memobj) \
    if(NULL != memobj) { \
    clReleaseMemObject(memobj); \
    memobj = NULL; \
    }                        \

#define COND_REL_KERNEL(kernel) \
    if(NULL != kernel) { \
    clReleaseKernel(kernel); \
    kernel = NULL; \
    }

#define COND_REL_QUEUE(queue) \
    if(NULL != queue) { \
    clReleaseCommandQueue(queue); \
    queue = NULL; \
    }                         \


int64_t get_timestamp_ns();

void
add_ns_to_time(struct timespec *const ts, const long increment);

long
compare_timespec(const struct timespec *const ts_a, const struct timespec *const ts_b);

int64_t get_diff_timespec(const struct timespec *const start,
                          const struct timespec *const stop);

/** Helpers for logging values into a file **/

void log_frame_int(int fd, int frame_index, const char* tag, const char* parameter, int value);
void log_frame_i64(int fd, int frame_index, const char* tag, const char* parameter, int64_t value);
void log_frame_f(int fd, int frame_index, const char* tag, const char* parameter, float value);
void log_frame_str(int fd, int frame_index, const char *tag, const char *parameter, const char *value);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_SHAREDUTILS_H
