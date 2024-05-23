/* Misc testing code */

#ifndef POCL_AISA_DEMO_TESTAPPS_H
#define POCL_AISA_DEMO_TESTAPPS_H

#include "event_logger.h"

#ifdef __cplusplus
extern "C" {
#endif

int test_vec_add();

// Measuring ping by calling clEnqueueFillBuffer
typedef struct {
    cl_mem buf;
    cl_event event;
} ping_fillbuffer_context_t;

int ping_fillbuffer_init(ping_fillbuffer_context_t **ctx, cl_context context);

int ping_fillbuffer_run(ping_fillbuffer_context_t *ctx, cl_command_queue queue,
                        event_array_t *event_array);

int ping_fillbuffer_destroy(ping_fillbuffer_context_t **ctx);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_TESTAPPS_H
