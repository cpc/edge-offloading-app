//
// Created by rabijl on 3.7.2024.
//

#ifndef POCL_AISA_DEMO_SEGMENT_4B_COMPRESSION_HPP
#define POCL_AISA_DEMO_SEGMENT_4B_COMPRESSION_HPP

#include "rename_opencl.h"
#include <CL/cl.h>
#include "event_logger.h"

#include <Tracy.hpp>
#include <TracyOpenCL.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_command_queue const encode_queue;
    cl_command_queue const decode_queue;
    cl_mem const compress_buf;
    cl_kernel const encode_kernel;
    cl_kernel const decode_kernel;
    uint32_t const width;
    uint32_t const height;
    uint32_t const no_class_id;
    size_t const global_size[3];
    size_t const local_size[3];
    uint32_t const work_dim;
    // todo: workout tracy
    TracyCLCtx *tracy_ctxs;
} segment_4b_context_t;

segment_4b_context_t *
init_segment_4b(cl_context cl_ctx, cl_command_queue encode_queue, cl_command_queue decode_queue,
                cl_device_id devs[2], uint32_t width, uint32_t height,
                size_t source_size, char const source[],
                cl_int *ret_status);

cl_int
encode_segment_4b(segment_4b_context_t *ctx, const cl_event *wait_event,
                  cl_mem input, cl_mem detections, cl_mem output,
                  event_array_t *event_array, cl_event *event);

cl_int
destroy_segment_4b(segment_4b_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_SEGMENT_4B_COMPRESSION_HPP
