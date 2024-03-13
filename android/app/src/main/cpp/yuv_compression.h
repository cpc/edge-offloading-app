//
// Created by rabijl on 11/16/23.
//

#ifndef POCL_AISA_DEMO_YUV_COMPRESSION_H
#define POCL_AISA_DEMO_YUV_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"

typedef struct {
    cl_mem out_enc_y_buf;
    cl_mem out_enc_uv_buf;
    cl_kernel enc_y_kernel;
    cl_kernel enc_uv_kernel;
    cl_kernel dec_y_kernel;
    cl_kernel dec_uv_kernel;
    uint32_t height;
    uint32_t width;
    cl_command_queue enc_queue; // needs to be freed manually
    cl_command_queue dec_queue; // needs to be freed manually
    int32_t quality; // currently not used
    uint32_t work_dim;
    size_t y_global_size[3];
    size_t uv_global_size[3];
    int32_t output_format;
} yuv_codec_context_t;

yuv_codec_context_t *create_yuv_context();

int
init_yuv_context(yuv_codec_context_t *codec_context, cl_context cl_context, cl_device_id enc_device,
                 cl_device_id dec_device, const char *source, const size_t
                 src_size);

cl_int
write_buffer_yuv(const yuv_codec_context_t *ctx, const uint8_t *inp_host_buf, size_t buf_size,
                 cl_mem cl_buf, const cl_event *wait_event,
                 event_array_t *event_array, cl_event *result_event);

cl_int
enqueue_yuv_compression(const yuv_codec_context_t *cxt, cl_event wait_event, cl_mem inp_buf,
                        cl_mem out_buf, event_array_t *event_array,
                        cl_event *result_event);

cl_int
destroy_yuv_context(yuv_codec_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_YUV_COMPRESSION_H
