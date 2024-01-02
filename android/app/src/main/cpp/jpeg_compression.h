//
// Created by rabijl on 11/20/23.
//

#ifndef POCL_AISA_DEMO_JPEG_COMPRESSION_H
#define POCL_AISA_DEMO_JPEG_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"

typedef struct {
    int32_t quality;
} jpeg_config_t;

typedef struct {
    cl_mem inp_buf;
    cl_mem comp_buf;
    cl_mem size_buf;
    cl_mem out_buf;
    cl_kernel enc_kernel;
    cl_kernel dec_kernel;

    uint32_t height;
    uint32_t width;
    cl_command_queue enc_queue; // needs to be freed manually
    cl_command_queue dec_queue; // needs to be freed manually
    uint8_t *host_img_buf; // needs to be freed manually
    size_t img_buf_size;
    uint8_t *host_postprocess_buf; // needs to be freed manually
    int32_t quality; // currently not used
    uint32_t work_dim;
    size_t enc_global_size[3];
    size_t dec_global_size[3];
    int32_t output_format;

} jpeg_codec_context_t;

jpeg_codec_context_t *create_jpeg_context();

cl_int
init_jpeg_context(jpeg_codec_context_t *codec_context, cl_context ocl_context,
                  cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize);

cl_int
enqueue_jpeg_compression(const jpeg_codec_context_t *cxt, event_array_t *event_array,
                         cl_event *result_event);

cl_int
destory_jpeg_context(jpeg_codec_context_t **context);


#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_JPEG_COMPRESSION_H
