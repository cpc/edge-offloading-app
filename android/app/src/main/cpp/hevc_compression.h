//
// Created by rabijl on 11/20/23.
//

#ifndef POCL_AISA_DEMO_HEVC_COMPRESSION_H
#define POCL_AISA_DEMO_HEVC_COMPRESSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"

typedef struct {
    int32_t quality;
    int32_t bitrate;
} hevc_config_t;

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

    /**
     * used to indicate that the codec has been configured
     */
    int32_t codec_configured;
    /**
     * the number of seconds between I-frames
     */
    int32_t i_frame_interval;
    /**
     * the number of bits sent per second
     */
    int32_t bitrate;
    /**
     * the target number of frames per second
     */
    int32_t framerate;
    /**
     * kernel used to configure the codec
     */
    cl_kernel config_kernel;


} hevc_codec_context_t;

hevc_codec_context_t *create_hevc_context();

cl_int
init_hevc_context(hevc_codec_context_t *codec_context, cl_context ocl_context,
                  cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize);

cl_int
enqueue_hevc_compression(const hevc_codec_context_t *cxt, event_array_t *event_array,
                         cl_event *wait_event, cl_event *result_event);

cl_int
configure_hevc_codec(hevc_codec_context_t *const codec_context, event_array_t *event_array,
                     cl_event *result_event);

cl_int
destory_hevc_context(hevc_codec_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_HEVC_COMPRESSION_H
