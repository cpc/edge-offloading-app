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

/**
 * A struct that contains all configuration info
 * required for the pocl image processor to configure
 * the hevc codec.
 */
typedef struct {
    // todo: eventually use float for i frame interval
    int32_t i_frame_interval; // seconds between i frames
    int32_t framerate; // number of frames per second
    int32_t bitrate; // number of bits sent per second
} hevc_config_t;

/**
 * A struct that contains all objects required to use
 * the hevc codec.
 */
typedef struct {
    cl_mem comp_buf;
    cl_mem size_buf;

    cl_kernel enc_kernel;
    cl_kernel dec_kernel;

    uint32_t height;
    uint32_t width;
    cl_command_queue enc_queue; // needs to be freed manually
    cl_command_queue dec_queue; // needs to be freed manually

    size_t input_size;
    size_t output_size;

    uint32_t work_dim;
    size_t enc_global_size[3];
    size_t dec_global_size[3];
    int32_t output_format;

    /**
     * used to indicate that the codec has been configured
     */
    int32_t codec_configured;

    hevc_config_t config;
    /**
     * kernel used to configure the codec
     */
    cl_kernel config_kernel;

    uint64_t compressed_size;
} hevc_codec_context_t;

hevc_codec_context_t *create_hevc_context();

cl_int
init_c2_android_hevc_context(hevc_codec_context_t *codec_context, cl_context ocl_context,
                             cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize);

cl_int
init_hevc_context(hevc_codec_context_t *codec_context, cl_context ocl_context,
                  cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize);

cl_int
enqueue_hevc_compression(const hevc_codec_context_t *cxt, cl_event *wait_event, cl_mem inp_buf,
                         cl_mem out_buf, event_array_t *event_array, cl_event *result_event);

cl_int
configure_hevc_codec(hevc_codec_context_t *const codec_context, event_array_t *event_array,
                     cl_event *result_event);

cl_int
destroy_hevc_context(hevc_codec_context_t **context);

cl_int
hevc_configs_different(hevc_config_t A, hevc_config_t B);

void
set_hevc_config(hevc_codec_context_t *ctx, const hevc_config_t *const new_config);

cl_int
write_buffer_hevc(const hevc_codec_context_t *ctx, uint8_t *inp_host_buf, size_t buf_size,
                  cl_mem cl_buf, cl_event *wait_event, event_array_t *event_array,
                  cl_event *result_event);

size_t
get_compression_size_hevc(hevc_codec_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_HEVC_COMPRESSION_H
