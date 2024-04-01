//
// Created by rabijl on 11/17/23.
//

#ifndef POCL_AISA_DEMO_DNN_STAGE_H
#define POCL_AISA_DEMO_DNN_STAGE_H

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"
#include "poclImageProcessor.h"

#include <Tracy.hpp>
#include <TracyOpenCL.hpp>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_mem out_mask_buf; // input for postprocess kernel
    cl_mem postprocess_buf; // input for postprocess kernel

    // the ctx knows which queues to use for reading these results back
    // that is why it is in charge of maintaining these
    cl_mem detect_buf; // holds the detected objects and some other things
    cl_mem segmentation_buf; // reconstructed result to be returned

    cl_kernel dnn_kernel;
    cl_kernel postprocess_kernel;
    cl_kernel reconstruct_kernel;

    size_t global_size[3];
    size_t local_size[3];
    uint32_t work_dim;
    cl_command_queue remote_queue; // used to run the dnn and postprocess
    cl_command_queue local_queue; // used for reading and reconstruction
    int32_t rotate_cw_degrees;
    uint32_t height;
    uint32_t width;

    TracyCLCtx remote_tracy_ctx;
    TracyCLCtx local_tracy_ctx;
    // TODO: create char arrays to hold unique tracy names that can be used to mark zones

    cl_kernel eval_kernel; // kernel that that calculates intersection over union
    cl_mem eval_buf; // used to read the iou
    float iou; // store the iou

} dnn_context_t;

dnn_context_t *create_dnn_context();

int
init_dnn_context(dnn_context_t *dnn_context, cl_context ocl_context, int width, int height,
                 cl_device_id *dnn_device,
                 cl_device_id *reconstruct_device, int enable_eval);

cl_int
write_buffer_dnn(const dnn_context_t *ctx, device_type_enum device_type, uint8_t *inp_host_buf,
                 size_t buf_size,
                 cl_mem cl_buf, const cl_event *wait_event, event_array_t *event_array,
                 cl_event *result_event);

cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const codec_config_t config,
            const pixel_format_enum input_format, cl_mem inp_buf,
            event_array_t *event_array, cl_event *out_event);

cl_int
enqueue_eval_dnn(dnn_context_t *eval_ctx, dnn_context_t *ctx, codec_config_t *config,
                 cl_event *wait_list, int wait_list_size,
                 event_array_t *event_array, cl_event *result_event);

cl_int
enqueue_read_results_dnn(dnn_context_t *ctx, codec_config_t *config,
                         int32_t *detection_array, uint8_t *segmentation_array,
                         event_array_t *event_array, int wait_size,
                         cl_event *wait_list);

cl_int
destroy_dnn_context(dnn_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_DNN_STAGE_H
