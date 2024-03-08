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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_mem out_mask_buf;
    cl_mem postprocess_buf;
    cl_kernel dnn_kernel;
    cl_kernel postprocess_kernel;
    cl_kernel reconstruct_kernel;

    size_t global_size[3];
    size_t local_size[3];
    uint32_t work_dim;
    cl_command_queue remote_queue;
    cl_command_queue local_queue;
    int32_t rotate_cw_degrees;
    uint32_t height;
    uint32_t width;

#ifdef TRACY_ENABLE
    tracy_cl_ctx tracy_dnn_queue;
    tracy_cl_ctx tracy_reconstruct_queue;
#endif

} dnn_context_t;

typedef struct {
    cl_command_queue  eval_queue;
    cl_kernel eval_kernel;
    cl_mem eval_img_buf;
    cl_mem eval_out_buf;
    cl_mem eval_out_mask_buf;
    cl_mem eval_postprocess_buf;
    cl_mem eval_iou_buf;
}eval_context_t;

dnn_context_t *create_dnn_context();

int
init_dnn_context(dnn_context_t *dnn_context, eval_context_t *eval_context, cl_context ocl_context,
                 cl_device_id *dnn_device,
                 cl_device_id *reconstruct_device);

cl_int
write_buffer_dnn(const dnn_context_t *ctx, devic_type_enum device_type, uint8_t *inp_host_buf, size_t buf_size,
                 cl_mem cl_buf, const cl_event *wait_event, event_array_t *event_array,
                 cl_event *result_event);

cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const codec_config_t config,
            const pixel_format_enum input_format, cl_mem inp_buf,
            cl_mem detection_array, cl_mem segmentation_array,
            event_array_t *event_array, cl_event *out_event);

cl_int
destroy_dnn_context(dnn_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_DNN_STAGE_H
