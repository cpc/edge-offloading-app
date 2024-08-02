//
// Created by rabijl on 11/17/23.
//

#ifndef POCL_AISA_DEMO_DNN_STAGE_H
#define POCL_AISA_DEMO_DNN_STAGE_H

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"
#include "poclImageProcessorTypes.h"
#include "segment_4b_compression.hpp"

#include <Tracy.hpp>
#include <TracyOpenCL.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECTIONS 10
#define MASK_SZ1 160
#define MASK_SZ2 120

#define DET_COUNT (1 + MAX_DETECTIONS * 6)
#define SEG_COUNT (MAX_DETECTIONS * MASK_SZ1 * MASK_SZ2)
#define SEG_OUT_COUNT (MASK_SZ1 * MASK_SZ2 * 4)
// TODO: check if this size actually needed, or one value can be dropped
#define TOT_OUT_COUNT (DET_COUNT + SEG_COUNT)

/**
 *  Macro that sets the queue object to the right queue based on the codec config
 */
#define PICK_QUEUE(queue, ctx, config) \
    if (LOCAL_DEVICE == config->device_type) {\
        queue = ctx->local_queue;\
    } else if (REMOTE_DEVICE == config->device_type) {\
        queue = ctx->remote_queue;\
    } else {\
        LOGE("unknown device type to enqueue dnn to\n");\
        return -1;\
    }\

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
//    float iou; // store the iou
    int config_flags;
    segment_4b_context_t *segment_4b_ctx;
    cl_mem decompress_output_buf;

} dnn_context_t;

typedef struct {
    cl_event copy_event_det;      // Saving the detections from the compressed frame
    cl_event copy_event_seg_post; // Saving the segmentations from the compressed frame
    cl_mem det;                   // Temporary buffer to store detections from the compressed frame
    cl_mem seg_post;              // Temporary buffer to store segmentations from the compressed frame
    event_array_t *event_array;
} tmp_buf_ctx_t;

dnn_context_t *create_dnn_context();

int
init_dnn_context(dnn_context_t *dnn_context, int config_flags, cl_context ocl_context, int width,
                 int height, cl_device_id *dnn_device, cl_device_id *reconstruct_device,
                 int enable_eval);

cl_int
write_buffer_dnn(const dnn_context_t *ctx, device_type_enum device_type, uint8_t *inp_host_buf,
                 size_t buf_size, cl_mem cl_buf, const cl_event *wait_event,
                 event_array_t *event_array, cl_event *result_event);

cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const codec_config_t config,
            const pixel_format_enum input_format, const bool do_reconstruct, const cl_mem inp_buf,
            event_array_t *event_array, cl_event *out_event, tmp_buf_ctx_t *tmp_buf_ctx);

cl_int
enqueue_read_results_dnn(dnn_context_t *ctx, codec_config_t *config, int32_t *detection_array,
                         uint8_t *segmentation_array, event_array_t *event_array, int wait_size,
                         cl_event *wait_list);

cl_int destroy_dnn_context(dnn_context_t **context);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_DNN_STAGE_H
