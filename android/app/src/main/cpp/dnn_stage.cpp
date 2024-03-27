//
// Created by rabijl on 11/17/23.
//

#include "dnn_stage.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"
#include <Tracy.hpp>
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define DNN_VERBOSITY 0

const static int detection_count = 1 + MAX_DETECTIONS * 6;
const static int segmentation_count = MAX_DETECTIONS * MASK_W * MASK_H;
const static int seg_out_count = MASK_W * MASK_H * 4; // RGBA image
static int total_out_count = detection_count + segmentation_count;

dnn_context_t * create_dnn_context() {

    dnn_context_t *context = (dnn_context_t *) calloc(1, sizeof(dnn_context_t));
    context->iou = -5.0;

    return context;
}

#define dnn_kernel_name  "pocl.dnn.detection.u8"
#define postprocess_kernel_name "pocl.dnn.segmentation.postprocess.u8"
#define reconstruct_kernel_name "pocl.dnn.segmentation.reconstruct.u8"
#define eval_kernel_name "pocl.dnn.eval.iou.f32"
char *kernel_names = dnn_kernel_name ";"
                     postprocess_kernel_name ";"
                     reconstruct_kernel_name ";"
                     eval_kernel_name;

/**
 * init the dnn context
 * @param dnn_context
 * @param ocl_context
 * @param dnn_device device to run the dnn on
 * @param reconstruct_device device to do post processing on,
 * can be the same as dnn_device
 * @param enable_eval whether or not to init the eval kernels as well
 * @return OpenCL status message
 */
int
init_dnn_context(dnn_context_t *dnn_context, cl_context ocl_context, cl_device_id *dnn_device,
                 cl_device_id *reconstruct_device, int enable_eval) {

    int status;

    // if both devices are the same, don't pass the same device twice to
    // the program
    int num_devs = 2;
    if (*dnn_device == *reconstruct_device) {
        num_devs = 1;
    }

    cl_device_id devices[] = {*dnn_device, *reconstruct_device};
    cl_program program = clCreateProgramWithBuiltInKernels(ocl_context, num_devs, devices,
                                                           kernel_names, &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    status = clBuildProgram(program, num_devs, devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building of program failed");

    dnn_context->dnn_kernel = clCreateKernel(program, dnn_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    dnn_context->postprocess_kernel = clCreateKernel(program, postprocess_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    dnn_context->reconstruct_kernel = clCreateKernel(program, reconstruct_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating reconstruct kernel failed");

    // TODO: don't hardcode image dimensions
    dnn_context->out_mask_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                               640 * 480 * MAX_DETECTIONS *
                                               sizeof(cl_char) / 4,
                                               NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    dnn_context->postprocess_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                  MASK_W * MASK_H * sizeof(cl_uchar),
                                                  NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

    // set the kernel parameters
    status = clSetKernelArg(dnn_context->dnn_kernel, 1, sizeof(cl_uint), &(dnn_context->width));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 2, sizeof(cl_uint), &(dnn_context->height));
    status |=
            clSetKernelArg(dnn_context->dnn_kernel, 3, sizeof(cl_int),
                           &(dnn_context->rotate_cw_degrees));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 6, sizeof(cl_mem),
                             &(dnn_context->out_mask_buf));
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    status = clSetKernelArg(dnn_context->postprocess_kernel, 1, sizeof(cl_mem),
                            &(dnn_context->out_mask_buf));
    status |= clSetKernelArg(dnn_context->postprocess_kernel, 2, sizeof(cl_mem),
                             &(dnn_context->postprocess_buf));
    CHECK_AND_RETURN(status, "could not assign postprocess kernel args");


    status = clSetKernelArg(dnn_context->reconstruct_kernel, 0, sizeof(cl_mem),
                            &(dnn_context->postprocess_buf));
    CHECK_AND_RETURN(status, "could not assign reconstruct_kernel args");

    if (enable_eval) {
        dnn_context->eval_kernel = clCreateKernel(program, eval_kernel_name, &status);
        CHECK_AND_RETURN(status, "could not create eval kernel");
        dnn_context->eval_buf = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, sizeof(cl_float),
                                               NULL,
                                               &status);
        CHECK_AND_RETURN(status, "could create eval_buf");
        status |= clSetKernelArg(dnn_context->eval_kernel, 5, sizeof(cl_mem),
                                 &(dnn_context->eval_buf));
    }

    dnn_context->work_dim = 1;
    dnn_context->global_size[0] = 1;
    dnn_context->local_size[0] = 1;


//    dnn_context->tracy_dnn_queue = TracyCLContext(ocl_context, &dnn_device);
//    dnn_context->tracy_reconstruct_queue = TracyCLContext(ocl_context, &reconstruct_device);

    clReleaseProgram(program);

    return 0;
}

/**
 * function to write the raw image to the right image. Useful for when using
 * no compression.
 * @param ctx the dnn context
 * @param device_type device to enqueue on, either LOCAL_ or REMOTE_DEVICE
 * @param inp_host_buf buffer to read from
 * @param buf_size size of the buffer
 * @param cl_buf the buffer write to
 * @param wait_event event to wait on before writing
 * @param event_array array to append the read event to
 * @param result_event event that can be waited on
 * @return CL_SUCCESS or an error otherwise
 */
cl_int
write_buffer_dnn(const dnn_context_t *ctx, device_type_enum device_type, uint8_t *inp_host_buf, size_t buf_size,
                 cl_mem cl_buf, const cl_event *wait_event, event_array_t *event_array,
                 cl_event *result_event) {
    ZoneScoped;

    cl_int status;
    cl_event write_img_event;

    // figure out on which queue to run
    cl_command_queue write_queue;
    if (LOCAL_DEVICE == device_type) {
        write_queue = ctx->local_queue;
    } else if (REMOTE_DEVICE == device_type) {
        write_queue = ctx->remote_queue;
    } else {
        LOGE("unknown device type to enqueue to\n");
        return CL_INVALID_VALUE;
    }

    int wait_size = 0;
    if (NULL != wait_event) {
        wait_size = 1;
    }

    status = clEnqueueWriteBuffer(write_queue, cl_buf, CL_FALSE, 0,
                                  buf_size, inp_host_buf, wait_size, wait_event, &write_img_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));
    *result_event = write_img_event;
    return status;
}

/**
 * Function to enqueue the opencl commands related to object detection.
 * @param ctx with relevant info
 * @param wait_event event to wait on before starting these commands
 * @param config config with relevant info
 * @param input_format the format that the inp_buf is in
 * @param inp_buf  uncompressed image, either yuv or rgb
 * @param detection_array array of bounding boxes of detected objects
 * @param segmentation_array bitmap of segments of detected objects
 * @param event_array where to append the command events to
 * @param out_event event that can be waited on
 * @return CL_SUCCESS if everything went well, otherwise a cl error number
 */
cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const codec_config_t config,
            const pixel_format_enum input_format, const cl_mem inp_buf,
            cl_mem detection_array, cl_mem segmentation_array,
            event_array_t *event_array, cl_event *out_event) {

    // TODO: either enable tracy again or remove it
    ZoneScoped;
    cl_int status;

    // cast enum to int
    cl_int inp_format = (cl_int) input_format;

    status = clSetKernelArg(ctx->dnn_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |=
            clSetKernelArg(ctx->dnn_kernel, 3, sizeof(cl_int), &(config.rotation));
    status |= clSetKernelArg(ctx->dnn_kernel, 4, sizeof(cl_int), &inp_format);
    status |= clSetKernelArg(ctx->dnn_kernel, 5, sizeof(cl_mem), &detection_array);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(ctx->postprocess_kernel, 0, sizeof(cl_mem), &detection_array);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

    status = clSetKernelArg(ctx->reconstruct_kernel, 1, sizeof(cl_mem),
                            &segmentation_array);
    CHECK_AND_RETURN(status, "could not assign buffers to reconstruct kernel");

    // figure out on which queue to run the dnn
    cl_command_queue dnn_queue;
    if (LOCAL_DEVICE == config.device_type) {
        dnn_queue = ctx->local_queue;
    } else if (REMOTE_DEVICE == config.device_type) {
        dnn_queue = ctx->remote_queue;
    } else {
        LOGE("unknown device type to enqueue dnn to\n");
        return -1;
    }

    cl_event run_dnn_event;

    {
        TracyCLZone(ctx->remote_tracy_ctx, "DNN");
        status = clEnqueueNDRangeKernel(dnn_queue, ctx->dnn_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1,
                                        wait_event, &run_dnn_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
        append_to_event_array(event_array, run_dnn_event, VAR_NAME(dnn_event));
        TracyCLZoneSetEvent(run_dnn_event);
    }

    if (config.do_segment) {
        cl_event run_postprocess_event, mig_seg_event, run_reconstruct_event, read_segment_event;

        {
            // postprocess
            TracyCLZone(ctx->remote_tracy_ctx, "postprocess");
            status = clEnqueueNDRangeKernel(dnn_queue, ctx->postprocess_kernel, ctx->work_dim,
                                            NULL,
                                            (ctx->global_size), (ctx->local_size), 1,
                                            &run_dnn_event, &run_postprocess_event);
            CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");
            append_to_event_array(event_array, run_postprocess_event, VAR_NAME(postprocess_event));
            TracyCLZoneSetEvent(run_postprocess_event);
        }

        {
            // move postprocessed segmentation data to host device
            TracyCLZone(ctx->local_tracy_ctx, "migrate DNN");
            status = clEnqueueMigrateMemObjects(ctx->local_queue, 1,
                                                &(ctx->postprocess_buf), 0, 1,
                                                &run_postprocess_event,
                                                &mig_seg_event);
            CHECK_AND_RETURN(status, "failed to enqueue migration of postprocess buffer");
            append_to_event_array(event_array, mig_seg_event, VAR_NAME(mig_seg_event));
            TracyCLZoneSetEvent(mig_seg_event);
        }

        {
            // reconstruct postprocessed data to RGBA segmentation mask
            TracyCLZone(ctx->local_tracy_ctx, "reconstruct");
            status = clEnqueueNDRangeKernel(ctx->local_queue, ctx->reconstruct_kernel, ctx->work_dim,
                                            NULL,
                                            ctx->global_size, ctx->local_size, 1,
                                            &mig_seg_event, &run_reconstruct_event);
            CHECK_AND_RETURN(status, "failed to enqueue ND range reconstruct kernel");
            append_to_event_array(event_array, run_reconstruct_event, VAR_NAME(reconstruct_event));
            TracyCLZoneSetEvent(run_reconstruct_event);
        }

        *out_event = run_reconstruct_event;
    } else {
        *out_event = run_dnn_event;
    }

    // TODO: migrate result buffers to local device

    return 0;
}

/**
 * enqueue the eval kernel which calculates the iou and stores it in the iou var of the ctx
 * @param ctx
 * @param base_detect_buf
 * @param base_seg_buf
 * @param eval_detect_buf
 * @param eval_seg_buf
 * @param config
 * @param wait_list
 * @param wait_list_size
 * @param event_array used to keep track of the events
 * @param result_event can be waited on
 * @return a OpenCL status message
 */
cl_int
enqueue_eval_dnn(dnn_context_t *ctx, cl_mem base_detect_buf, cl_mem base_seg_buf,
                 cl_mem eval_detect_buf, cl_mem eval_seg_buf, codec_config_t *config,
                 cl_event *wait_list, int wait_list_size,
                 event_array_t *event_array, cl_event *result_event) {

    cl_int status;
    status = clSetKernelArg(ctx->eval_kernel, 0, sizeof(cl_mem), &(base_detect_buf));
    status |= clSetKernelArg(ctx->eval_kernel, 1, sizeof(cl_mem), &(base_seg_buf));
    status |= clSetKernelArg(ctx->eval_kernel, 2, sizeof(cl_mem), &(eval_detect_buf));
    status |= clSetKernelArg(ctx->eval_kernel, 3, sizeof(cl_mem), &(eval_seg_buf));
    status |= clSetKernelArg(ctx->eval_kernel, 4, sizeof(cl_int), &(config->do_segment));
    CHECK_AND_RETURN(status, "could not assign eval kernel params");

    // figure out on which queue to run the dnn
    cl_command_queue queue;
    if (LOCAL_DEVICE == config->device_type) {
        queue = ctx->local_queue;
    } else if (REMOTE_DEVICE == config->device_type) {
        queue = ctx->remote_queue;
    } else {
        LOGE("unknown device type to enqueue dnn to\n");
        return -1;
    }

    cl_event eval_iou_event, eval_read_iou_event;

    status = clEnqueueNDRangeKernel(queue, ctx->eval_kernel, ctx->work_dim, NULL,
                                    ctx->global_size, ctx->local_size,
                                    wait_list_size, wait_list, &eval_iou_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range eval kernel");
    append_to_event_array(event_array, eval_iou_event, VAR_NAME(eval_iou_event));

    status = clEnqueueReadBuffer(queue, ctx->eval_buf, CL_FALSE, 0,
                                 1 * sizeof(cl_float), &(ctx->iou), 1, &eval_iou_event,
                                 &eval_read_iou_event);
    CHECK_AND_RETURN(status, "failed to read eval iou result buffer");
    append_to_event_array(event_array, eval_read_iou_event, VAR_NAME(eval_read_iou_event));

    *result_event = eval_read_iou_event;

    return CL_SUCCESS;
}

/**
 * deallocate the dnn context and address to NULL
 * @param context
 * @return CL_SUCCESS or an error otherwise
 */
cl_int
destroy_dnn_context(dnn_context_t **context) {

    dnn_context_t *c = *context;

    if(NULL == c) {
        return CL_SUCCESS;
    }

    COND_REL_MEM(c->out_mask_buf)

    COND_REL_MEM(c->postprocess_buf)

    COND_REL_KERNEL(c->dnn_kernel)

    COND_REL_KERNEL(c->postprocess_kernel)

    COND_REL_KERNEL(c->reconstruct_kernel)

    COND_REL_KERNEL(c->eval_kernel)

    COND_REL_MEM(c->eval_buf)

    free(c);
    *context = NULL;
    return CL_SUCCESS;
}

#ifdef __cplusplus
}
#endif