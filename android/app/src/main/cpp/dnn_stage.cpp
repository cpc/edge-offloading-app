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

#define DNN_VERBOSITY 0

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define DET_COUNT (1 + MAX_DETECTIONS * 6)
#define SEG_COUNT (MAX_DETECTIONS * MASK_W * MASK_H)
#define SEG_OUT_COUNT (MASK_W * MASK_H * 4)
// TODO: check if this size actually needed, or one value can be dropped
#define TOT_OUT_COUNT (DET_COUNT + SEG_COUNT)

dnn_context_t *create_dnn_context() {

    dnn_context_t *context = (dnn_context_t *) calloc(1, sizeof(dnn_context_t));
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
int init_dnn_context(dnn_context_t *dnn_context, cl_context ocl_context, int width, int height,
                     cl_device_id *dnn_device, cl_device_id *reconstruct_device, int enable_eval) {

    int status;

    dnn_context->width = width;
    dnn_context->height = height;

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

    dnn_context->out_mask_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                               width * height * MAX_DETECTIONS * sizeof(cl_char) /
                                               4, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    dnn_context->postprocess_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                  MASK_W * MASK_H * sizeof(cl_uchar), NULL,
                                                  &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

    dnn_context->detect_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                             TOT_OUT_COUNT * sizeof(cl_int), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the detection + other stuff buffer");

    dnn_context->segmentation_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                   SEG_OUT_COUNT * sizeof(cl_uchar), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create segmentation output buffer")

    // set the kernel parameters
    status = clSetKernelArg(dnn_context->dnn_kernel, 1, sizeof(cl_uint), &(dnn_context->width));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 2, sizeof(cl_uint), &(dnn_context->height));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 3, sizeof(cl_int),
                             &(dnn_context->rotate_cw_degrees));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 5, sizeof(cl_mem),
                             &(dnn_context->detect_buf));

    status |= clSetKernelArg(dnn_context->dnn_kernel, 6, sizeof(cl_mem),
                             &(dnn_context->out_mask_buf));
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    status = clSetKernelArg(dnn_context->postprocess_kernel, 0, sizeof(cl_mem),
                            &(dnn_context->detect_buf));
    status |= clSetKernelArg(dnn_context->postprocess_kernel, 1, sizeof(cl_mem),
                             &(dnn_context->out_mask_buf));
    status |= clSetKernelArg(dnn_context->postprocess_kernel, 2, sizeof(cl_mem),
                             &(dnn_context->postprocess_buf));
    CHECK_AND_RETURN(status, "could not assign postprocess kernel args");


    status = clSetKernelArg(dnn_context->reconstruct_kernel, 0, sizeof(cl_mem),
                            &(dnn_context->postprocess_buf));
    status = clSetKernelArg(dnn_context->reconstruct_kernel, 1, sizeof(cl_mem),
                            &(dnn_context->segmentation_buf));
    CHECK_AND_RETURN(status, "could not assign reconstruct_kernel args");

    if (enable_eval) {
        dnn_context->eval_kernel = clCreateKernel(program, eval_kernel_name, &status);
        CHECK_AND_RETURN(status, "could not create eval kernel");
        dnn_context->eval_buf = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, sizeof(cl_float),
                                               NULL, &status);
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
write_buffer_dnn(const dnn_context_t *ctx, device_type_enum device_type, uint8_t *inp_host_buf,
                 size_t buf_size, cl_mem cl_buf, const cl_event *wait_event,
                 event_array_t *event_array, cl_event *result_event) {
    ZoneScoped;

    cl_int status;
    cl_event write_img_event, undef_img_mig_event;

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

    status = clEnqueueMigrateMemObjects(write_queue, 1, &cl_buf,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, wait_size,
                                        wait_event, &undef_img_mig_event);
    CHECK_AND_RETURN(status, "could migrate enc buffer back before writing");
    append_to_event_array(event_array, undef_img_mig_event, VAR_NAME(undef_img_mig_event));

    status = clEnqueueWriteBuffer(write_queue, cl_buf, CL_FALSE, 0, buf_size, inp_host_buf, 1,
                                  &undef_img_mig_event, &write_img_event);
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
 * @param do_reconstruct reconstruct full segmentation mask (not necessary for the eval frame, only used for segmentation)
 * @param inp_buf  uncompressed image, either yuv or rgb
 * @param detection_array array of bounding boxes of detected objects
 * @param segmentation_array bitmap of segments of detected objects
 * @param event_array where to append the command events to
 * @param out_event event that can be waited on
 * @param tmp_buf_ctx optional context for saving results to buffers for eval; Ignored if NULL
 * @return CL_SUCCESS if everything went well, otherwise a cl error number
 */
cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const codec_config_t config,
            const pixel_format_enum input_format, const bool do_reconstruct, const cl_mem inp_buf,
            event_array_t *event_array, cl_event *out_event, tmp_buf_ctx_t *tmp_buf_ctx) {
    ZoneScoped;
    cl_int status;

    // cast enum to int
    cl_int inp_format = (cl_int) input_format;

    status = clSetKernelArg(ctx->dnn_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |= clSetKernelArg(ctx->dnn_kernel, 3, sizeof(cl_int), &(config.rotation));
    status |= clSetKernelArg(ctx->dnn_kernel, 4, sizeof(cl_int), &inp_format);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    // figure out on which queue to run the dnn
    cl_command_queue dnn_queue;
    TracyCLCtx dnn_tracy_ctx;
    if (LOCAL_DEVICE == config.device_type) {
        dnn_queue = ctx->local_queue;
        dnn_tracy_ctx = ctx->local_tracy_ctx;
    } else if (REMOTE_DEVICE == config.device_type) {
        dnn_queue = ctx->remote_queue;
        dnn_tracy_ctx = ctx->remote_tracy_ctx;
    } else {
        LOGE("unknown device type to enqueue dnn to\n");
        return -1;
    }

    cl_event dnn_event;

    {
        TracyCLZone(dnn_tracy_ctx, "DNN");
        status = clEnqueueNDRangeKernel(dnn_queue, ctx->dnn_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1, wait_event,
                                        &dnn_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
        append_to_event_array(event_array, dnn_event, VAR_NAME(dnn_event));
        TracyCLZoneSetEvent(dnn_event);
    }

    if (tmp_buf_ctx != NULL) {
        // Save detections to a temporary buffer for the quality evaluation pipeline
        size_t sz = DETECTION_SIZE * sizeof(cl_int);

        cl_event copy_det_event;
        status = clEnqueueCopyBuffer(dnn_queue, ctx->detect_buf, tmp_buf_ctx->det, 0, 0, sz, 1,
                                     &dnn_event, &copy_det_event);
        CHECK_AND_RETURN(status, "failed to copy result buffer");
        append_to_event_array(tmp_buf_ctx->event_array, copy_det_event, VAR_NAME(copy_det_event));
        tmp_buf_ctx->copy_event_det = copy_det_event;
    }

    if (config.do_segment) {
        cl_event postprocess_event, mig_seg_event, reconstruct_event;

        {
            // postprocess
            TracyCLZone(dnn_tracy_ctx, "postprocess");
            status = clEnqueueNDRangeKernel(dnn_queue, ctx->postprocess_kernel, ctx->work_dim, NULL,
                                            (ctx->global_size), (ctx->local_size), 1,
                                            &dnn_event, &postprocess_event);
            CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");
            append_to_event_array(event_array, postprocess_event, VAR_NAME(postprocess_event));
            TracyCLZoneSetEvent(postprocess_event);
        }

        if (tmp_buf_ctx != NULL) {
            // Save postprocessed segmentation to a temporary buffer for the quality evaluation pipeline
            size_t sz = MASK_W * MASK_H * sizeof(cl_uchar);

            cl_event copy_event_seg;
            status = clEnqueueCopyBuffer(dnn_queue, ctx->postprocess_buf, tmp_buf_ctx->seg_post, 0,
                                         0, sz, 1, &postprocess_event, &copy_event_seg);
            CHECK_AND_RETURN(status, "failed to copy result buffer");
            append_to_event_array(tmp_buf_ctx->event_array, copy_event_seg,
                                  VAR_NAME(copy_seg_event));
            tmp_buf_ctx->copy_event_seg_post = copy_event_seg;
        }

        if (do_reconstruct) {
            {
                // move postprocessed segmentation data to host device
                TracyCLZone(ctx->local_tracy_ctx, "migrate DNN");
                status = clEnqueueMigrateMemObjects(ctx->local_queue, 1, &(ctx->postprocess_buf), 0,
                                                    1, &postprocess_event, &mig_seg_event);
                CHECK_AND_RETURN(status, "failed to enqueue migration of postprocess buffer");
                append_to_event_array(event_array, mig_seg_event, VAR_NAME(mig_seg_event));
                TracyCLZoneSetEvent(mig_seg_event);
            }

            {
                // reconstruct postprocessed data to RGBA segmentation mask
                TracyCLZone(ctx->local_tracy_ctx, "reconstruct");
                status = clEnqueueNDRangeKernel(ctx->local_queue, ctx->reconstruct_kernel,
                                                ctx->work_dim, NULL, ctx->global_size,
                                                ctx->local_size, 1, &mig_seg_event,
                                                &reconstruct_event);
                CHECK_AND_RETURN(status, "failed to enqueue ND range reconstruct kernel");
                append_to_event_array(event_array, reconstruct_event,
                                      VAR_NAME(reconstruct_event));
                TracyCLZoneSetEvent(reconstruct_event);
            }

            *out_event = reconstruct_event;
        } else {
            *out_event = postprocess_event;
        }
    } else {
        *out_event = dnn_event;
    }

    // TODO: migrate result buffers to local device

    return 0;
}

/**
 * read the results of the dnn stage and put them in the arrays
 * @param ctx
 * @param config contains relevant things like device type and do_segmentation
 * @param detection_array output: contains bounding boxes
 * @param segmentation_array output: contains a segmentation mask
 * @param event_array for bookkeeping
 * @param wait_size
 * @param wait_list list of event to wait on before starting
 * @return OpenCL status
 */
cl_int
enqueue_read_results_dnn(dnn_context_t *ctx, codec_config_t *config, int32_t *detection_array,
                         uint8_t *segmentation_array, event_array_t *event_array, int wait_size,
                         cl_event *wait_list) {
    ZoneScoped;
    // TODO: create special remote reading queue to not block on dnn

    // figure out on which queue to run the dnn
    cl_command_queue queue;
    PICK_QUEUE(queue, ctx, config)

    int status;
    cl_event read_detect_event, read_segment_event = NULL;
    int wait_event_size = 1;

    status = clEnqueueReadBuffer(queue, ctx->detect_buf, CL_FALSE, 0, DET_COUNT * sizeof(cl_int),
                                 detection_array, wait_size, wait_list, &read_detect_event);
    CHECK_AND_RETURN(status, "could not read detection array");

    append_to_event_array(event_array, read_detect_event, VAR_NAME(read_detect_event));

    if (config->do_segment) {
        status = clEnqueueReadBuffer(ctx->local_queue, ctx->segmentation_buf, CL_FALSE, 0,
                                     SEG_OUT_COUNT * sizeof(cl_uchar), segmentation_array,
                                     wait_size, wait_list, &read_segment_event);
        CHECK_AND_RETURN(status, "could not read segmentation array");
        append_to_event_array(event_array, read_segment_event, VAR_NAME(read_segment_event));
        // increment the size of the wait event list
        wait_event_size = 2;
    }

    {
        ZoneScopedN("wait");
        cl_event wait_events[] = {read_detect_event, read_segment_event};
        // after this wait, the detection and segmentation arrays are valid
        status = clWaitForEvents(wait_event_size, wait_events);
        CHECK_AND_RETURN(status, "could not wait on final event");
    }

    return CL_SUCCESS;
}

/**
 * deallocate the dnn context and address to NULL
 * @param context
 * @return CL_SUCCESS or an error otherwise
 */
cl_int destroy_dnn_context(dnn_context_t **context) {

    dnn_context_t *c = *context;

    if (NULL == c) {
        return CL_SUCCESS;
    }

    if (c->local_tracy_ctx == c->remote_tracy_ctx) {
        TracyCLDestroy(c->local_tracy_ctx);
    } else {
        TracyCLDestroy(c->local_tracy_ctx);
        TracyCLDestroy(c->remote_tracy_ctx);
    }

    COND_REL_MEM(c->out_mask_buf)

    COND_REL_MEM(c->postprocess_buf)

    COND_REL_KERNEL(c->dnn_kernel)

    COND_REL_KERNEL(c->postprocess_kernel)

    COND_REL_KERNEL(c->reconstruct_kernel)

    COND_REL_KERNEL(c->eval_kernel)

    COND_REL_MEM(c->eval_buf)

    COND_REL_MEM(c->detect_buf)

    COND_REL_MEM(c->segmentation_buf)

    free(c);
    *context = NULL;
    return CL_SUCCESS;
}

#ifdef __cplusplus
}
#endif