//
// Created by rabijl on 11/17/23.
//

#include "dnn_stage.hpp"
#include "sharedUtils.h"
#include <Tracy.hpp>
#include <cassert>

#ifdef __cplusplus
extern "C" {
#endif

#define DNN_VERBOSITY 0

dnn_context_t *create_dnn_context() {

    dnn_context_t *context = (dnn_context_t *) calloc(1, sizeof(dnn_context_t));
    return context;
}

#define dnn_kernel_name  "pocl.dnn.detection.u8"
#define postprocess_kernel_name "pocl.dnn.segmentation.postprocess.u8"
#define reconstruct_kernel_name "pocl.dnn.segmentation.reconstruct.u8"
#define eval_kernel_name "pocl.dnn.eval.iou.f32"
char const *kernel_names = dnn_kernel_name ";"
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
init_dnn_context(dnn_context_t *dnn_context, int config_flags, cl_context ocl_context, int width,
                 int height,
                 cl_device_id *dnn_device, cl_device_id *reconstruct_device, int enable_eval) {

    // make sure that the segment_4b_ctx has been dependency injected when calling with segment_4b
    assert((config_flags & SEGMENT_4B) ? dnn_context->segment_4b_ctx != NULL : 1);

    int status;

    dnn_context->width = width;
    dnn_context->height = height;
    dnn_context->config_flags = config_flags;

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
                                                  MASK_SZ1 * MASK_SZ2 * sizeof(cl_uchar), NULL,
                                                  &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

    dnn_context->detect_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                             DET_COUNT * sizeof(cl_int), NULL, &status);
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

    if (dnn_context->config_flags & SEGMENT_4B) {
        dnn_context->decompress_output_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                            MASK_SZ1 * MASK_SZ2 * sizeof(cl_uchar),
                                                            NULL,
                                                            &status);
    }

    dnn_context->work_dim = 1;
    dnn_context->global_size[0] = 1;
    dnn_context->local_size[0] = 1;

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

    cl_event dnn_event, detect_mig_event;

    cl_mem mem_objs[] = {ctx->detect_buf, ctx->postprocess_buf};
    status = clEnqueueMigrateMemObjects(dnn_queue, 2, mem_objs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                        1, wait_event, &detect_mig_event);
    CHECK_AND_RETURN(status, "failed to migrate detect buf back");
    append_to_event_array(event_array, detect_mig_event, VAR_NAME(detect_mig_event));

    {
        TracyCLZone(dnn_tracy_ctx, "DNN");
        status = clEnqueueNDRangeKernel(dnn_queue, ctx->dnn_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1, &detect_mig_event,
                                        &dnn_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
        append_to_event_array(event_array, dnn_event, VAR_NAME(dnn_event));
        TracyCLZoneSetEvent(dnn_event);
    }

    if (tmp_buf_ctx != NULL) {
        // Save detections to a temporary buffer for the quality evaluation pipeline
        size_t sz = DET_COUNT * sizeof(cl_int);

        cl_event copy_det_event;
        status = clEnqueueCopyBuffer(dnn_queue, ctx->detect_buf, tmp_buf_ctx->det, 0, 0, sz, 1,
                                     &dnn_event, &copy_det_event);
        CHECK_AND_RETURN(status, "failed to copy result buffer");
        append_to_event_array(tmp_buf_ctx->event_array, copy_det_event, VAR_NAME(copy_det_event));
        tmp_buf_ctx->copy_event_det = copy_det_event;
    }

    // segmentation not needed, so early exit
    if (!config.do_segment) {
        *out_event = dnn_event;
        return CL_SUCCESS;
    }

    cl_event postprocess_event, reconstruct_event;

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
        size_t sz = MASK_SZ1 * MASK_SZ2 * sizeof(cl_uchar);

        cl_event copy_event_seg;
        status = clEnqueueCopyBuffer(dnn_queue, ctx->postprocess_buf, tmp_buf_ctx->seg_post, 0,
                                     0, sz, 1, &postprocess_event, &copy_event_seg);
        CHECK_AND_RETURN(status, "failed to copy result buffer");
        append_to_event_array(tmp_buf_ctx->event_array, copy_event_seg,
                              VAR_NAME(copy_seg_event));
        tmp_buf_ctx->copy_event_seg_post = copy_event_seg;
    }

    // no reconstruction needed, so early exit
    if (!do_reconstruct) {
        *out_event = postprocess_event;
        return CL_SUCCESS;
    }

    cl_event reconstruct_wait_event = postprocess_event;

    if (ctx->config_flags & SEGMENT_4B) {

        status = encode_segment_4b(ctx->segment_4b_ctx, &postprocess_event, ctx->postprocess_buf,
                                   ctx->detect_buf,
                                   ctx->decompress_output_buf, event_array,
                                   &reconstruct_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue segment compression");
        status = clSetKernelArg(ctx->reconstruct_kernel, 0, sizeof(cl_mem),
                                &(ctx->decompress_output_buf));
    }

    {
        // reconstruct postprocessed data to RGBA segmentation mask
        TracyCLZone(ctx->local_tracy_ctx, "reconstruct");
        status = clEnqueueNDRangeKernel(ctx->local_queue, ctx->reconstruct_kernel,
                                        ctx->work_dim, NULL, ctx->global_size,
                                        ctx->local_size, 1, &reconstruct_wait_event,
                                        &reconstruct_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range reconstruct kernel");
        append_to_event_array(event_array, reconstruct_event,
                              VAR_NAME(reconstruct_event));
        TracyCLZoneSetEvent(reconstruct_event);
    }

    *out_event = reconstruct_event;

    return CL_SUCCESS;
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

    // figure out on which queue to run the dnn
    cl_command_queue queue;

    // if compression is enabled, the detect buf will be moved to the phone (passthrough device)
    if (LOCAL_DEVICE == config->device_type || ctx->config_flags & SEGMENT_4B) {
        queue = ctx->local_queue;
    } else if (REMOTE_DEVICE == config->device_type) {
        queue = ctx->remote_queue;
    } else {
        printf("ERROR: "
               "unknown device type to enqueue dnn to\n");
        return -1;
    }

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

//#define SAVE_OUTPUT
#ifdef SAVE_OUTPUT
        static bool saved = false;

        if(!saved) {
          int ret;
          char * postprocess_host = (char *) malloc(MASK_W * MASK_H * sizeof(cl_uchar));
          status = clEnqueueReadBuffer(ctx->local_queue, ctx->postprocess_buf, CL_TRUE, 0, MASK_W * MASK_H * sizeof(cl_uchar), postprocess_host, 2, wait_events, NULL);
          ret = write_bin_file("../tests/data/segmentation.bin",
                               postprocess_host, MASK_W * MASK_H * sizeof(cl_uchar));
          assert(ret == 0);
          free(postprocess_host);
          saved = true;
        }
#endif
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

    if (c->config_flags & SEGMENT_4B) {
        destroy_segment_4b(&(c->segment_4b_ctx));
    }

    free(c);
    *context = NULL;
    return CL_SUCCESS;
}

#ifdef __cplusplus
}
#endif