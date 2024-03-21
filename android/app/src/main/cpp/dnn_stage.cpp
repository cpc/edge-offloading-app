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

    dnn_context_t *context = (dnn_context_t *) malloc( sizeof(dnn_context_t));
    context->out_mask_buf = NULL;
    context->postprocess_buf = NULL;
    context->dnn_kernel = NULL;
    context->postprocess_kernel = NULL;
    context->reconstruct_kernel = NULL;

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

int
init_eval_context(eval_context_t *eval_context, cl_context ocl_context, cl_program program) {

    assert(0 && "function hasn't been tested");

    cl_int status;

    eval_context->eval_kernel = clCreateKernel(program, eval_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating eval kernel failed");

    eval_context->eval_img_buf = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, total_out_count * sizeof(cl_int),
                                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval image buffer");

    eval_context->eval_out_buf = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval output buffer");

    eval_context->eval_out_mask_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                     MASK_W * MASK_H * MAX_DETECTIONS * sizeof(cl_char),
                                                     NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval segmentation mask buffer");

    eval_context->eval_postprocess_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                                        MASK_W * MASK_H * sizeof(cl_uchar), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create eval segmentation postprocessing buffer");

    eval_context->eval_iou_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, 1 * sizeof(cl_float), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval iou buffer");

}

int
init_dnn_context(dnn_context_t * dnn_context, eval_context_t *eval_context, cl_context ocl_context, cl_device_id *dnn_device,
                 cl_device_id *reconstruct_device) {

    int status;
    cl_device_id devices[] = {*dnn_device, *reconstruct_device};
    cl_program program = clCreateProgramWithBuiltInKernels(ocl_context, 2, devices,
                                                           kernel_names, &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    assert(NULL == eval_context && "eval context is not supported yet");
    if (NULL != eval_context) {

        init_eval_context(eval_context, ocl_context, program);
    }

    status = clBuildProgram(program, 2, devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building of program failed");

    dnn_context->dnn_kernel = clCreateKernel(program, dnn_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    dnn_context->postprocess_kernel = clCreateKernel(program, postprocess_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    dnn_context->reconstruct_kernel = clCreateKernel(program, reconstruct_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating reconstruct kernel failed");

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

    dnn_context->work_dim = 1;
    dnn_context->global_size[0] = 1;
    dnn_context->local_size[0] = 1;


//    dnn_context->tracy_dnn_queue = TracyCLContext(ocl_context, &dnn_device);
//    dnn_context->tracy_reconstruct_queue = TracyCLContext(ocl_context, &reconstruct_device);

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
write_buffer_dnn(const dnn_context_t *ctx, devic_type_enum device_type, uint8_t *inp_host_buf, size_t buf_size,
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

    cl_event run_dnn_event, read_detect_event;

    {
//        TracyCLZone(ctx->tracy_dnn_queue, "DNN");
        status = clEnqueueNDRangeKernel(dnn_queue, ctx->dnn_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1,
                                        wait_event, &run_dnn_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
        append_to_event_array(event_array, run_dnn_event, VAR_NAME(dnn_event));
//        TracyCLZoneSetEvent(run_dnn_event);
    }

    if (config.do_segment) {
        cl_event run_postprocess_event, mig_seg_event, run_reconstruct_event, read_segment_event;

        {
            // postprocess
//            TracyCLZone(ctx->tracy_dnn_queue, "postprocess");
            status = clEnqueueNDRangeKernel(dnn_queue, ctx->postprocess_kernel, ctx->work_dim,
                                            NULL,
                                            (ctx->global_size), (ctx->local_size), 1,
                                            &run_dnn_event, &run_postprocess_event);
            CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");
            append_to_event_array(event_array, run_postprocess_event, VAR_NAME(postprocess_event));
//            TracyCLZoneSetEvent(run_postprocess_event);
        }

        {
            // move postprocessed segmentation data to host device
//            TracyCLZone(ctx->tracy_reconstruct_queue, "migrate DNN");
            status = clEnqueueMigrateMemObjects(ctx->local_queue, 1,
                                                &(ctx->postprocess_buf), 0, 1,
                                                &run_postprocess_event,
                                                &mig_seg_event);
            CHECK_AND_RETURN(status, "failed to enqueue migration of postprocess buffer");
            append_to_event_array(event_array, mig_seg_event, VAR_NAME(mig_seg_event));
//            TracyCLZoneSetEvent(mig_seg_event);
        }

        {
            // reconstruct postprocessed data to RGBA segmentation mask
//            TracyCLZone(ctx->tracy_reconstruct_queue, "reconstruct");
            status = clEnqueueNDRangeKernel(ctx->local_queue, ctx->reconstruct_kernel, ctx->work_dim,
                                            NULL,
                                            ctx->global_size, ctx->local_size, 1,
                                            &mig_seg_event, &run_reconstruct_event);
            CHECK_AND_RETURN(status, "failed to enqueue ND range reconstruct kernel");
            append_to_event_array(event_array, run_reconstruct_event, VAR_NAME(reconstruct_event));
//            TracyCLZoneSetEvent(run_reconstruct_event);
        }

        *out_event = run_reconstruct_event;
    } else {
        *out_event = read_detect_event;
    }

    // TODO: migrate result buffers to local device

    return 0;
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

    free(c);
    *context = NULL;
    return CL_SUCCESS;
}

// TODO: finish destory eval context
cl_int
destroy_eval_context(eval_context_t **context) {
    assert(0 && "function not implemented yet");
}



#ifdef __cplusplus
}
#endif