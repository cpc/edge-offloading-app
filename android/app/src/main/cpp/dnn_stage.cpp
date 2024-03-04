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
    context->reconstruct_buf = NULL;
    context->dnn_kernel = NULL;
    context->postprocess_kernel = NULL;
    context->reconstruct_kernel = NULL;

    return context;
}

#define dnn_kernel_name  "pocl.dnn.detection.u8"
#define postprocess_kernel_name "pocl.dnn.segmentation_postprocess.u8"
#define reconstruct_kernel_name "pocl.dnn.segmentation_reconstruct.u8"
#define eval_kernel_name "pocl.dnn.eval_iou.f32"
char *kernel_names = dnn_kernel_name ";"
                     postprocess_kernel_name ";"
                     reconstruct_kernel_name ";"
                     eval_kernel_name
                     ;

int
init_eval_context(eval_context_t * eval_context, cl_context ocl_context, cl_program program) {

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

    cl_program  program = clCreateProgramWithBuiltInKernels(ocl_context, 1, dnn_device,
                                                            kernel_names, &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    assert(NULL == eval_context && "eval context is not supported yet");
    if(NULL != eval_context) {

        init_eval_context(eval_context, ocl_context, program);
    }

    cl_device_id devices[] = {*dnn_device, *reconstruct_device};
    status = clBuildProgram(program, 2, devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building of program failed");

    dnn_context->dnn_kernel = clCreateKernel(program, dnn_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    dnn_context->postprocess_kernel = clCreateKernel(program, postprocess_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    dnn_context->reconstruct_kernel = clCreateKernel(program, reconstruct_kernel_name, &status);
    CHECK_AND_RETURN(status, "creating reconstruct kernel failed");

    // set the kernel parameters
    status = clSetKernelArg(dnn_context->dnn_kernel, 1, sizeof(cl_uint), &(dnn_context->width));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 2, sizeof(cl_uint), &(dnn_context->height));
    status |=
            clSetKernelArg(dnn_context->dnn_kernel, 3, sizeof(cl_int), &(dnn_context->rotate_cw_degrees));
    status |= clSetKernelArg(dnn_context->dnn_kernel, 6, sizeof(cl_mem), &(dnn_context->out_mask_buf));
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    status = clSetKernelArg(dnn_context->postprocess_kernel, 1, sizeof(cl_mem),
                             &(dnn_context->out_mask_buf));
    status |= clSetKernelArg(dnn_context->postprocess_kernel, 2, sizeof(cl_mem),
                             &(dnn_context->postprocess_buf));
    CHECK_AND_RETURN(status, "could not assign postprocess kernel args");


    status = clSetKernelArg(dnn_context->reconstruct_kernel, 0, sizeof(cl_mem),
                            &(dnn_context->postprocess_buf));
    CHECK_AND_RETURN(status, "could not assign reconstruct_kernel args");

//    dnn_context->out_buf = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
//                                NULL, &status);
//    CHECK_AND_RETURN(status, "failed to create the output buffer");

    dnn_context->out_mask_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                     MASK_W * MASK_H * MAX_DETECTIONS * sizeof(cl_char),
                                     NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    dnn_context->postprocess_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                        MASK_W * MASK_H * sizeof(cl_uchar),
                                        NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

    dnn_context->reconstruct_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                        seg_out_count * sizeof(cl_uchar),
                                        NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation reconstructed buffer");

    dnn_context->work_dim = 1;
    dnn_context->global_size[0] = 1;
    dnn_context->local_size[0] = 1;


//    dnn_context->tracy_dnn_queue = TracyCLContext(ocl_context, &dnn_device);
//    dnn_context->tracy_reconstruct_queue = TracyCLContext(ocl_context, &reconstruct_device);

    return 0;
}


/**
 * Function to enqueue the opencl commands related to object detection.
 * @param wait_event event to wait on before starting these commands
 * @param dnn_device_idx index of device doing object detection
 * @param do_segment bool to enable/disable segmentation
 * @param detection_array array of bounding boxes of detected objects
 * @param segmentation_array bitmap of segments of detected objects
 * @param events struct of events of these commands
 * @return 0 if everything went well, otherwise a cl error number
 */
cl_int
enqueue_dnn(const dnn_context_t *ctx, const cl_event *wait_event, const int device_type, const uint32_t do_segment,
            const uint32_t input_format, cl_mem input_image,
            cl_mem detection_array, cl_mem segmentation_array,
            int save_out_buf, event_array_t *event_array,
            event_array_t *eval_event_array, cl_event *out_tmp_detect_copy_event,
            cl_event *out_tmp_postprocess_copy_event, cl_event *out_event, cl_mem inp_buf) {

    ZoneScoped;
    cl_int status;

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

    status = clSetKernelArg(ctx->dnn_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |= clSetKernelArg(ctx->dnn_kernel, 4, sizeof(cl_int), &input_format);
    status |= clSetKernelArg(ctx->dnn_kernel, 5, sizeof(cl_mem), &detection_array);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(ctx->postprocess_kernel, 0, sizeof(cl_mem), &detection_array);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

    status = clSetKernelArg(ctx->reconstruct_kernel, 1, sizeof(cl_mem),
                             &segmentation_array);
    CHECK_AND_RETURN(status, "could not assign buffers to reconstruct kernel");

    // figure out on which queue to run the dnn
    cl_command_queue dnn_queue;
    if(LOCAL_DEVICE == device_type){
        dnn_queue = ctx->local_queue;
    }else if(REMOTE_DEVICE == device_type) {
        dnn_queue = ctx->remote_queue;
    }else {
        LOGE("unknown device type to enqueue dnn to\n");
        return -1;
    }

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "assigning kernel params");
#endif

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

    if (do_segment) {
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

    return 0;
}

// TODO: update deallocation
cl_int
destroy_dnn_context(dnn_context_t **context) {

    dnn_context_t *c = *context;

    if(NULL == c) {
        return 0;
    }

//    COND_REL_MEM(c->out_detect_buf)

    COND_REL_MEM(c->out_mask_buf)

//    COND_REL_MEM(c->out_buf)

    COND_REL_MEM(c->postprocess_buf)

    COND_REL_MEM(c->reconstruct_buf)

    COND_REL_KERNEL(c->dnn_kernel)

    COND_REL_KERNEL(c->postprocess_kernel)

    COND_REL_KERNEL(c->reconstruct_kernel)

    free(c);
    *context = NULL;
    return 0;
}

// TODO: finish destory eval context
cl_int
destroy_eval_context(eval_context_t **context) {
    
}



#ifdef __cplusplus
}
#endif