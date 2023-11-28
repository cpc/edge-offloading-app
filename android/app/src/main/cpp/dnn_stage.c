//
// Created by rabijl on 11/17/23.
//

#include "dnn_stage.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define EVAL_INTERVAL 9999999
#define VERBOSITY 0

const static int detection_count = 1 + MAX_DETECTIONS * 6;
const static int segmentation_count = MAX_DETECTIONS * MASK_W * MASK_H;
const static int seg_out_count = MASK_W * MASK_H * 4; // RGBA image
static int total_out_count = detection_count + segmentation_count;

dnn_context_t * create_dnn_context() {

    dnn_context_t *context = (dnn_context_t *) malloc( sizeof(dnn_context_t));
    context->out_detect_buf = NULL;
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
init_dnn_context(dnn_context_t * dnn_context, eval_context_t *eval_context, cl_context ocl_context, cl_device_id *device) {

    int status;

    cl_program  program = clCreateProgramWithBuiltInKernels(ocl_context, 1, device,
                                                            kernel_names, &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    if(NULL != eval_context) {

        init_eval_context(eval_context, ocl_context, program);
    }


    status = clBuildProgram(program, 1, device, NULL, NULL, NULL);
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
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    dnn_context->out_buf = clCreateBuffer(ocl_context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

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

    return 0;
}

cl_int
destroy_dnn_context(dnn_context_t **context) {

    dnn_context_t *c = *context;

    if(NULL == c) {
        return 0;
    }

    COND_REL_MEM(c->out_detect_buf)

    COND_REL_MEM(c->out_mask_buf)

    COND_REL_MEM(c->out_buf)

    COND_REL_MEM(c->postprocess_buf)

    COND_REL_MEM(c->reconstruct_buf)

    COND_REL_KERNEL(c->dnn_kernel)

    COND_REL_KERNEL(c->postprocess_kernel)

    COND_REL_KERNEL(c->reconstruct_kernel)

    free(c);
    *context = NULL;
    return 0;
}

cl_int
destroy_eval_context(eval_context_t **context) {
    
}