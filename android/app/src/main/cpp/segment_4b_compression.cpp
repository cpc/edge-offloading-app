//
// Created by rabijl on 3.7.2024.
//

#include "segment_4b_compression.hpp"
#include "string.h"
#include "sharedUtils.h"
#include "opencl_utils.hpp"


segment_4b_context_t *
init_segment_4b(cl_context cl_ctx, cl_command_queue encode_queue, cl_command_queue decode_queue,
                cl_device_id devs[2], uint32_t width, uint32_t height,
                size_t source_size, char const source[],
                cl_int *ret_status) {

    cl_int status;
    cl_program program = clCreateProgramWithSource(cl_ctx, 1, &source, &source_size, &status);
    CHECK_AND_RETURN_NULL(status, ret_status, "could not create program");

    status = clBuildProgram(program, 2, devs, NULL, NULL, NULL);
    if (status == CL_BUILD_PROGRAM_FAILURE) {
      print_program_build_log(program, devs[1]);
    }
    CHECK_AND_RETURN_NULL(status, ret_status, "could not build program");

    cl_mem compress_buf = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, (height * width) / 2, NULL,
                                         &status);
    CHECK_AND_RETURN_NULL(status, ret_status, "could not create buffer");

    cl_kernel encode_kernel = clCreateKernel(program, "encode_bits_4", &status);
    CHECK_AND_RETURN_NULL(status, ret_status, "could not create encode kernel");

    cl_kernel decode_kernel = clCreateKernel(program, "decode_bits_4", &status);
    CHECK_AND_RETURN_NULL(status, ret_status, "could not create decode kernel");

    const uint32_t no_class_id = 80;

    status = clSetKernelArg(encode_kernel, 2, sizeof(cl_mem), &compress_buf);
    status |= clSetKernelArg(decode_kernel, 0, sizeof(cl_mem), &compress_buf);
    status |= clSetKernelArg(decode_kernel, 2, sizeof(cl_int), &no_class_id);
    CHECK_AND_RETURN_NULL(status, ret_status, "could not set kernel args");

    // todo: refactor so that on error program is released
    clReleaseProgram(program);

    // fancy trickery to allocate a struct with only const values
    segment_4b_context_t const_template = {
            encode_queue,
            decode_queue,
            compress_buf,
            encode_kernel,
            decode_kernel,
            width,
            height,
            no_class_id,
            {width / 2, height, 0},
            // the mali gpu only allows a max of 512 workitems per compute unit
            // this will give 16 * 30 = 480 workitems
            {width / 10, height / 4, 0},
            2, NULL};

    segment_4b_context_t *ret = (segment_4b_context_t *) malloc(sizeof(segment_4b_context_t));
    memcpy(ret, &const_template, sizeof(segment_4b_context_t));
    return ret;
}

cl_int
encode_segment_4b(segment_4b_context_t *ctx, const cl_event *wait_event,
                  cl_mem input, cl_mem detections, cl_mem output,
                  event_array_t *event_array, cl_event *event) {

    cl_int status;
    cl_event seg_enc_event, seg_dec_event, seg_mig_event;

    status = clEnqueueMigrateMemObjects(ctx->encode_queue, 1, &(ctx->compress_buf),
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1, wait_event,
                                        &(seg_mig_event));
    CHECK_AND_RETURN(status, "failed to migrate segment buffer back");
    append_to_event_array(event_array, seg_mig_event, VAR_NAME(seg_mig_event));

    status = clSetKernelArg(ctx->encode_kernel, 0, sizeof(cl_mem), &input);
    status |= clSetKernelArg(ctx->encode_kernel, 1, sizeof(cl_mem), &detections);
    CHECK_AND_RETURN(status, "could not set encode kernel args");

    status = clSetKernelArg(ctx->decode_kernel, 1, sizeof(cl_mem), &detections);
    status |= clSetKernelArg(ctx->decode_kernel, 3, sizeof(cl_mem), &output);
    CHECK_AND_RETURN(status, "could not set decode kernel args");

    {
//        TracyCLZone(dnn_ctx->remote_tracy_ctx, "seg_enc");
        status = clEnqueueNDRangeKernel(ctx->encode_queue, ctx->encode_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1, &seg_mig_event,
                                        &seg_enc_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range seg_enc kernel");
        append_to_event_array(event_array, seg_enc_event, VAR_NAME(seg_enc_event));
//        TracyCLZoneSetEvent(seg_enc_event);
    }

    {
//        TracyCLZone(dnn_ctx->local_tracy_ctx, "reconstruct");
        status = clEnqueueNDRangeKernel(ctx->decode_queue, ctx->decode_kernel, ctx->work_dim, NULL,
                                        ctx->global_size, ctx->local_size, 1, &seg_enc_event,
                                        &seg_dec_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range seg_dec kernel");
        append_to_event_array(event_array, seg_dec_event, VAR_NAME(seg_dec_event));
//        TracyCLZoneSetEvent(seg_dec_event);
    }

    *event = seg_dec_event;
    return CL_SUCCESS;

}

cl_int
destroy_segment_4b(segment_4b_context_t **context) {

    segment_4b_context_t *c = *context;
    if (NULL == c) {
        return CL_SUCCESS;
    }

    // no need to conditionally release since the object was created with const members,
    // and should only exist with valid data.
    clReleaseKernel(c->encode_kernel);
    clReleaseKernel(c->decode_kernel);
    clReleaseMemObject(c->compress_buf);

    free(c);
    *context = NULL;
    return CL_SUCCESS;
}
