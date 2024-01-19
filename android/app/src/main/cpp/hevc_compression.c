//
// Created by rabijl on 11/20/23.
//

#include "hevc_compression.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"
#include <CL/cl_ext_pocl.h>
#include <assert.h>

/**
 * allocate memory for the codec context and set pointers to NULL;
 * @return pointer to context
 */
hevc_codec_context_t *create_hevc_context() {
    hevc_codec_context_t *context = (hevc_codec_context_t *) malloc(sizeof(hevc_codec_context_t));
    context->inp_buf = NULL;
    context->comp_buf = NULL;
    context->size_buf = NULL;
    context->out_buf = NULL;
    context->enc_kernel = NULL;
    context->dec_kernel = NULL;
    context->config_kernel = NULL;
    context->codec_configured = 0;

    return context;
}

/**
 * setup the context so that it can be used.
 * @warning requires the following fields to be set beforehand <br>
 *  * height <br>
 *  * width <br>
 *  * img_buf_size <br>
 *  * enc_queue <br>
 *  * dec_queue <br>
 *  * host_imge_buf <br>
 *  * host_postprocess_buf <br>
 * @param codec_context context to init
 * @param ocl_context used to create all ocl objects
 * @param enc_device device to build encoder kernel for
 * @param dec_device device to build decoder kernel for
 * @param enable_resize parameter to enable resize extensions
 * @return cl status
 */
cl_int
init_hevc_context(hevc_codec_context_t *codec_context, cl_context ocl_context,
                  cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize) {
    // todo: set a better value
    uint64_t out_buf_size = 2000000;
    int status;
    cl_program enc_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, enc_device,
                                                               "pocl.configure.hevc.yuv420nv21;pocl.encode.hevc.yuv420nv21",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create enc program");

    status = clBuildProgram(enc_program, 1, enc_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build enc program");

    cl_program dec_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, dec_device,
                                                               "pocl.decode.hevc.yuv420nv21",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create dec program");

    status = clBuildProgram(dec_program, 1, dec_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build dec program");

    // input buffer for the encoder
    codec_context->inp_buf = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY,
                                            codec_context->img_buf_size,
                                            NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    // overprovision this buffer since the compressed output size can vary
    codec_context->comp_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, out_buf_size, NULL,
                                             &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    // needed to indicate how big the compressed image is
    codec_context->size_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL,
                                             &status);
    CHECK_AND_RETURN(status, "failed to create the output size buffer");

    codec_context->out_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE,
                                            codec_context->img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_buf");

    // pocl content extension, allows for only the used part of the buffer to be transferred
    // https://registry.khronos.org/OpenCL/extensions/pocl/cl_pocl_content_size.html
    if (enable_resize > 0) {
        status = clSetContentSizeBufferPoCL(codec_context->comp_buf, codec_context->size_buf);
        CHECK_AND_RETURN(status, "could not apply content size extension");
    }

    codec_context->enc_kernel = clCreateKernel(enc_program, "pocl.encode.hevc.yuv420nv21", &status);
    CHECK_AND_RETURN(status, "failed to create enc kernel");

    codec_context->config_kernel = clCreateKernel(enc_program, "pocl.configure.hevc.yuv420nv21", &status);
    CHECK_AND_RETURN(status, "failed to create config kernel");

    codec_context->dec_kernel = clCreateKernel(dec_program, "pocl.decode.hevc.yuv420nv21", &status);
    CHECK_AND_RETURN(status, "failed to create dec kernel");

    // the kernel uses uint64_t values for sizes, so put it in a bigger type
    uint64_t input_buf_size = codec_context->img_buf_size;

    // configure the encoder kernel
    status = clSetKernelArg(codec_context->enc_kernel, 0, sizeof(cl_mem),
                            &(codec_context->inp_buf));
    status |= clSetKernelArg(codec_context->enc_kernel, 1, sizeof(cl_ulong), &input_buf_size);
    status |= clSetKernelArg(codec_context->enc_kernel, 2, sizeof(cl_mem),
                             &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->enc_kernel, 3, sizeof(cl_ulong), &input_buf_size);
    status |= clSetKernelArg(codec_context->enc_kernel, 4, sizeof(cl_mem),
                             &(codec_context->size_buf));
    CHECK_AND_RETURN(status, "could not set hevc encoder kernel params \n");

    // configure the decoder kernel
    status = clSetKernelArg(codec_context->dec_kernel, 0, sizeof(cl_mem),
                            &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 1, sizeof(cl_mem),
                             &(codec_context->size_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 2, sizeof(cl_mem),
                             &(codec_context->out_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 3, sizeof(cl_ulong), &input_buf_size);
    CHECK_AND_RETURN(status, "could not set hevc decoder kernel params \n");

    // configure the configure kernel
    status = clSetKernelArg(codec_context->config_kernel, 0, sizeof(cl_int),
                            &(codec_context->width));
    status |= clSetKernelArg(codec_context->config_kernel, 1, sizeof(cl_int),
                             &(codec_context->height));
    CHECK_AND_RETURN(status, "could not set configure kernel params \n");

    codec_context->work_dim = 1;
    // built-in kernels, so one dimensional
    codec_context->enc_global_size[0] = 1;
    codec_context->dec_global_size[0] = 1;

    codec_context->output_format = YUV_PLANAR;

    clReleaseProgram(enc_program);
    clReleaseProgram(dec_program);

    return 0;
}

/**
 * compress the image with the given context
 * @param cxt the compression context
 * @param event_array
 * @param wait_event can be NULL if there is no event to wait on
 * @param result_event
 * @return cl status value
 */
cl_int
enqueue_hevc_compression(const hevc_codec_context_t *cxt, event_array_t *event_array,
                         cl_event *wait_event, cl_event *result_event) {

    assert((1 == cxt->codec_configured) && "hevc codec was not configured before work was enqueued!\n");

    cl_int status;

    cl_event enc_image_event, enc_event, dec_event, mig_event;

    // the compressed image and the size of the image are in these buffers respectively
    cl_mem migrate_bufs[] = {cxt->comp_buf, cxt->size_buf};

    // check if there is a wait event,
    // if the size is 0, the list needs to be NULL
    int32_t wait_event_size = 0;
    cl_event *wait_event_pointer = NULL;
    if (NULL != *wait_event) {
        wait_event_size = 1;
        wait_event_pointer = wait_event;
    }

    // The latest intermediary buffers can be ANYWHERE, therefore preemptively migrate them to
    // the enc device.
    status = clEnqueueMigrateMemObjects(cxt->enc_queue, 2, migrate_bufs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, wait_event_size,
                                        wait_event_pointer,
                                        &mig_event);
    CHECK_AND_RETURN(status, "could not migrate buffers back");

    status = clEnqueueWriteBuffer(cxt->enc_queue, cxt->inp_buf, CL_FALSE, 0,
                                  cxt->img_buf_size, cxt->host_img_buf, 0, NULL, &enc_image_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(event_array, enc_image_event, VAR_NAME(enc_image_event));

    cl_event wait_events[] = {enc_image_event, mig_event};
    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_kernel, cxt->work_dim, NULL,
                                    cxt->enc_global_size,
                                    NULL, 2, wait_events, &enc_event);
    CHECK_AND_RETURN(status, "failed to enqueue compression kernel");
    append_to_event_array(event_array, enc_event, VAR_NAME(enc_event));

    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_kernel, cxt->work_dim, NULL,
                                    cxt->dec_global_size,
                                    NULL, 1, &enc_event, &dec_event);
    CHECK_AND_RETURN(status, "failed to enqueue decompression kernel");
    append_to_event_array(event_array, dec_event, VAR_NAME(dec_event));

    *result_event = dec_event;

    return 0;
}
/**
 * A function to call the configure builtin kernel.
 * @NOTE this function is expensive
 * @param codec_context the hevc codec context configured with the right parameters
 * @param event_array array to append events to
 * @param result_event an output event that can be waited on
 * @return CL_SUCCESS and otherwise a cl error
 */
cl_int
configure_hevc_codec(hevc_codec_context_t *const codec_context, event_array_t *event_array,
                     cl_event *result_event) {
    cl_int status;

    cl_event config_codec_event;

    status = clSetKernelArg(codec_context->config_kernel, 2, sizeof(cl_int),
                            &(codec_context->framerate));
    status |= clSetKernelArg(codec_context->config_kernel, 3, sizeof(cl_int),
                             &(codec_context->i_frame_interval));
    status |= clSetKernelArg(codec_context->config_kernel, 4, sizeof(cl_int),
                             &(codec_context->bitrate));
    CHECK_AND_RETURN(status, "could not set configure kernel params \n");

    status = clEnqueueNDRangeKernel(codec_context->enc_queue, codec_context->config_kernel,
                                    codec_context->work_dim, NULL, codec_context->enc_global_size,
                                    NULL, 0, NULL, &config_codec_event);
    CHECK_AND_RETURN(status, "failed to enqueue configure kernel");
    append_to_event_array(event_array, config_codec_event, VAR_NAME(config_codec_event));

    *result_event = config_codec_event;

    codec_context->codec_configured = 1;
}

/**
 * Release all opencl objects created during init and set pointer to NULL
 * @warning objects such as enc_queue will still need to be released
 * @param context
 * @return CL_SUCCESS and otherwise an error
 */
cl_int
destory_hevc_context(hevc_codec_context_t **context) {

    hevc_codec_context_t *c = *context;

    if (NULL == c) {
        return 0;
    }

    COND_REL_MEM(c->inp_buf)

    COND_REL_MEM(c->comp_buf)

    COND_REL_MEM(c->size_buf)

    COND_REL_MEM(c->out_buf)

    COND_REL_KERNEL(c->enc_kernel)

    COND_REL_KERNEL(c->dec_kernel)

    COND_REL_KERNEL(c->config_kernel)

    free(c);
    *context = NULL;
    return 0;
}