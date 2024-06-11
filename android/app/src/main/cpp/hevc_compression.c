//
// Created by rabijl on 11/20/23.
//

#include "hevc_compression.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"
#include <CL/cl_ext_pocl.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

/**
 * allocate memory for the codec context and set pointers to NULL;
 * @return pointer to context
 */
hevc_codec_context_t *create_hevc_context() {
    hevc_codec_context_t *context = (hevc_codec_context_t *) calloc(1,
                                                                    sizeof(hevc_codec_context_t));
    return context;
}

/**
 * private function to configure the hevc pipeline
 * @param codec_context
 * @param ocl_context
 * @param enc_device
 * @param dec_device
 * @param enable_resize
 * @param configure_kernel_name string of the configure kernel
 * @param encode_kernel_name string of the encode kernel
 * @param decode_kernel_name string of the decode kernel
 * @return
 */
cl_int
init_hevc_context_with_kernel_names(hevc_codec_context_t *codec_context, cl_context ocl_context,
                                    cl_device_id *enc_device, cl_device_id *dec_device,
                                    int enable_resize,
                                    const char *configure_kernel_name,
                                    const char *encode_kernel_name,
                                    const char *decode_kernel_name) {
    // this is the same value the media codec sets
    const uint64_t out_buf_size = 1024 * 1024;
    int status;

    // +2 for the null char and ';' char
    int kernel_names_length = strlen(configure_kernel_name) + strlen(encode_kernel_name) + 2;
    char *enc_program_kernel_names = malloc(kernel_names_length);
    strcpy(enc_program_kernel_names, configure_kernel_name);
    strcat(enc_program_kernel_names, ";");
    strcat(enc_program_kernel_names, encode_kernel_name);

    cl_program enc_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, enc_device,
                                                               enc_program_kernel_names,
                                                               &status);
    free(enc_program_kernel_names);
    CHECK_AND_RETURN(status, "could not create enc program");

    status = clBuildProgram(enc_program, 1, enc_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build enc program");

    cl_program dec_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, dec_device,
                                                               decode_kernel_name,
                                                               &status);
    CHECK_AND_RETURN(status, "could not create dec program");

    status = clBuildProgram(dec_program, 1, dec_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build dec program");

    // overprovision this buffer since the compressed output size can vary
    codec_context->comp_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, out_buf_size, NULL,
                                             &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    // needed to indicate how big the compressed image is
    codec_context->size_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL,
                                             &status);
    CHECK_AND_RETURN(status, "failed to create the output size buffer");

    // pocl content extension, allows for only the used part of the buffer to be transferred
    // https://registry.khronos.org/OpenCL/extensions/pocl/cl_pocl_content_size.html
    if (enable_resize > 0) {
        status = clSetContentSizeBufferPoCL(codec_context->comp_buf, codec_context->size_buf);
        CHECK_AND_RETURN(status, "could not apply content size extension");
    }

    codec_context->enc_kernel = clCreateKernel(enc_program, encode_kernel_name, &status);
    CHECK_AND_RETURN(status, "failed to create enc kernel");

    codec_context->config_kernel = clCreateKernel(enc_program, configure_kernel_name, &status);
    CHECK_AND_RETURN(status, "failed to create config kernel");

    codec_context->dec_kernel = clCreateKernel(dec_program, decode_kernel_name, &status);
    CHECK_AND_RETURN(status, "failed to create dec kernel");

    status = clSetKernelArg(codec_context->enc_kernel, 1, sizeof(cl_ulong),
                            &codec_context->input_size);
    status |= clSetKernelArg(codec_context->enc_kernel, 2, sizeof(cl_mem),
                             &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->enc_kernel, 3, sizeof(cl_ulong), &out_buf_size);
    status |= clSetKernelArg(codec_context->enc_kernel, 4, sizeof(cl_mem),
                             &(codec_context->size_buf));
    CHECK_AND_RETURN(status, "could not set hevc encoder kernel params \n");

    // configure the decoder kernel
    status = clSetKernelArg(codec_context->dec_kernel, 0, sizeof(cl_mem),
                            &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 1, sizeof(cl_mem),
                             &(codec_context->size_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 3, sizeof(cl_ulong),
                             &codec_context->output_size);
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
 * setup the context that uses the c2.android.hevc codec
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
init_c2_android_hevc_context(hevc_codec_context_t *codec_context, cl_context ocl_context,
                             cl_device_id *enc_device, cl_device_id *dec_device,
                             int enable_resize) {

    return init_hevc_context_with_kernel_names(codec_context, ocl_context, enc_device, dec_device,
                                               enable_resize,
                                               "pocl.configure.c2.android.hevc.yuv420nv21",
                                               "pocl.encode.c2.android.hevc.yuv420nv21",
                                               "pocl.decode.hevc.yuv420nv21");

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

    return init_hevc_context_with_kernel_names(codec_context, ocl_context, enc_device, dec_device,
                                               enable_resize,
                                               "pocl.configure.hevc.yuv420nv21",
                                               "pocl.encode.hevc.yuv420nv21",
                                               "pocl.decode.hevc.yuv420nv21");
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
enqueue_hevc_compression(const hevc_codec_context_t *cxt, cl_event *wait_event, cl_mem inp_buf,
                         cl_mem out_buf, event_array_t *event_array, cl_event *result_event) {

    assert((1 == cxt->codec_configured) &&
           "hevc codec was not configured before work was enqueued!\n");

    cl_int status;

//    cl_event enc_image_event,
    cl_event enc_event, dec_event, mig_event;

    status = clSetKernelArg(cxt->enc_kernel, 0, sizeof(cl_mem),
                            &(inp_buf));

    status |= clSetKernelArg(cxt->dec_kernel, 2, sizeof(cl_mem),
                             &(out_buf));

    // the compressed image and the size of the image are in these buffers respectively
    cl_mem migrate_bufs[] = {cxt->comp_buf, cxt->size_buf};

    // check if there is a wait event,
    // if the size is 0, the list needs to be NULL
    int32_t wait_event_size = 0;
    cl_event *wait_event_pointer = NULL;
    if (NULL != wait_event) {
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
    append_to_event_array(event_array, mig_event, VAR_NAME(mig_event));

    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_kernel, cxt->work_dim, NULL,
                                    cxt->enc_global_size,
                                    NULL, 1, &mig_event, &enc_event);
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
                            &(codec_context->config.framerate));
    status |= clSetKernelArg(codec_context->config_kernel, 3, sizeof(cl_int),
                             &(codec_context->config.i_frame_interval));
    status |= clSetKernelArg(codec_context->config_kernel, 4, sizeof(cl_int),
                             &(codec_context->config.bitrate));
    CHECK_AND_RETURN(status, "could not set configure kernel params \n");

    status = clEnqueueNDRangeKernel(codec_context->enc_queue, codec_context->config_kernel,
                                    codec_context->work_dim, NULL, codec_context->enc_global_size,
                                    NULL, 0, NULL, &config_codec_event);
    CHECK_AND_RETURN(status, "failed to enqueue configure kernel");
    append_to_event_array(event_array, config_codec_event, VAR_NAME(config_codec_event));

    *result_event = config_codec_event;

    codec_context->codec_configured = 1;

    return CL_SUCCESS;
}

/**
 * Release all opencl objects created during init and set pointer to NULL
 * @warning objects such as enc_queue will still need to be released
 * @param context
 * @return CL_SUCCESS and otherwise an error
 */
cl_int
destroy_hevc_context(hevc_codec_context_t **context) {

    hevc_codec_context_t *c = *context;

    if (NULL == c) {
        return 0;
    }

//    COND_REL_MEM(c->inp_buf)

    COND_REL_MEM(c->comp_buf)

    COND_REL_MEM(c->size_buf)

//    COND_REL_MEM(c->out_buf)

    COND_REL_KERNEL(c->enc_kernel)

    COND_REL_KERNEL(c->dec_kernel)

    COND_REL_KERNEL(c->config_kernel)

    free(c);
    *context = NULL;
    return 0;
}

/**
 * return 1 if the codecs are not the same
 * @param A
 * @param B
 * @return
 */
cl_int
hevc_configs_different(hevc_config_t A, hevc_config_t B) {

    return (A.i_frame_interval != B.i_frame_interval) ||
           (A.framerate != B.framerate) ||
           (A.bitrate != B.bitrate);
}

/**
 * set the values of the config to the context
 * @param ctx
 * @param new_config
 */
void
set_hevc_config(hevc_codec_context_t *ctx, const hevc_config_t *const new_config) {
    ctx->config.bitrate = new_config->bitrate;
    ctx->config.i_frame_interval = new_config->i_frame_interval;
    ctx->config.framerate = new_config->framerate;
}

/**
 * write the contents of the host buffer to the opencl buffer.
 * @param ctx
 * @param inp_host_buf
 * @param buf_size
 * @param cl_buf
 * @param wait_event
 * @param event_array
 * @param result_event
 * @return
 */
cl_int
write_buffer_hevc(const hevc_codec_context_t *ctx, uint8_t *inp_host_buf, size_t buf_size,
                  cl_mem cl_buf, cl_event *wait_event, event_array_t *event_array,
                  cl_event *result_event) {

    cl_int status;
    cl_event write_img_event, undef_mig_event;

    // check if there is a wait event,
    // if the size is 0, the list needs to be NULL
    int32_t wait_event_size = 0;
    if (NULL != wait_event) {
        wait_event_size = 1;
    }

    status = clEnqueueMigrateMemObjects(ctx->enc_queue, 1, &cl_buf,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                        wait_event_size, wait_event, &undef_mig_event);
    CHECK_AND_RETURN(status, "could migrate enc buffer back before writing");
    append_to_event_array(event_array, undef_mig_event, VAR_NAME(undef_mig_event));

    status = clEnqueueWriteBuffer(ctx->enc_queue, cl_buf, CL_FALSE, 0,
                                  buf_size, inp_host_buf, 1, &undef_mig_event, &write_img_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));
    *result_event = write_img_event;
    return status;
}

size_t
get_compression_size_hevc(hevc_codec_context_t *ctx) {
    return ctx->compressed_size;
}