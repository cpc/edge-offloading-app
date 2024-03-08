//
// Created by rabijl on 11/20/23.
//

#include "jpeg_compression.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"
#include <CL/cl_ext_pocl.h>
#include <assert.h>

#include <TracyC.h>

/**
 * allocate memory for the codec context and set pointers to NULL;
 * @return pointer to context
 */
jpeg_codec_context_t *
create_jpeg_context() {
    jpeg_codec_context_t *context = (jpeg_codec_context_t *) malloc(sizeof(jpeg_codec_context_t));
//    context->inp_buf = NULL;
    context->comp_buf = NULL;
    context->size_buf = NULL;
//    context->out_buf = NULL;
    context->enc_kernel = NULL;
    context->dec_kernel = NULL;

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
int
init_jpeg_context(jpeg_codec_context_t *codec_context, cl_context ocl_context,
                  cl_device_id *enc_device, cl_device_id *dec_device, int enable_resize) {

    cl_int status;
    int total_pixels = codec_context->width * codec_context->height;
    // a default value
    codec_context->quality = 80;
    codec_context->output_format = RGB;
    cl_program enc_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, enc_device,
                                                               "pocl.compress.to.jpeg.yuv420nv21",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create enc program");

    status = clBuildProgram(enc_program, 1, enc_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build enc program");

    cl_program dec_program = clCreateProgramWithBuiltInKernels(ocl_context, 1, dec_device,
                                                               "pocl.decompress.from.jpeg.rgb888",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create dec program");

    status = clBuildProgram(dec_program, 1, dec_device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "could not build dec program");

//    codec_context->inp_buf = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY, total_pixels * 3 / 2,
//                                            NULL, &status);
//    CHECK_AND_RETURN(status, "failed to create the input buffer");

    // overprovision this buffer since the compressed output size can vary
    codec_context->comp_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, total_pixels * 3 / 2,
                                             NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    // needed to indicate how big the compressed image is
    codec_context->size_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL,
                                             &status);
    CHECK_AND_RETURN(status, "failed to create the output size buffer");

//    codec_context->out_buf = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, total_pixels * 3, NULL,
//                                            &status);
//    CHECK_AND_RETURN(status, "failed to create out_buf");

    // pocl content extension, allows for only the used part of the buffer to be transferred
    // https://registry.khronos.org/OpenCL/extensions/pocl/cl_pocl_content_size.html
    if (enable_resize > 0) {
        status = clSetContentSizeBufferPoCL(codec_context->comp_buf, codec_context->size_buf);
        CHECK_AND_RETURN(status, "could not apply content size extension");
    }

    codec_context->enc_kernel = clCreateKernel(enc_program, "pocl.compress.to.jpeg.yuv420nv21",
                                               &status);
    CHECK_AND_RETURN(status, "failed to create enc kernel");

    codec_context->dec_kernel = clCreateKernel(dec_program, "pocl.decompress.from.jpeg.rgb888",
                                               &status);
    CHECK_AND_RETURN(status, "failed to create dec kernel");

//    status = clSetKernelArg(codec_context->enc_kernel, 0, sizeof(cl_mem),
//                            &(codec_context->inp_buf));
    status |= clSetKernelArg(codec_context->enc_kernel, 1, sizeof(cl_int), &(codec_context->width));
    status |= clSetKernelArg(codec_context->enc_kernel, 2, sizeof(cl_int),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->enc_kernel, 3, sizeof(cl_int),
                             &(codec_context->quality));
    status |= clSetKernelArg(codec_context->enc_kernel, 4, sizeof(cl_mem),
                             &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->enc_kernel, 5, sizeof(cl_mem),
                             &(codec_context->size_buf));
    CHECK_AND_RETURN(status, "failed to assign kernel parameters to  enc kernel");

    status = clSetKernelArg(codec_context->dec_kernel, 0, sizeof(cl_mem),
                            &(codec_context->comp_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 1, sizeof(cl_mem),
                             &(codec_context->size_buf));
    status |= clSetKernelArg(codec_context->dec_kernel, 2, sizeof(cl_int), &(codec_context->width));
    status |= clSetKernelArg(codec_context->dec_kernel, 3, sizeof(cl_int),
                             &(codec_context->height));
//    status |= clSetKernelArg(codec_context->dec_kernel, 4, sizeof(cl_mem),
//                             &(codec_context->out_buf));
    CHECK_AND_RETURN(status, "failed to assign kernel parameters to  dec kernel");

    // built-in kernels, so one dimensional
    codec_context->work_dim = 1;
    codec_context->enc_global_size[0] = 1;
    codec_context->dec_global_size[0] = 1;

    clReleaseProgram(enc_program);
    clReleaseProgram(dec_program);

    return 0;
}

cl_int
write_buffer_jpeg(const jpeg_codec_context_t * ctx, uint8_t * inp_host_buf, size_t buf_size,
                  cl_mem cl_buf, event_array_t *event_array, cl_event *result_event) {

    cl_int status;
    cl_event write_img_event;
    status = clEnqueueWriteBuffer(ctx->enc_queue, cl_buf, CL_FALSE, 0,
                                  buf_size, inp_host_buf, 0, NULL, &write_img_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));
    *result_event = write_img_event;
    return status;
}

///**
// * compress the image with the given context
// * @param cxt the compression context
// * @param event_array
// * @param result_event
// * @return cl status value
// * @param cxt
// * @param wait_event
// * @param inp_buf
// * @param out_buf
//
// */
/**
 * compress the image with the given context
 * @param cxt compression context
 * @param wait_event wait on this event before compressing
 * @param inp_buf raw yuv image to read
 * @param out_buf compressed jpeg image
 * @param event_array
 * @param result_event event that can be waited on
 * @return
 */
cl_int
enqueue_jpeg_compression(const jpeg_codec_context_t *cxt, cl_event wait_event, cl_mem inp_buf,
                         cl_mem out_buf, event_array_t *event_array, cl_event *result_event) {

    assert (0 <= cxt->quality && cxt->quality <= 100);
    TracyCZone(ctx, 1);

    cl_int status;

    status = clSetKernelArg(cxt->enc_kernel, 0, sizeof(cl_mem),
                            &inp_buf);
    status = clSetKernelArg(cxt->enc_kernel, 3, sizeof(cl_int), &(cxt->quality));
    CHECK_AND_RETURN(status, "could not set compression quality");

    status |= clSetKernelArg(cxt->dec_kernel, 4, sizeof(cl_mem),
                             &out_buf);

//    cl_event write_img_event,
    cl_event enc_event, dec_event, undef_mig_event, mig_event;

    // the compressed image and the size of the image are in these buffers respectively
    cl_mem migrate_bufs[] = {cxt->comp_buf, cxt->size_buf};

    // The latest intermediary buffers can be ANYWHERE, therefore preemptively migrate them to
    // the enc device.
    status = clEnqueueMigrateMemObjects(cxt->enc_queue, 2, migrate_bufs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL,
                                        &undef_mig_event);
    CHECK_AND_RETURN(status, "could not migrate buffers back");
    append_to_event_array(event_array, undef_mig_event, VAR_NAME(undef_mig_event));

//    status = clEnqueueWriteBuffer(cxt->enc_queue, cxt->inp_buf, CL_FALSE, 0,
//                                  cxt->img_buf_size, cxt->host_img_buf, 0, NULL, &write_img_event);
//    CHECK_AND_RETURN(status, "failed to write image to enc buffers");

//    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));
//
//    cl_event wait_events[] = {write_img_event, undef_mig_event};

    cl_event wait_events[] = {wait_event, undef_mig_event};
    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_kernel, cxt->work_dim, NULL,
                                    cxt->enc_global_size,
                                    NULL, 2, wait_events, &enc_event);
    CHECK_AND_RETURN(status, "failed to enqueue compression kernel");
    append_to_event_array(event_array, enc_event, VAR_NAME(enc_event));

    // explicitly migrate the image instead of having enqueueNDrangekernel do it implicitly
    // so that we can profile the transfer times
    status = clEnqueueMigrateMemObjects(cxt->dec_queue, 2, migrate_bufs, 0, 1, &enc_event,
                                        &mig_event);
    CHECK_AND_RETURN(status, "failed to enqueue migration to remote");
    append_to_event_array(event_array, mig_event, VAR_NAME(mig_event));

    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_kernel, cxt->work_dim, NULL,
                                    cxt->dec_global_size,
                                    NULL, 1, &mig_event, &dec_event);
    CHECK_AND_RETURN(status, "failed to enqueue decompression kernel");
    append_to_event_array(event_array, dec_event, VAR_NAME(dec_event));

    *result_event = dec_event;

    TracyCZoneEnd(ctx);
    return 0;
}

/**
 * Release all opencl objects created during init and set pointer to NULL
 * @warning objects such as enc_queue will still need to be released
 * @param context
 * @return CL_SUCCESS and otherwise an error
 */
cl_int
destroy_jpeg_context(jpeg_codec_context_t **context) {

    jpeg_codec_context_t *c = *context;

    if (NULL == c) {
        return 0;
    }

//    COND_REL_MEM(c->inp_buf)

    COND_REL_MEM(c->comp_buf)

    COND_REL_MEM(c->size_buf)

//    COND_REL_MEM(c->out_buf)

    COND_REL_KERNEL(c->enc_kernel)

    COND_REL_KERNEL(c->dec_kernel)

    free(c);
    *context = NULL;
    return 0;
}