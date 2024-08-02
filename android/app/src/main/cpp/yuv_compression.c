//
// Created by rabijl on 11/16/23.
//

#include "yuv_compression.h"
#include "sharedUtils.h"
#include "poclImageProcessorTypes.h"

// todo: increase these values to match the values in dct.cl
#define BLK_W 1
#define BLK_H 1

/**
 * allocate memory for the codec context and set pointers to NULL;
 * @return pointer to context
 */
yuv_codec_context_t *create_yuv_context() {
    yuv_codec_context_t *context = (yuv_codec_context_t *) calloc(1, sizeof(yuv_codec_context_t));
    return context;
}

/**
 * setup the context so that it can be used.
 * @warning requires the following fields to be set beforehand <br>
 *  * height <br>
 *  * width <br>
 *  * enc_queue <br>
 *  * dec_queue <br>
 * @param codec_context context to init
 * @param ocl_context used to create all ocl objects
 * @param enc_device device to build encoder kernel for
 * @param dec_device device to build decoder kernel for
 * @param enable_resize parameter to enable resize extensions
 * @return cl status
 */
int
init_yuv_context(yuv_codec_context_t *codec_context, cl_context cl_context, cl_device_id enc_device,
                 cl_device_id dec_device, const char *source, const size_t src_size,
                 const int profile_compression_size) {

    int status;

    cl_device_id codec_devices[] = {enc_device, dec_device};

    cl_program program = clCreateProgramWithSource(cl_context, 1, &source, &src_size, &status);
    CHECK_AND_RETURN(status, "creation of codec program failed");

    status = clBuildProgram(program, 2, codec_devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building codec program failed");

    int total_pixels = codec_context->width * codec_context->height;
    codec_context->out_enc_y_buf = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, total_pixels, NULL,
                                                  &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_y_buf");
    codec_context->out_enc_uv_buf = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, total_pixels / 2,
                                                   NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_uv_buf");

    codec_context->enc_y_kernel = clCreateKernel(program, "encode_y", &status);
    CHECK_AND_RETURN(status, "creating encode_y kernel failed");

    status |= clSetKernelArg(codec_context->enc_y_kernel, 1, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->enc_y_kernel, 2, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->enc_y_kernel, 3, sizeof(cl_mem),
                             &(codec_context->out_enc_y_buf));
    CHECK_AND_RETURN(status, "setting encode_y kernel args failed");

    codec_context->enc_uv_kernel = clCreateKernel(program, "encode_uv", &status);
    CHECK_AND_RETURN(status, "creating encode_uv kernel failed");

    // todo: look into why height and width are swapped compared to the enc_y_kernel function
    status |= clSetKernelArg(codec_context->enc_uv_kernel, 1, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->enc_uv_kernel, 2, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->enc_uv_kernel, 3, sizeof(cl_mem),
                             &(codec_context->out_enc_uv_buf));
    CHECK_AND_RETURN(status, "setting encode_uv kernel args failed");

    codec_context->dec_y_kernel = clCreateKernel(program, "decode_y", &status);
    CHECK_AND_RETURN(status, "creating decode_y kernel failed");

    status = clSetKernelArg(codec_context->dec_y_kernel, 0, sizeof(cl_mem),
                            &(codec_context->out_enc_y_buf));
    status |= clSetKernelArg(codec_context->dec_y_kernel, 1, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->dec_y_kernel, 2, sizeof(cl_uint),
                             &(codec_context->height));
    CHECK_AND_RETURN(status, "setting decode_y kernel args failed");

    codec_context->dec_uv_kernel = clCreateKernel(program, "decode_uv", &status);
    CHECK_AND_RETURN(status, "creating decode_uv kernel failed");

    status = clSetKernelArg(codec_context->dec_uv_kernel, 0, sizeof(cl_mem),
                            &(codec_context->out_enc_uv_buf));
    // todo: same story as enc_uv_kernel
    status |= clSetKernelArg(codec_context->dec_uv_kernel, 1, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->dec_uv_kernel, 2, sizeof(cl_uint),
                             &(codec_context->width));
    CHECK_AND_RETURN(status, "setting decode_uv kernel args failed");

    codec_context->work_dim = 2;
    // set global work group sizes for codec kernels
    codec_context->y_global_size[0] = (size_t) (codec_context->width / BLK_W);
    codec_context->y_global_size[1] = (size_t) (codec_context->height / BLK_H);

    // todo: same story as enc_uv_kernel
    codec_context->uv_global_size[0] = (size_t) (codec_context->height / BLK_H) / 2;
    codec_context->uv_global_size[1] = (size_t) (codec_context->width / BLK_W) / 2;

    codec_context->output_format = YUV_NV12;

    codec_context->profile_compressed_size = profile_compression_size;
    codec_context->compressed_size = total_pixels * 3 / 2;

    clReleaseProgram(program);

    return 0;
}

/**
 * function to write the host data to the specified buffer
 * @param ctx yuv context
 * @param inp_host_buf array of image data
 * @param buf_size size of inp_host_buf
 * @param cl_buf buf to write to
 * @param wait_event optional event to wait on before writing
 * @param event_array save created event to this
 * @param result_event event that can be waited on
 * @return
 */
cl_int
write_buffer_yuv(const yuv_codec_context_t *ctx, const uint8_t *inp_host_buf, size_t buf_size,
                 cl_mem cl_buf, const cl_event *wait_event, event_array_t *event_array,
                 cl_event *result_event) {
    cl_int status;
    cl_event write_img_event, undef_img_mig_event;

    int wait_size = 0;
    if (NULL != wait_event) {
        wait_size = 1;
    }

    status = clEnqueueMigrateMemObjects(ctx->enc_queue, 1, &cl_buf,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, wait_size,
                                        wait_event, &undef_img_mig_event);
    CHECK_AND_RETURN(status, "could migrate enc buffer back before writing");
    append_to_event_array(event_array, undef_img_mig_event, VAR_NAME(undef_img_mig_event));

    status = clEnqueueWriteBuffer(ctx->enc_queue, cl_buf, CL_FALSE, 0, buf_size, inp_host_buf, 1,
                                  &undef_img_mig_event, &write_img_event);
    CHECK_AND_RETURN(status, "failed to write input image with yuv ctx");
    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));
    *result_event = write_img_event;
    return CL_SUCCESS;

}

/**
 * compress the image with the given context
 * @param cxt yuv context
 * @param wait_event event to wait on before enqueing compression
 * @param inp_buf yuv image to compress
 * @param out_buf resulting compressed yuv image
 * @param event_array used to append events to
 * @param result_event event that can be waited on when compression is done
 * @return CL_SUCCESS if everything goes well
 */
cl_int enqueue_yuv_compression(const yuv_codec_context_t *cxt, cl_event wait_event, cl_mem inp_buf,
                               cl_mem out_buf, event_array_t *event_array, cl_event *result_event) {
    cl_int status;
    cl_event enc_y_event, enc_uv_event, dec_y_event, dec_uv_event, mig_event;

    status = clSetKernelArg(cxt->enc_y_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |= clSetKernelArg(cxt->enc_uv_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |= clSetKernelArg(cxt->dec_y_kernel, 3, sizeof(cl_mem), &out_buf);
    status |= clSetKernelArg(cxt->dec_uv_kernel, 3, sizeof(cl_mem), &out_buf);
    CHECK_AND_RETURN(status, "failed to set kernel args");

    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_y_kernel, cxt->work_dim, NULL,
                                    cxt->y_global_size, NULL, 1, &wait_event, &enc_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_y_kernel");
    append_to_event_array(event_array, enc_y_event, VAR_NAME(enc_y_event));

    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_uv_kernel, cxt->work_dim, NULL,
                                    cxt->uv_global_size, NULL, 1, &wait_event, &enc_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_uv_kernel");
    append_to_event_array(event_array, enc_uv_event, VAR_NAME(enc_uv_event));

    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_y_kernel, cxt->work_dim, NULL,
                                    cxt->y_global_size, NULL, 1, &enc_y_event, &dec_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_y_kernel");
    append_to_event_array(event_array, dec_y_event, VAR_NAME(dec_y_event));

    // we have to wait for both since dec_y and dec_uv write to the same buffer and there is
    // no guarantee what happens if both dec_y and dec_uv write at the same time.
    cl_event dec_uv_wait_events[] = {enc_uv_event, dec_y_event};
    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_uv_kernel, cxt->work_dim, NULL,
                                    cxt->uv_global_size, NULL, 2, dec_uv_wait_events,
                                    &dec_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_uv_kernel");
    append_to_event_array(event_array, dec_uv_event, VAR_NAME(dec_uv_event));

    // move the intermediate buffers back to the phone after decompression.
    // Since we don't care about the contents, the latest state of the buffer is not moved
    // back from the remote.
    // https://man.opencl.org/clEnqueueMigrateMemObjects.html
    cl_mem migrate_bufs[] = {cxt->out_enc_y_buf, cxt->out_enc_uv_buf};
    status = clEnqueueMigrateMemObjects(cxt->enc_queue, 2, migrate_bufs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1, &dec_uv_event,
                                        &mig_event);
    CHECK_AND_RETURN(status, "failed to migrate buffers back");
    append_to_event_array(event_array, mig_event, VAR_NAME(mig_event));

    *result_event = mig_event;
    return 0;

}

size_t get_compression_size_yuv(const yuv_codec_context_t *cxt) {
    return cxt->compressed_size;
}

/**
 * Release all opencl objects created during init and set pointer to NULL
 * @warning objects such as enc_queue will still need to be released
 * @param context
 * @return CL_SUCCESS and otherwise an error
 */
cl_int destroy_yuv_context(yuv_codec_context_t **context) {

    yuv_codec_context_t *c = *context;

    if (NULL == c) {
        return 0;
    }

    COND_REL_MEM(c->out_enc_y_buf)

    COND_REL_MEM(c->out_enc_uv_buf)

    COND_REL_KERNEL(c->enc_y_kernel)

    COND_REL_KERNEL(c->enc_uv_kernel)

    COND_REL_KERNEL(c->dec_y_kernel)

    COND_REL_KERNEL(c->dec_uv_kernel)

    free(c);
    *context = NULL;
    return 0;
}