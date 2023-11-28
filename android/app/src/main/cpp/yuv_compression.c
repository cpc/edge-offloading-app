//
// Created by rabijl on 11/16/23.
//

#include "yuv_compression.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"


// todo: increase these values to match the values in dct.cl
#define BLK_W 1
#define BLK_H 1

/**
 * allocate memory for the codec context and set pointers to NULL;
 * @return pointer to context
 */
yuv_codec_context_t *
create_yuv_context() {
    yuv_codec_context_t *context = (yuv_codec_context_t *) malloc(sizeof(yuv_codec_context_t));
    context->inp_buf = NULL;
    context->out_enc_y_buf = NULL;
    context->out_enc_uv_buf = NULL;
    context->out_buf = NULL;
    context->enc_y_kernel = NULL;
    context->enc_uv_kernel = NULL;
    context->dec_y_kernel = NULL;
    context->dec_uv_kernel = NULL;
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
init_yuv_context(yuv_codec_context_t *codec_context, cl_context cl_context, cl_device_id enc_device,
                 cl_device_id dec_device, const char *source, const size_t
                 src_size) {

    int status;

    cl_device_id codec_devices[] = {enc_device, dec_device};

    cl_program program = clCreateProgramWithSource(cl_context, 1, &source,
                                                   &src_size,
                                                   &status);
    CHECK_AND_RETURN(status, "creation of codec program failed");

    status = clBuildProgram(program, 2, codec_devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building codec program failed");

    // proxy device buffers
    // input for compression
    // important that it is read only since both enc kernels read from it.
    codec_context->inp_buf = clCreateBuffer(cl_context, CL_MEM_READ_ONLY,
                                            codec_context->img_buf_size,
                                            NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    int total_pixels = codec_context->width * codec_context->height;
    codec_context->out_enc_y_buf = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, total_pixels, NULL,
                                                  &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_y_buf");
    codec_context->out_enc_uv_buf = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, total_pixels / 2,
                                                   NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_uv_buf");

    codec_context->out_buf = clCreateBuffer(cl_context, CL_MEM_READ_WRITE,
                                            codec_context->img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_buf");

    codec_context->enc_y_kernel = clCreateKernel(program, "encode_y", &status);
    CHECK_AND_RETURN(status, "creating encode_y kernel failed");
    int narg = 0;
    status = clSetKernelArg(codec_context->enc_y_kernel, narg++, sizeof(cl_mem),
                            &(codec_context->inp_buf));
    status |= clSetKernelArg(codec_context->enc_y_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->enc_y_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->enc_y_kernel, narg++, sizeof(cl_mem),
                             &(codec_context->out_enc_y_buf));
    CHECK_AND_RETURN(status, "setting encode_y kernel args failed");

    codec_context->enc_uv_kernel = clCreateKernel(program, "encode_uv", &status);
    CHECK_AND_RETURN(status, "creating encode_uv kernel failed");
    narg = 0;
    status = clSetKernelArg(codec_context->enc_uv_kernel, narg++, sizeof(cl_mem),
                            &(codec_context->inp_buf));
    // todo: look into why height and width are swapped compared to the enc_y_kernel function
    status |= clSetKernelArg(codec_context->enc_uv_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->enc_uv_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->enc_uv_kernel, narg++, sizeof(cl_mem),
                             &(codec_context->out_enc_uv_buf));
    CHECK_AND_RETURN(status, "setting encode_uv kernel args failed");

    codec_context->dec_y_kernel = clCreateKernel(program, "decode_y", &status);
    CHECK_AND_RETURN(status, "creating decode_y kernel failed");
    narg = 0;
    status = clSetKernelArg(codec_context->dec_y_kernel, narg++, sizeof(cl_mem),
                            &(codec_context->out_enc_y_buf));
    status |= clSetKernelArg(codec_context->dec_y_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->dec_y_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->dec_y_kernel, narg++, sizeof(cl_mem),
                             &(codec_context->out_buf));
    CHECK_AND_RETURN(status, "setting decode_y kernel args failed");

    codec_context->dec_uv_kernel = clCreateKernel(program, "decode_uv", &status);
    CHECK_AND_RETURN(status, "creating decode_uv kernel failed");
    narg = 0;
    status = clSetKernelArg(codec_context->dec_uv_kernel, narg++, sizeof(cl_mem),
                            &(codec_context->out_enc_uv_buf));
    // todo: same story as enc_uv_kernel
    status |= clSetKernelArg(codec_context->dec_uv_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->height));
    status |= clSetKernelArg(codec_context->dec_uv_kernel, narg++, sizeof(cl_uint),
                             &(codec_context->width));
    status |= clSetKernelArg(codec_context->dec_uv_kernel, narg++, sizeof(cl_mem),
                             &(codec_context->out_buf));
    CHECK_AND_RETURN(status, "setting decode_uv kernel args failed");

    codec_context->work_dim = 2;
    // set global work group sizes for codec kernels
    codec_context->y_global_size[0] = (size_t) (codec_context->width / BLK_W);
    codec_context->y_global_size[1] = (size_t) (codec_context->height / BLK_H);

    // todo: same story as enc_uv_kernel
    codec_context->uv_global_size[0] = (size_t) (codec_context->height / BLK_H) / 2;
    codec_context->uv_global_size[1] = (size_t) (codec_context->width / BLK_W) / 2;

    codec_context->output_format = YUV_SEMI_PLANAR;

    clReleaseProgram(program);

    return 0;
}

/**
 * compress the image with the given context
 * @param cxt the compression context
 * @param event_array
 * @param result_event
 * @return cl status value
 */
cl_int
enqueue_yuv_compression(const yuv_codec_context_t *cxt, event_array_t *event_array,
                        cl_event *result_event) {
    cl_int status;
    cl_event write_img_event, enc_y_event, enc_uv_event,
            dec_y_event, dec_uv_event, migrate_event;

    status = clEnqueueWriteBuffer(cxt->enc_queue, cxt->inp_buf, CL_FALSE, 0,
                                  cxt->img_buf_size, cxt->host_img_buf, 0, NULL, &write_img_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(event_array, write_img_event, VAR_NAME(write_img_event));

    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_y_kernel, cxt->work_dim, NULL,
                                    cxt->y_global_size,
                                    NULL, 1, &write_img_event, &enc_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_y_kernel");
    append_to_event_array(event_array, enc_y_event, VAR_NAME(enc_y_event));

    status = clEnqueueNDRangeKernel(cxt->enc_queue, cxt->enc_uv_kernel, cxt->work_dim, NULL,
                                    cxt->uv_global_size,
                                    NULL, 1, &write_img_event, &enc_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_uv_kernel");
    append_to_event_array(event_array, enc_uv_event, VAR_NAME(enc_uv_event));

    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_y_kernel, cxt->work_dim, NULL,
                                    cxt->y_global_size,
                                    NULL, 1, &enc_y_event, &dec_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_y_kernel");
    append_to_event_array(event_array, dec_y_event, VAR_NAME(dec_y_event));

    // we have to wait for both since dec_y and dec_uv write to the same buffer and there is
    // no guarantee what happens if both dec_y and dec_uv write at the same time.
    cl_event dec_uv_wait_events[] = {enc_uv_event, dec_y_event};
    status = clEnqueueNDRangeKernel(cxt->dec_queue, cxt->dec_uv_kernel, cxt->work_dim, NULL,
                                    cxt->uv_global_size,
                                    NULL, 2, dec_uv_wait_events, &dec_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_uv_kernel");
    append_to_event_array(event_array, dec_uv_event, VAR_NAME(dec_uv_event));

    // move the intermediate buffers back to the phone after decompression.
    // Since we don't care about the contents, the latest state of the buffer is not moved
    // back from the remote.
    // https://man.opencl.org/clEnqueueMigrateMemObjects.html
    cl_mem migrate_bufs[] = {cxt->out_enc_y_buf, cxt->out_enc_uv_buf};
    status = clEnqueueMigrateMemObjects(cxt->enc_queue, 2, migrate_bufs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1,
                                        &dec_uv_event, &migrate_event);
    CHECK_AND_RETURN(status, "failed to migrate buffers back");
    append_to_event_array(event_array, migrate_event, VAR_NAME(migrate_event));

    // set the event that other ocl commands can wait for
//    *result_event = dec_uv_event;
    *result_event = migrate_event;
    return 0;

}

/**
 * Release all opencl objects created during init and set pointer to NULL
 * @warning objects such as enc_queue will still need to be released
 * @param context
 * @return CL_SUCCESS and otherwise an error
 */
cl_int
destory_yuv_context(yuv_codec_context_t **context) {

    yuv_codec_context_t *c = *context;

    if (NULL == c) {
        return 0;
    }

    COND_REL_MEM(c->inp_buf)

    COND_REL_MEM(c->out_enc_y_buf)

    COND_REL_MEM(c->out_enc_uv_buf)

    COND_REL_MEM(c->out_buf)

    COND_REL_KERNEL(c->enc_y_kernel)

    COND_REL_KERNEL(c->enc_uv_kernel)

    COND_REL_KERNEL(c->dec_y_kernel)

    COND_REL_KERNEL(c->dec_uv_kernel)
    
    free(c);
    *context = NULL;
    return 0;
}