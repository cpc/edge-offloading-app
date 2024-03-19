//
// Created by rabijl on 27.2.2024.
//

#include <time.h>
#include <errno.h>
#include <assert.h>
#include "poclImageProcessorV2.h"
#include "platform.h"
#include "sharedUtils.h"
#include <ctime>
#include <stdlib.h>
#include "config.h"

#define DEBUB_SEMAPHORES

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_EVENTS 99
#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define DET_COUNT (1 + MAX_DETECTIONS * 6)
#define SEG_COUNT (MAX_DETECTIONS * MASK_W * MASK_H)
#define SEG_OUT_COUNT (MASK_W * MASK_H * 4)
// TODO: check if this size actually needed, or one value can be dropped
#define TOT_OUT_COUNT (DET_COUNT + SEG_COUNT)

/**
 * check that the code was built with the right features enables
 * @param config_flags
 * @return 0 if everything oke, otherwise -1
 */
int
supports_config_flags(const int config_flags) {

#ifdef DISABLE_HEVC
    if(config_flags & HEVC_COMPRESSION) {
      LOGE("poclImageProcessor was built without HEVC compression");
      return -1;
    }
#endif

#ifdef DISABLE_JPEG
    if(config_flags & JPEG_COMPRESSION) {
      LOGE("poclImageProcessor was built without JPEG compression");
      return -1;
    }
#endif

    return 0;
}

/**
 * create a pipeline context that has the requested codecs initialized
 * @param ctx address to the pipeline ctx
 * @param width
 * @param height
 * @param config_flags
 * @param codec_sources
 * @param src_size size of codec_sources
 * @param cl_ctx opencl context used to create opencl objects
 * @param devices list of devices
 * @param no_devs number of devices
 * @return opencl status
 */
int
setup_pipeline_context(pipeline_context *ctx, const int width, const int height,
                       const int config_flags,
                       const char *codec_sources, const size_t src_size, cl_context cl_ctx,
                       cl_device_id *devices, cl_uint no_devs) {


    if (supports_config_flags(config_flags) != 0) {
        return -1;
    }
    cl_int status;

    ctx->config_flags = config_flags;
    ctx->width = width;
    ctx->height = height;
    cl_command_queue_properties cq_properties = CL_QUEUE_PROFILING_ENABLE;

    // current assumes that there are three devices
    // 1. a local compression device
    // 2. a remote decompression device
    // 3. a remote dnn device
    assert(3 <= no_devs);
    ctx->queue_count = no_devs;
    ctx->enq_queues = (cl_command_queue *) calloc(no_devs, sizeof(cl_command_queue));
    for (unsigned i = 0; i < no_devs; ++i) {
        ctx->enq_queues[i] = clCreateCommandQueue(cl_ctx, devices[i], cq_properties, &status);
        CHECK_AND_RETURN(status, "creating command queue failed");
    }


    size_t img_buf_size = sizeof(cl_uchar) * width * height * 3 / 2;
    ctx->inp_yuv_mem = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY, img_buf_size,
                                      NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the input buffer");
    // setting it to the maximum size which is an rgb image
    ctx->comp_to_dnn_buf = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, height * width * 3,
                                          NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the comp to dnn buf");
    ctx->host_inp_buf = (cl_uchar *) malloc(img_buf_size);
    ctx->host_inp_buf_size = img_buf_size;

    // setup all codec contexts
    if (YUV_COMPRESSION & ctx->config_flags) {

        // for some reason the basic device fails to build the program
        // when building from source, therefore use proxy device (which is the mobile gpu)
        ctx->yuv_context = create_yuv_context();
        ctx->yuv_context->height = height;
        ctx->yuv_context->width = width;
        ctx->yuv_context->enc_queue = ctx->enq_queues[1];
        ctx->yuv_context->dec_queue = ctx->enq_queues[2];
        status = init_yuv_context(ctx->yuv_context, cl_ctx, devices[1], devices[2], codec_sources,
                                  src_size);

        CHECK_AND_RETURN(status, "init of codec kernels failed");
    }

#ifndef DISABLE_JPEG
    if ((JPEG_COMPRESSION & ctx->config_flags)) {

        ctx->jpeg_context = create_jpeg_context();
        ctx->jpeg_context->height = height;
        ctx->jpeg_context->width = width;
        ctx->jpeg_context->enc_queue = ctx->enq_queues[0];
        ctx->jpeg_context->dec_queue = ctx->enq_queues[2];
        status = init_jpeg_context(ctx->jpeg_context, cl_ctx, &devices[0], &devices[2], 1);

        CHECK_AND_RETURN(status, "init of codec kernels failed");
    }
#endif // DISABLE_JPEG
#ifndef DISABLE_HEVC
    // TODO: update hevc contexts
    if (HEVC_COMPRESSION & ctx->config_flags) {

        ctx->hevc_context = create_hevc_context();
        ctx->hevc_context->height = height;
        ctx->hevc_context->width = width;
//        ctx->hevc_context->img_buf_size = img_buf_size;
        ctx->hevc_context->enc_queue = ctx->enq_queues[0];
        ctx->hevc_context->dec_queue = ctx->enq_queues[2];
//        ctx->hevc_context->host_img_buf = host_img_buf;
//        ctx->hevc_context->host_postprocess_buf = host_postprocess_buf;
        status = init_hevc_context(ctx->hevc_context, cl_ctx, &devices[0], &devices[2], 1);
        CHECK_AND_RETURN(status, "init of hevc codec kernels failed");
    }
    if (SOFTWARE_HEVC_COMPRESSION & ctx->config_flags) {

        ctx->software_hevc_context = create_hevc_context();
        ctx->software_hevc_context->height = height;
        ctx->software_hevc_context->width = width;
//        ctx->software_hevc_context->img_buf_size = img_buf_size;
        ctx->software_hevc_context->enc_queue = ctx->enq_queues[0];
        ctx->software_hevc_context->dec_queue = ctx->enq_queues[2];
//        software_hevc_context->host_img_buf = host_img_buf;
//        software_hevc_context->host_postprocess_buf = host_postprocess_buf;
        status = init_c2_android_hevc_context(ctx->software_hevc_context, cl_ctx, &devices[0],
                                              &devices[2], 1);
        CHECK_AND_RETURN(status, "init of hevc codec kernels failed");
    }
#endif // DISABLE_HEVC

    ctx->dnn_context = create_dnn_context();
    ctx->dnn_context->height = height;
    ctx->dnn_context->width = width;
    ctx->dnn_context->rotate_cw_degrees = 90;
    ctx->dnn_context->remote_queue = ctx->enq_queues[2];
    ctx->dnn_context->local_queue = ctx->enq_queues[0];
    // set the eval context to NULL for now
    // TODO: setup eval context
    status = init_dnn_context(ctx->dnn_context, NULL, cl_ctx,
                              &devices[2], &devices[0]);


    ctx->event_array = create_event_array(MAX_EVENTS);

    return CL_SUCCESS;
}

/**
 * free up all the objects in the pipeline context
 * @note does not free the memory of the context itself, just objects it
 * is in charge with managing
 * @param ctx context to be destroyed
 * @return opencl status code
 */
cl_int
destroy_pipeline_context(pipeline_context ctx) {
    destroy_dnn_context(&ctx.dnn_context);

#ifndef DISABLE_HEVC
    destroy_hevc_context(&ctx.hevc_context);
    destroy_hevc_context(&ctx.software_hevc_context);
#endif DISABLE_HEVC
#ifndef DISABLE_JPEG
    destroy_jpeg_context(&ctx.jpeg_context);
#endif DISABLE_JPEG
    destroy_yuv_context(&ctx.yuv_context);

    COND_REL_MEM(ctx.inp_yuv_mem);
    COND_REL_MEM(ctx.comp_to_dnn_buf);
    free(ctx.host_inp_buf);
    for (int i = 0; i < ctx.queue_count; i++) {
        COND_REL_QUEUE(ctx.enq_queues[i]);
    }
    free_event_array(&ctx.event_array);

    return CL_SUCCESS;
}

/**
 * create a context that with a number of pipelines
 * @param ret_ctx destination pointer to the created object
 * @param max_lanes the number of images that can be simultaneously processed
 * @param width
 * @param height
 * @param config_flags what options to enable
 * @param codec_sources
 * @param src_size
 * @param fd filedescriptor of the file where profiling data is written to
 * @return
 */
int
create_pocl_image_processor_context(pocl_image_processor_context **ret_ctx, const int max_lanes,
                                    const int width, const int height, const int config_flags,
                                    const char *codec_sources, const size_t src_size, int fd) {

    if (supports_config_flags(config_flags) != 0) {
        return -1;
    }

    pocl_image_processor_context *ctx = (pocl_image_processor_context *) calloc(1,
                                                                                sizeof(pocl_image_processor_context));

    ctx->frame_index_head = 0;
    ctx->frame_index_tail = 0;
    ctx->file_descriptor = fd;
    ctx->lane_count = max_lanes;
    ctx->metadata_array = (image_metadata_t *) calloc(max_lanes, sizeof(image_metadata_t));
    if (sem_init(&ctx->pipe_sem, 0, max_lanes) == -1) {
        LOGE("could not init semaphore\n");
        return -1;
    }
    if (sem_init(&ctx->image_sem, 0, 0) == -1) {
        LOGE("could not init semaphore\n");
        return -1;
    }


    cl_platform_id platform;
    cl_context context;
    cl_device_id devices[MAX_NUM_CL_DEVICES] = {nullptr};
    cl_uint devices_found;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_NUM_CL_DEVICES, devices,
                            &devices_found);
    CHECK_AND_RETURN(status, "getting device id failed");
    LOGI("Platform has %d devices\n", devices_found);
    assert(devices_found > 0);

    // some info
    char result_array[256];
    for (unsigned i = 0; i < devices_found; ++i) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DEVICE_NAME:    %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 256 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DEVICE_VERSION: %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, 256 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DRIVER_VERSION: %s\n", i, result_array);
    }

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    context = clCreateContext(cps, devices_found, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    // create the pipelines
    ctx->pipeline_array = (pipeline_context *) calloc(max_lanes, sizeof(pipeline_context));
    for (int i = 0; i < max_lanes; i++) {
        // NOTE: it is probably possible to put the decompression and dnn kernels on the same
        // device by making the second and third cl_device_id the same. Something to look at in the future.
        status = setup_pipeline_context(&(ctx->pipeline_array[i]), width, height, config_flags,
                                        codec_sources, src_size, context, devices,
                                        devices_found);
        CHECK_AND_RETURN(status, "could not create pipeline context ");
    }

    // create a collection of cl buffers to store results in
    ctx->collected_results = (dnn_results *) calloc(max_lanes, sizeof(dnn_results));
    for (int i = 0; i < max_lanes; i++) {
        ctx->collected_results[i].detection_array = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                                   TOT_OUT_COUNT * sizeof(cl_int),
                                                                   NULL, &status);
        CHECK_AND_RETURN(status, "could not create detection array\n");
        ctx->collected_results[i].segmentation_array = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                                      SEG_OUT_COUNT *
                                                                      sizeof(cl_uchar), NULL,
                                                                      &status);
        CHECK_AND_RETURN(status, "could not create segmentation array\n");

    }

    // create a queue used to receive images
    assert(3 <= devices_found && "expected at least 3 devices, but not the case");
    cl_command_queue_properties cq_properties = CL_QUEUE_PROFILING_ENABLE;
    ctx->read_queue = clCreateCommandQueue(context, devices[LOCAL_DEVICE], cq_properties, &status);
    CHECK_AND_RETURN(status, "creating read queue failed");

    for (unsigned i = 0; i < MAX_NUM_CL_DEVICES; ++i) {
        if (nullptr != devices[i]) {
            clReleaseDevice(devices[i]);
        }
    }

    clReleaseContext(context);

    // setup profiling
    if (ENABLE_PROFILING & config_flags) {
        status = dprintf(fd, CSV_HEADER);
        CHECK_AND_RETURN((status < 0), "could not write csv header");
        std::time_t t = std::time(nullptr);
        status = dprintf(fd, "-1,unix_timestamp,time_s,%ld\n", t);
        CHECK_AND_RETURN((status < 0), "could not write timestamp");
    }


// TODO: create eval pipeline
// TODO: create ping pipeline
// TODO: create tracy cl context


    *ret_ctx = ctx;
    return CL_SUCCESS;
}

/**
 * function to get the last intersection over union of the image processor
 * @param ctx
 * @return
 */
float
get_last_iou(pocl_image_processor_context *ctx) {
    // TODO: actually get the right iou
    return -5.0f;
}

/**
 * release the members of the dnn_results struct
 * @note this does not free the memory of the struct
 * @param results to be released
 * @return opencl status
 */
cl_int
destroy_ddn_results(dnn_results results) {
    COND_REL_MEM(results.segmentation_array);
    COND_REL_MEM(results.detection_array);
    return CL_SUCCESS;
}

/**
 * free everything in the context and set the pointer to NULL
 * @param ctx_ptr address to be freed
 * @return opencl status
 */
int
destroy_pocl_image_processor_context(pocl_image_processor_context **ctx_ptr) {


    pocl_image_processor_context *ctx = *ctx_ptr;

    if (NULL == ctx) {
        return CL_SUCCESS;
    }

    for (int i = 0; i < ctx->lane_count; i++) {
        destroy_ddn_results(ctx->collected_results[i]);
        destroy_pipeline_context(ctx->pipeline_array[i]);
    }
    free(ctx->collected_results);
    free(ctx->pipeline_array);

    free(ctx->metadata_array);
    sem_destroy(&(ctx->pipe_sem));
    sem_destroy(&(ctx->image_sem));
    clReleaseCommandQueue(ctx->read_queue);

    free(ctx);
    *ctx_ptr = NULL;
    return CL_SUCCESS;
}

/**
 * helper function to add nanoseconds to a timespec
 * @param ts pointer to timespec to increment
 * @param increment in nanoseconds
 */
void
add_ns_to_time(struct timespec * const ts, const int increment) {
    ts->tv_nsec += increment;
    // nsec can not be larger than a second
    if (ts->tv_nsec >= 1000000000) {
        ts->tv_sec += 1;
        ts->tv_nsec -= 1000000000;
    }
}

/**
 * function to check that an image can be submitted to the context.
 * this function should be called before submitting an image.
 * @param ctx context to dequeue from
 * @param timeout in miliseconds
 * @return 0 if successful otherwise -1
 */
int
dequeue_spot(pocl_image_processor_context * const ctx, const int timeout) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }

    add_ns_to_time(&ts, timeout * 1000000);

    int ret;
    while ((ret = sem_timedwait(&(ctx->pipe_sem), &ts)) == -1 && errno == EINTR) {
        // continue if interrupted for any reason
        continue;
    }

#ifdef DEBUB_SEMAPHORES
    if (ret == 0) {
        int sem_value;
        sem_getvalue(&(ctx->pipe_sem), &sem_value);
        LOGW("dequeue_spot decremented pipe sem to: %d \n", sem_value);
    }
#endif

    return ret;
}

 /**
  * function to copy raw buffers from the image to a local array and make sure the result is in
  * nv21 format.
  * @param width
  * @param height
  * @param image needs to be a yuv image
  * @param compression_type
  * @param dest_buf
  */
void
copy_yuv_to_arrayV2(const int width, const int height, const image_data_t image,
                    const compression_t compression_type,
                    cl_uchar * const dest_buf) {

    assert(image.type == YUV_DATA_T && "image is not a yuv image");

    // this will be optimized out by the compiler
    const int yrow_stride = image.data.yuv.row_strides[0];
    const uint8_t *y_ptr = image.data.yuv.planes[0];
    const uint8_t *u_ptr = image.data.yuv.planes[1];
    const uint8_t *v_ptr = image.data.yuv.planes[2];
    const int ypixel_stride = image.data.yuv.pixel_strides[0];
    const int upixel_stride = image.data.yuv.pixel_strides[1];
    const int vpixel_stride = image.data.yuv.pixel_strides[2];

    // copy y plane into buffer
    for (int i = 0; i < height; i++) {
        // row_stride is in bytes
        for (int j = 0; j < yrow_stride; j++) {
            dest_buf[i * yrow_stride + j] = y_ptr[(i * yrow_stride + j) * ypixel_stride];
        }
    }

    int uv_start_index = height * yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for (int i = 0; i < (height * width) / 4; i++) {
        if (HEVC_COMPRESSION == compression_type) {
            dest_buf[uv_start_index + 1 + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 2 * i] = u_ptr[i * upixel_stride];
        } else {
            dest_buf[uv_start_index + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 1 + 2 * i] = u_ptr[i * upixel_stride];
        }
    }

}

/**
 * submit an image to the respective pipeline
 * @param ctx pipeline config to run on
 * @param config codec config with runtime options
 * @param image_data object containing image planes
 * @param output dnn_results that can be waited and read from
 * @return opencl return status
 */
cl_int
submit_image_to_pipeline(pipeline_context ctx, const codec_config_t config,
                         const image_data_t image_data, dnn_results *output) {

    FrameMark;
    // TODO: look into if host_ts_ns is still a thing
//    int64_t start_ns = get_timestamp_ns();
    cl_int status;

    compression_t compression_type = config.compression_type;

    // make sure that the input is actually a valid compression type
    assert(CHECK_COMPRESSION_T(compression_type));

    // check that this compression type is enabled
    assert(compression_type & ctx.config_flags);

    // check that no compression is passed to local device
    assert((0 == config.device_type) ? (NO_COMPRESSION == compression_type) : 1);

    // this is done at the beginning so that the quality algorithm has
    // had the option to use the events
    release_events(&ctx.event_array);
    reset_event_array(&ctx.event_array);

    // even though inp_format is assigned pixel_format_enum,
    // the type is set to cl_int since underlying enum types
    // can vary and we want a known size_bytes on both the client and server.
    cl_int inp_format;

    // used to glue stages of the pipeline together
    cl_event dnn_wait_event, dnn_read_event;

    // the default dnn input buffer
    cl_mem dnn_input_buf = NULL;

    // copy the yuv image data over to the host_img_buf and make sure it's semiplanar.
    copy_yuv_to_arrayV2(ctx.width, ctx.height, image_data, compression_type, ctx.host_inp_buf);

    // the local device does not support other compression types, but this function with
    // local devices should only be called with no compression, so other paths will not be
    // reached. There is also an assert to make sure of this.
    if (NO_COMPRESSION == compression_type) {
        // normal execution
        inp_format = YUV_SEMI_PLANAR;

        write_buffer_dnn(ctx.dnn_context, config.device_type, ctx.host_inp_buf,
                         ctx.host_inp_buf_size,
                         ctx.inp_yuv_mem, NULL, &(ctx.event_array), &dnn_wait_event);
        // no compression is an edge case since it uses the uncompressed
        // yuv buffer as input for the dnn stage
        dnn_input_buf = ctx.inp_yuv_mem;

    } else if (YUV_COMPRESSION == compression_type) {
        inp_format = ctx.yuv_context->output_format;

        cl_event wait_on_yuv_event;
        write_buffer_yuv(ctx.yuv_context, ctx.host_inp_buf, ctx.host_inp_buf_size,
                         ctx.inp_yuv_mem, NULL, &(ctx.event_array), &wait_on_yuv_event);

        status = enqueue_yuv_compression(ctx.yuv_context, wait_on_yuv_event,
                                         ctx.inp_yuv_mem, ctx.comp_to_dnn_buf,
                                         &(ctx.event_array), &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue yuv compression work");
        dnn_input_buf = ctx.comp_to_dnn_buf;
    }
#ifndef DISABLE_JPEG
    else if (JPEG_COMPRESSION == compression_type) {
        inp_format = ctx.jpeg_context->output_format;

        ctx.jpeg_context->quality = config.config.jpeg.quality;

        cl_event wait_on_write_event;
        write_buffer_jpeg(ctx.jpeg_context, ctx.host_inp_buf, ctx.host_inp_buf_size,
                          ctx.inp_yuv_mem, &(ctx.event_array), &wait_on_write_event);
        status = enqueue_jpeg_compression(ctx.jpeg_context, wait_on_write_event, ctx.inp_yuv_mem,
                                          ctx.comp_to_dnn_buf, &(ctx.event_array), &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue jpeg compression");
        dnn_input_buf = ctx.comp_to_dnn_buf;
    }
#endif // DISABLE_JPEG
#ifndef DISABLE_HEVC
    // TODO: test hecv compression
    else if (HEVC_COMPRESSION == compression_type) {

        // TODO: refactor hevc to first read image
        assert(0 && "not refactored yet");

        inp_format = ctx.hevc_context->output_format;

        cl_event configure_event = NULL;
        // expensive to configure, only do it if it is actually different than
        // what is currently configured.
        if (ctx.hevc_context->i_frame_interval != config.config.hevc.i_frame_interval ||
            ctx.hevc_context->framerate != config.config.hevc.framerate ||
            ctx.hevc_context->bitrate != config.config.hevc.bitrate ||
            1 != ctx.hevc_context->codec_configured) {

            ctx.hevc_context->codec_configured = 0;
            ctx.hevc_context->i_frame_interval = config.config.hevc.i_frame_interval;
            ctx.hevc_context->framerate = config.config.hevc.framerate;
            ctx.hevc_context->bitrate = config.config.hevc.bitrate;
            configure_hevc_codec(ctx.hevc_context, &(ctx.event_array), &configure_event);
        }

        status = enqueue_hevc_compression(ctx.hevc_context, &(ctx.event_array), &configure_event,
                                          &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue hevc compression");
        dnn_input_buf = ctx.comp_to_dnn_buf;
    } else if (SOFTWARE_HEVC_COMPRESSION == compression_type) {

        // TODO: refactor hev to first read image
        assert(0 && "not refactored yet");

        inp_format = ctx.software_hevc_context->output_format;

        cl_event configure_event = NULL;

        // expensive to configure, only do it if it is actually different than
        // what is currently configured.
        if (ctx.software_hevc_context->i_frame_interval != config.config.hevc.i_frame_interval ||
            ctx.software_hevc_context->framerate != config.config.hevc.framerate ||
            ctx.software_hevc_context->bitrate != config.config.hevc.bitrate ||
            1 != ctx.software_hevc_context->codec_configured) {

            ctx.software_hevc_context->codec_configured = 0;
            ctx.software_hevc_context->i_frame_interval = config.config.hevc.i_frame_interval;
            ctx.software_hevc_context->framerate = config.config.hevc.framerate;
            ctx.software_hevc_context->bitrate = config.config.hevc.bitrate;
            configure_hevc_codec(ctx.software_hevc_context, &(ctx.event_array), &configure_event);
        }

        status = enqueue_hevc_compression(ctx.software_hevc_context, &(ctx.event_array),
                                          &configure_event,
                                          &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue hevc compression");
        dnn_input_buf = ctx.comp_to_dnn_buf;
    }
#endif // DISABLE_HEVC
    else {
        CHECK_AND_RETURN(-1, "jpeg image is not supported with pipelining");
    }

    status = enqueue_dnn(ctx.dnn_context, &dnn_wait_event, config, (pixel_format_enum) inp_format,
                         dnn_input_buf, output->detection_array, output->segmentation_array,
                         &(ctx.event_array), &(output->result_event));
    CHECK_AND_RETURN(status, "could not enqueue dnn stage");

    // todo: preemptively transfer output buffers to the device that will be reading them

    return CL_SUCCESS;
}

/**
 * submit an image to the processor
 * @param ctx
 * @param codec_config runtime options on how to process the image
 * @param image_data image to process
 * @param is_eval_frame indicate that the frame will also be used to
 * evaluate the performance
 * @return opencl status
 */
int
submit_image(pocl_image_processor_context *ctx, codec_config_t codec_config,
             image_data_t image_data, int is_eval_frame) {

    // this function should be called when dequeue_spot acquired a semaphore,
    int index = ctx->frame_index_head % ctx->lane_count;
    ctx->frame_index_head += 1;
    LOGI("submit_image index: %d\n", index);

    // store metadata
    image_metadata_t *image_metadata = &(ctx->metadata_array[index]);
    image_metadata->frame_index = ctx->frame_index_head - 1;
    image_metadata->image_timestamp = image_data.image_timestamp;
    image_metadata->is_eval_frame = is_eval_frame;

    int status;
    status = submit_image_to_pipeline(ctx->pipeline_array[index], codec_config, image_data,
                                      &(ctx->collected_results[index]));

    // increment the semaphore so that image reading thread knows there is an image ready
    sem_post(&(ctx->image_sem));
#ifdef    DEBUB_SEMAPHORES
    int sem_value;
    sem_getvalue(&(ctx->image_sem), &sem_value);
    LOGW("submit_image incremented image sem to: %d \n", sem_value);
#endif

    // TODO: enqueue eval pipeline
    return status;
}

/**
 * wait until an image is done being processed
 * @param ctx to wait on
 * @param timeout in milliseconds
 * @return opencl return status
 */
int
wait_image_available(pocl_image_processor_context *ctx, int timeout) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }

    add_ns_to_time(&ts, timeout * 1000000);

    int ret;
    while ((ret = sem_timedwait(&(ctx->image_sem), &ts)) == -1 && errno == EINTR) {
        // continue if interrupted for any reason
        continue;
    }

#ifdef DEBUB_SEMAPHORES
    if (0 == ret) {
        int sem_value;
        sem_getvalue(&(ctx->image_sem), &sem_value);
        LOGW("wait_image_available decremented image sem to: %d \n", sem_value);
    }
#endif

    return ret;

}

/**
 * read done images from the pipeline
 * @param ctx to read from
 * @param detection_array output with detections
 * @param segmentation_array mask of detections
 * @param eval_metadata evaluation results
 * @param segmentation indicate that segmentation has been done
 * @return opencl status results
 */
int
receive_image(pocl_image_processor_context *ctx, int32_t *detection_array,
              uint8_t *segmentation_array,
              eval_metadata_t *eval_metadata, int *segmentation) {

    int index = ctx->frame_index_tail % ctx->lane_count;
    ctx->frame_index_tail += 1;
    LOGI("receive_image index: %d\n", index);

    dnn_results results = ctx->collected_results[index];
    image_metadata_t image_metadata = ctx->metadata_array[index];

    eval_metadata->event_array = &ctx->pipeline_array[index].event_array;

    int status;
    cl_event read_detect_event, read_segment_event = NULL;
    int wait_event_size = 1;

    status = clEnqueueReadBuffer(ctx->read_queue, results.detection_array, CL_FALSE, 0,
                                 DET_COUNT * sizeof(cl_int), detection_array, 1,
                                 &(results.result_event),
                                 &read_detect_event);
    CHECK_AND_RETURN(status, "could not read detection array");
    append_to_event_array(eval_metadata->event_array, read_detect_event,
                          VAR_NAME(read_detect_event));


    if (image_metadata.segmentation) {
        status = clEnqueueReadBuffer(ctx->read_queue, results.segmentation_array, CL_FALSE, 0,
                                     SEG_OUT_COUNT * sizeof(cl_uchar), segmentation_array,
                                     1, &(results.result_event), &read_segment_event);
        CHECK_AND_RETURN(status, "could not read segmentation array");
        append_to_event_array(eval_metadata->event_array, read_segment_event,
                              VAR_NAME(read_segment_event));
        // increment the size of the wait event list
        wait_event_size = 2;
    }
    // used to send segmentation back to java
    *segmentation = image_metadata.segmentation;

    cl_event wait_events[] = {read_detect_event, read_segment_event};
    // after this wait, the detection and segmentation arrays are valid
    status = clWaitForEvents(wait_event_size, wait_events);
    CHECK_AND_RETURN(status, "could not wait on final event");

    // TODO: populate metadata

    // finally release the semaphore
    sem_post(&(ctx->pipe_sem));

#ifdef DEBUB_SEMAPHORES
    int sem_value;
    sem_getvalue(&(ctx->pipe_sem), &sem_value);
    LOGW("receive_image incremented pipe sem to: %d \n", sem_value);
#endif

    return CL_SUCCESS;
}


#ifdef __cplusplus
}
#endif