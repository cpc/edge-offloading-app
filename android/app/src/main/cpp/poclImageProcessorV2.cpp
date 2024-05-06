//
// Created by rabijl on 27.2.2024.
//

#include <time.h>
#include <errno.h>
#include <assert.h>
#include "poclImageProcessorV2.h"
#include "platform.h"
#include "sharedUtils.h"
#include "poclImageProcessorUtils.h"
#include <ctime>
#include <stdlib.h>
#include <cstring>
#include <pthread.h>
#include "config.h"

#include "Tracy.hpp"
#include "TracyC.h"
#include "TracyOpenCL.hpp"

#define DEBUG_SEMAPHORES

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

    if (JPEG_IMAGE & config_flags) {
        LOGE("poclImageProcessorV2 does not support jpeg_image compression type");
        return -1;
    }

    return 0;
}

#ifdef DEBUG_SEMAPHORES

#define SET_CTX_STATE(ctx, expected_state, new_state) \
        pthread_mutex_lock(&(ctx->state_mut)); \
        if(LANE_ERROR != new_state) {\
            assert(expected_state == ctx->state);\
        }\
        ctx->state = new_state;                       \
                                         \
        LOGW("set state to: %d (ln : %d)\n", ctx->state, __LINE__);\
        pthread_mutex_unlock(&(ctx->state_mut));      \

#else

#define SET_CTX_STATE(ctx, expected_state, new_state) \
        pthread_mutex_lock(&(ctx->state_mut)); \
        ctx->state = new_state;                       \
        pthread_mutex_unlock(&(ctx->state_mut));

#endif

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
                       cl_device_id *devices, cl_uint no_devs, TracyCLCtx *tracy_ctxs,
                       int is_eval) {

    if (supports_config_flags(config_flags) != 0) {
        return -1;
    }
    assert(1 <= no_devs && "setup_pipeline_context requires atleast one device");
    if ((config_flags &
         (YUV_COMPRESSION | HEVC_COMPRESSION | SOFTWARE_HEVC_COMPRESSION | JPEG_COMPRESSION)) &&
        1 == no_devs) {
        CHECK_AND_RETURN(-1, "compression is not supported with one device");
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
                                  src_size, 1);

        CHECK_AND_RETURN(status, "init of codec kernels failed");
    }

#ifndef DISABLE_JPEG
    if ((JPEG_COMPRESSION & ctx->config_flags)) {

        ctx->jpeg_context = create_jpeg_context();
        ctx->jpeg_context->height = height;
        ctx->jpeg_context->width = width;
        ctx->jpeg_context->enc_queue = ctx->enq_queues[0];
        ctx->jpeg_context->dec_queue = ctx->enq_queues[2];
        status = init_jpeg_context(ctx->jpeg_context, cl_ctx, &devices[0], &devices[2], 1, 1);

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

    if (no_devs > 1) {
        ctx->dnn_context->remote_queue = ctx->enq_queues[2];
        ctx->dnn_context->local_queue = ctx->enq_queues[0];
#ifdef TRACY_ENABLE
        ctx->dnn_context->remote_tracy_ctx = tracy_ctxs[2];
        ctx->dnn_context->local_tracy_ctx = tracy_ctxs[0];
#endif
        status = init_dnn_context(ctx->dnn_context, cl_ctx, width, height,
                                  &devices[2], &devices[0], is_eval);
    } else {
        // setup local dnn
        ctx->dnn_context->remote_queue = ctx->enq_queues[0];
        ctx->dnn_context->local_queue = ctx->enq_queues[0];
#ifdef TRACY_ENABLE
        ctx->dnn_context->remote_tracy_ctx = tracy_ctxs[0];
        ctx->dnn_context->local_tracy_ctx = tracy_ctxs[0];
#endif
        status = init_dnn_context(ctx->dnn_context, cl_ctx, width, height,
                                  &devices[0], &devices[0], is_eval);
    }
    CHECK_AND_RETURN(status, "could not init dnn_context");

    ctx->event_array = create_event_array_pointer(MAX_EVENTS);

    ctx->state_mut = PTHREAD_MUTEX_INITIALIZER;
    ctx->state = LANE_READY;

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
#endif
#ifndef DISABLE_JPEG
    destroy_jpeg_context(&ctx.jpeg_context);
#endif
    destroy_yuv_context(&ctx.yuv_context);

    COND_REL_MEM(ctx.inp_yuv_mem);
    COND_REL_MEM(ctx.comp_to_dnn_buf);
    free(ctx.host_inp_buf);
    for (int i = 0; i < ctx.queue_count; i++) {
        COND_REL_QUEUE(ctx.enq_queues[i]);
    }
    free_event_array_pointer(&ctx.event_array);
    pthread_mutex_destroy(&(ctx.state_mut));

    return CL_SUCCESS;
}

/**
 * initialized the eval context
 * @param ctx
 * @param width
 * @param height
 * @param cl_context
 * @param device
 * @param tracy_ctx
 * @return
 */
cl_int
init_eval_ctx(eval_pipeline_context_t *const ctx, int width, int height, cl_context cl_context,
              cl_device_id *device, TracyCLCtx *tracy_ctx) {

    // create eval pipeline
    ctx->eval_pipeline = (pipeline_context *) calloc(1, sizeof(pipeline_context));
    setup_pipeline_context(ctx->eval_pipeline, width, height, ENABLE_PROFILING | NO_COMPRESSION,
                           NULL, 0, cl_context, device, 1, tracy_ctx, 1);
    ctx->eval_results = (dnn_results *) calloc(1, sizeof(dnn_results));

    clock_gettime(CLOCK_MONOTONIC, &(ctx->next_eval_ts));
    // start eval 4 seconds after starting
    ctx->next_eval_ts.tv_sec += 4;

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
    ctx->metadata_array = (eval_metadata_t *) calloc(max_lanes, sizeof(eval_metadata_t));
    if (sem_init(&ctx->pipe_sem, 0, max_lanes) == -1) {
        LOGE("could not init semaphore\n");
        return -1;
    }
    if (sem_init(&ctx->local_sem, 0, 1) == -1) {
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
    ctx->devices_found = devices_found;

    // some info
    char result_array[1024];
    for (unsigned i = 0; i < devices_found; ++i) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DEVICE_NAME:    %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 1024 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DEVICE_VERSION: %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, 1024 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DRIVER_VERSION: %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DEVICE_BUILT_IN_KERNELS, 1024 * sizeof(char), result_array, NULL);
        LOGI("device %d: CL_DRIVER_BUILT_IN_KERNELS: %s\n", i, result_array);
    }

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    context = clCreateContext(cps, devices_found, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

#ifdef TRACY_ENABLE
    ctx->tracy_ctxs = (TracyCLCtx *) calloc(devices_found, sizeof(TracyCLCtx));
    for (int i = 0; i< devices_found; i++) {
        ctx->tracy_ctxs[i] = TracyCLContext(context, devices[i]);
    }
#else
    ctx->tracy_ctxs= NULL;
#endif

    // create the pipelines
    ctx->pipeline_array = (pipeline_context *) calloc(max_lanes, sizeof(pipeline_context));
    for (int i = 0; i < max_lanes; i++) {
        // NOTE: it is probably possible to put the decompression and dnn kernels on the same
        // device by making the second and third cl_device_id the same. Something to look at in the future.
        status = setup_pipeline_context(&(ctx->pipeline_array[i]), width, height, config_flags,
                                        codec_sources, src_size, context, devices,
                                        devices_found, ctx->tracy_ctxs, 0);
        CHECK_AND_RETURN(status, "could not create pipeline context ");

        snprintf(ctx->pipeline_array[i].lane_name, 16, "lane: %d", i);
    }

    // create a collection of cl buffers to store results in
    ctx->collected_results = (dnn_results *) calloc(max_lanes, sizeof(dnn_results));

    // create a queue used to receive images
    assert(3 <= devices_found && "expected at least 3 devices, but not the case");
    cl_command_queue_properties cq_properties = CL_QUEUE_PROFILING_ENABLE;
    ctx->read_queue = clCreateCommandQueue(context, devices[LOCAL_DEVICE], cq_properties, &status);
    CHECK_AND_RETURN(status, "creating read queue failed");

    if (devices_found > 1) {
        ping_fillbuffer_init(&(ctx->ping_context), context);
        ctx->remote_queue = clCreateCommandQueue(context, devices[REMOTE_DEVICE], cq_properties,
                                                 &status);
        CHECK_AND_RETURN(status, "creating remote queue failed");

#ifdef TRACY_ENABLE
        TracyCLCtx * tracy_ctx = &(ctx->tracy_ctxs[REMOTE_DEVICE]);
#else
        TracyCLCtx * tracy_ctx = NULL;
#endif
        init_eval_ctx(&(ctx->eval_ctx), width, height, context, &(devices[REMOTE_DEVICE]),
                      tracy_ctx);
        // TODO: create a thread that waits on the eval queue
        // for now read eval data when reading results
    }

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
    return ctx->eval_ctx.eval_pipeline->dnn_context->iou;
}

cl_int
destroy_eval_context(eval_pipeline_context_t *ctx) {

    destroy_pipeline_context(*(ctx->eval_pipeline));
    free(ctx->eval_pipeline);
    free(ctx->eval_results);
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
        destroy_pipeline_context(ctx->pipeline_array[i]);
    }

    ping_fillbuffer_destroy(&(ctx->ping_context));

    free(ctx->collected_results);
    free(ctx->pipeline_array);

    free(ctx->metadata_array);
    sem_destroy(&(ctx->pipe_sem));
    sem_destroy(&(ctx->local_sem));
    sem_destroy(&(ctx->image_sem));
    clReleaseCommandQueue(ctx->read_queue);
    clReleaseCommandQueue(ctx->remote_queue);

    if (NULL != ctx->tracy_ctxs) {
        for (int i = 0; i < ctx->devices_found; i++) {
            TracyCLDestroy(ctx->tracy_ctxs[i]);
        }
        free(ctx->tracy_ctxs);
    }

    destroy_eval_context(&(ctx->eval_ctx));

    free(ctx);
    *ctx_ptr = NULL;
    return CL_SUCCESS;
}

/**
 * function to check that an image can be submitted to the context.
 * this function should be called before submitting an image.
 * @param ctx context to dequeue from
 * @param timeout in miliseconds
 * @return 0 if successful otherwise -1
 */
int
dequeue_spot(pocl_image_processor_context *const ctx, const int timeout, const device_type_enum dev_type) {
    ZoneScoped;
    FrameMark;

    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }

    add_ns_to_time(&ts, (long)timeout * 1000000);

    int ret;

    if(LOCAL_DEVICE == dev_type) {
        while ((ret = sem_timedwait(&(ctx->local_sem), &ts)) == -1 && errno == EINTR) {
            // continue if interrupted for any reason
        }
        if(ret != 0) {
            return ret;
        }
    }

    while ((ret = sem_timedwait(&(ctx->pipe_sem), &ts)) == -1 && errno == EINTR) {
        // continue if interrupted for any reason
    }

    // the local pipe succeeded, but the global didn't,
    // so give local one back
    if(LOCAL_DEVICE == dev_type && ret != 0) {
        sem_post(&(ctx->local_sem));
    }


#ifdef DEBUG_SEMAPHORES
    if (ret == 0) {
        int sem_value;
        sem_getvalue(&(ctx->pipe_sem), &sem_value);
        LOGW("dequeue_spot decremented pipe sem to: %d \n", sem_value);
        sem_getvalue(&(ctx->local_sem), &sem_value);
        LOGW("dequeue_spot decremented local sem to: %d \n", sem_value);
    }
#endif

    return ret;
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
submit_image_to_pipeline(pipeline_context *ctx, const codec_config_t config,
                         const image_data_t image_data, eval_metadata_t *metadata,
                         const int file_descriptor, dnn_results *output) {
    ZoneScoped;

    TracyCFrameMarkStart(ctx->lane_name);

    SET_CTX_STATE(ctx, LANE_READY, LANE_BUSY)

    if (NULL != metadata) {
        metadata->host_ts_ns.start = get_timestamp_ns();
    }

    cl_int status;

    compression_t compression_type = config.compression_type;

    // make sure that the input is actually a valid compression type
    assert(CHECK_COMPRESSION_T(compression_type));

    // check that this compression type is enabled
    assert(compression_type & ctx->config_flags);

    // check that no compression is passed to local device
    assert((0 == config.device_type) ? (NO_COMPRESSION == compression_type) : 1);

    // this is done at the beginning so that the quality algorithm has
    // had the option to use the events
    release_events(ctx->event_array);
    reset_event_array(ctx->event_array);

    // even though inp_format is assigned pixel_format_enum,
    // the type is set to cl_int since underlying enum types
    // can vary and we want a known size_bytes on both the client and server.
    cl_int inp_format;

    // used to glue stages of the pipeline together
    cl_event dnn_wait_event, dnn_read_event;

    // the default dnn input buffer
    cl_mem dnn_input_buf = NULL;

    if (NULL != metadata) {
        metadata->host_ts_ns.before_enc = get_timestamp_ns();
    }

    // copy the yuv image data over to the host_img_buf and make sure it's semiplanar.
    copy_yuv_to_arrayV2(ctx->width, ctx->height, image_data, compression_type, ctx->host_inp_buf);

    // the local device does not support other compression types, but this function with
    // local devices should only be called with no compression, so other paths will not be
    // reached. There is also an assert to make sure of this.
    if (NO_COMPRESSION == compression_type) {
        // normal execution
        inp_format = YUV_SEMI_PLANAR;

        write_buffer_dnn(ctx->dnn_context, config.device_type, ctx->host_inp_buf,
                         ctx->host_inp_buf_size,
                         ctx->inp_yuv_mem, NULL, ctx->event_array, &dnn_wait_event);
        // no compression is an edge case since it uses the uncompressed
        // yuv buffer as input for the dnn stage
        dnn_input_buf = ctx->inp_yuv_mem;

    } else if (YUV_COMPRESSION == compression_type) {
        inp_format = ctx->yuv_context->output_format;

        cl_event wait_on_yuv_event;
        write_buffer_yuv(ctx->yuv_context, ctx->host_inp_buf, ctx->host_inp_buf_size,
                         ctx->inp_yuv_mem, NULL, ctx->event_array, &wait_on_yuv_event);

        status = enqueue_yuv_compression(ctx->yuv_context, wait_on_yuv_event,
                                         ctx->inp_yuv_mem, ctx->comp_to_dnn_buf,
                                         ctx->event_array, &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue yuv compression work");
        dnn_input_buf = ctx->comp_to_dnn_buf;
    }
#ifndef DISABLE_JPEG
    else if (JPEG_COMPRESSION == compression_type) {
        inp_format = ctx->jpeg_context->output_format;

        ctx->jpeg_context->quality = config.config.jpeg.quality;

        cl_event wait_on_write_event;
        write_buffer_jpeg(ctx->jpeg_context, ctx->host_inp_buf, ctx->host_inp_buf_size,
                          ctx->inp_yuv_mem, ctx->event_array, &wait_on_write_event);
        status = enqueue_jpeg_compression(ctx->jpeg_context, wait_on_write_event, ctx->inp_yuv_mem,
                                          ctx->comp_to_dnn_buf, ctx->event_array, &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue jpeg compression");
        dnn_input_buf = ctx->comp_to_dnn_buf;
    }
#endif // DISABLE_JPEG
#ifndef DISABLE_HEVC
        // TODO: test hecv compression
    else if (HEVC_COMPRESSION == compression_type) {

        // TODO: refactor hevc to first read image
        assert(0 && "not refactored yet");

        inp_format = ctx->hevc_context->output_format;

        cl_event configure_event = NULL;
        // expensive to configure, only do it if it is actually different than
        // what is currently configured.
        if (ctx->hevc_context->i_frame_interval != config.config.hevc.i_frame_interval ||
            ctx->hevc_context->framerate != config.config.hevc.framerate ||
            ctx->hevc_context->bitrate != config.config.hevc.bitrate ||
            1 != ctx->hevc_context->codec_configured) {

            ctx->hevc_context->codec_configured = 0;
            ctx->hevc_context->i_frame_interval = config.config.hevc.i_frame_interval;
            ctx->hevc_context->framerate = config.config.hevc.framerate;
            ctx->hevc_context->bitrate = config.config.hevc.bitrate;
            configure_hevc_codec(ctx->hevc_context, ctx->event_array, &configure_event);
        }

        status = enqueue_hevc_compression(ctx->hevc_context, ctx->event_array, &configure_event,
                                          &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue hevc compression");
        dnn_input_buf = ctx->comp_to_dnn_buf;
    } else if (SOFTWARE_HEVC_COMPRESSION == compression_type) {

        // TODO: refactor hev to first read image
        assert(0 && "not refactored yet");

        inp_format = ctx->software_hevc_context->output_format;

        cl_event configure_event = NULL;

        // expensive to configure, only do it if it is actually different than
        // what is currently configured.
        if (ctx->software_hevc_context->i_frame_interval != config.config.hevc.i_frame_interval ||
            ctx->software_hevc_context->framerate != config.config.hevc.framerate ||
            ctx->software_hevc_context->bitrate != config.config.hevc.bitrate ||
            1 != ctx->software_hevc_context->codec_configured) {

            ctx->software_hevc_context->codec_configured = 0;
            ctx->software_hevc_context->i_frame_interval = config.config.hevc.i_frame_interval;
            ctx->software_hevc_context->framerate = config.config.hevc.framerate;
            ctx->software_hevc_context->bitrate = config.config.hevc.bitrate;
            configure_hevc_codec(ctx->software_hevc_context, ctx->event_array, &configure_event);
        }

        status = enqueue_hevc_compression(ctx->software_hevc_context, ctx->event_array,
                                          &configure_event,
                                          &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue hevc compression");
        dnn_input_buf = ctx->comp_to_dnn_buf;
    }
#endif // DISABLE_HEVC
    else {
        CHECK_AND_RETURN(-1, "jpeg image is not supported with pipelining");
    }

    if (NULL != metadata) {
        metadata->host_ts_ns.before_dnn = get_timestamp_ns();
    }

    status = enqueue_dnn(ctx->dnn_context, &dnn_wait_event, config, (pixel_format_enum) inp_format,
                         dnn_input_buf,
//                         output->detection_array, output->segmentation_array,
                         ctx->event_array, &(output->event_list[0]));
    CHECK_AND_RETURN(status, "could not enqueue dnn stage");

    // TODO: create goto statement putting the state to an error

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
  ZoneScoped;

    int index = ctx->frame_index_head % ctx->lane_count;
    ctx->frame_index_head += 1;

    char *markId = new char[16];
    snprintf(markId, 16, "frame start: %i", ctx->frame_index_head - 1);
    TracyMessage(markId, strlen(markId));
    LOGI("submit_image index: %d\n", index);

    // store metadata
    eval_metadata_t *image_metadata = &(ctx->metadata_array[index]);
    dnn_results *collected_result = &(ctx->collected_results[index]);

    int status;
    status = submit_image_to_pipeline(&(ctx->pipeline_array[index]), codec_config, image_data,
                                      image_metadata, ctx->file_descriptor,
                                      collected_result);
    CHECK_AND_RETURN(status, "could not submit image to pipeline");

    image_metadata->is_eval_frame = 0;

    // check if we need to run the eval kernels
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (compare_timespec(&ts, &(ctx->eval_ctx.next_eval_ts)) > 0 &&
        REMOTE_DEVICE == codec_config.device_type) {
        codec_config_t eval_config = {NO_COMPRESSION, codec_config.device_type,
                                      codec_config.rotation, codec_config.do_segment, NULL};
        status = submit_image_to_pipeline((ctx->eval_ctx.eval_pipeline), eval_config, image_data,
                                          NULL, ctx->file_descriptor, ctx->eval_ctx.eval_results);
        CHECK_AND_RETURN(status, "could not submit eval image");

        cl_event wait_list[] = {collected_result->event_list[0],
                                ctx->eval_ctx.eval_results->event_list[0]};
        status = enqueue_eval_dnn(ctx->eval_ctx.eval_pipeline->dnn_context,
                                  (ctx->pipeline_array[index].dnn_context), &codec_config,
                                  wait_list, 2,
                                  ctx->eval_ctx.eval_pipeline->event_array,
                                  &(ctx->eval_ctx.eval_results->event_list[1]));
        ctx->eval_ctx.eval_results->event_list_size = 2;
        CHECK_AND_RETURN(status, "could not enqueue eval kernel");

        image_metadata->is_eval_frame = 1;

        // increment the time for the next eval to be in 2 seconds
        clock_gettime(CLOCK_MONOTONIC, &(ctx->eval_ctx.next_eval_ts));
        ctx->eval_ctx.next_eval_ts.tv_sec += 2;

    }

    // run the ping buffer
    if (REMOTE_DEVICE == codec_config.device_type) {
        // todo: see if this should be run on every device
        image_metadata->host_ts_ns.before_fill = get_timestamp_ns();
        cl_event fill_event;
        ping_fillbuffer_run(ctx->ping_context, ctx->remote_queue,
                            ctx->pipeline_array[index].event_array, &fill_event);
        collected_result->event_list[1] = fill_event;
        collected_result->event_list_size = 2;
    } else {
        collected_result->event_list[1] = NULL;
        collected_result->event_list_size = 1;
    }

    // populate the metadata
    image_metadata->frame_index = ctx->frame_index_head - 1;
    image_metadata->image_timestamp = image_data.image_timestamp;
    image_metadata->codec = codec_config;
    image_metadata->event_array = ctx->pipeline_array[index].event_array;

    // do some logging
    if (ENABLE_PROFILING & ctx->pipeline_array[index].config_flags) {

        dprintf(ctx->file_descriptor, "%d,frame,timestamp,%ld\n", image_metadata->frame_index,
                image_data.image_timestamp);  // this should match device timestamp in camera log
        dprintf(ctx->file_descriptor, "%d,frame,is_eval,%d\n", image_metadata->frame_index,
                is_eval_frame);
        log_codec_config(ctx->file_descriptor, image_metadata->frame_index, codec_config);
    }

#ifdef DEBUG_SEMAPHORES
    int sem_value;
    sem_getvalue(&(ctx->image_sem), &sem_value);
    LOGW("submit_image incremented image sem to: %d \n", sem_value + 1);
#endif
    // increment the semaphore so that image reading thread knows there is an image ready
    sem_post(&(ctx->image_sem));


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

    ZoneScoped;

    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }

    add_ns_to_time(&ts, timeout * 1000000);

    int ret;
    while ((ret = sem_timedwait(&(ctx->image_sem), &ts)) == -1 && errno == EINTR) {
        // continue if interrupted for any reason
    }

#ifdef DEBUG_SEMAPHORES
    if (0 == ret) {
        int sem_value;
        sem_getvalue(&(ctx->image_sem), &sem_value);
        LOGW("wait_image_available decremented image sem to: %d \n", sem_value);
    }
#endif

    return ret;

}

size_t
get_compression_size(const pipeline_context ctx, const compression_t compression) {

    switch (compression) {

#ifndef  DISABLE_JPEG
        case JPEG_COMPRESSION:
            return get_compression_size_jpeg(ctx.jpeg_context);
#endif
#ifndef DISABLE_HEVC
        case HEVC_COMPRESSION:
            assert(0 && "not implemented yet");
        case SOFTWARE_HEVC_COMPRESSION:
            assert(0 && "not implemented yet");
#endif
        case YUV_COMPRESSION:
            return get_compression_size_yuv(ctx.yuv_context);
        case NO_COMPRESSION:
            return sizeof(cl_uchar) * ctx.height * ctx.width * 3 / 2;
        default:
            LOGE("requested size of unknown compression type\n");
            return 0;
    }
}

/**
 * read done images from the pipeline
 * @param ctx to read from
 * @param detection_array output with detections
 * @param segmentation_array mask of detections
 * @param return_metadata evaluation results
 * @param segmentation indicate that segmentation has been done
 * @return opencl status results
 */
int
receive_image(pocl_image_processor_context *const ctx, int32_t *detection_array,
              uint8_t *segmentation_array,
              eval_metadata_t *return_metadata, int *const segmentation) {
    ZoneScoped;

    int index = ctx->frame_index_tail % ctx->lane_count;
    ctx->frame_index_tail += 1;
    LOGI("receive_image index: %d\n", index);

    dnn_results results = ctx->collected_results[index];
    eval_metadata_t image_metadata = ctx->metadata_array[index];

    pipeline_context *pipeline = &(ctx->pipeline_array[index]);

    int config_flags = pipeline->config_flags;

    // used to send segmentation back to java
    *segmentation = image_metadata.codec.do_segment;

    image_metadata.host_ts_ns.before_wait = get_timestamp_ns();

    int status;

    // TODO: wrap with pipeline function
    status = enqueue_read_results_dnn(pipeline->dnn_context,
                                      &image_metadata.codec, detection_array, segmentation_array,
                                      image_metadata.event_array, results.event_list_size,
                                      results.event_list);
    CHECK_AND_RETURN(status, "could not read results back");


    if (image_metadata.is_eval_frame) {
        ZoneScopedN("wait_eval");
        status = clWaitForEvents(ctx->eval_ctx.eval_results->event_list_size,
                                 ctx->eval_ctx.eval_results->event_list);
        CHECK_AND_RETURN(status, "could not wait on eval event");
    }

    image_metadata.host_ts_ns.after_wait = get_timestamp_ns();

    if (NULL != ctx->tracy_ctxs) {
        for (int i = 0; i < ctx->devices_found; i++) {
            TracyCLCollect(ctx->tracy_ctxs[i]);
        }
    }

    image_metadata.host_ts_ns.stop = get_timestamp_ns();

    if (ENABLE_PROFILING & config_flags) {
        status = print_events(ctx->file_descriptor, image_metadata.frame_index,
                              image_metadata.event_array);
        CHECK_AND_RETURN(status, "failed to print events");

        if (1 == image_metadata.is_eval_frame) {
            status = print_events(ctx->file_descriptor, image_metadata.frame_index,
                                  ctx->eval_ctx.eval_pipeline->event_array);
            CHECK_AND_RETURN(status, "failed to print events");
        }

        uint64_t compressed_size = get_compression_size(*pipeline,
                                                        image_metadata.codec.compression_type);
        dprintf(ctx->file_descriptor, "%d,compression,size_bytes,%lu\n", image_metadata.frame_index,
                compressed_size);

        log_host_ts_ns(ctx->file_descriptor, image_metadata.frame_index,
                       image_metadata.host_ts_ns);
    }

    // todo: currently pass the data back for the eventual quality algo,
    // but the event array might get reset already on the next iteration
    // before being able to be used. so we should probably do the quality
    // algorithm before releasing the semaphore.
    if (NULL != return_metadata) {
        memcpy(return_metadata, &image_metadata, sizeof(eval_metadata_t));
    }

    char *markId = new char[16];
    snprintf(markId, 16, "frame end: %i", ctx->frame_index_tail - 1);
    TracyMessage(markId, strlen(markId));
    TracyCFrameMarkEnd(pipeline->lane_name);

    SET_CTX_STATE(pipeline, LANE_BUSY, LANE_READY)


    if (image_metadata.is_eval_frame) {
        SET_CTX_STATE(ctx->eval_ctx.eval_pipeline, LANE_BUSY, LANE_READY)
    }

#ifdef DEBUG_SEMAPHORES
    {
        int sem_value;
        sem_getvalue(&(ctx->pipe_sem), &sem_value);
        LOGW("receive_image incremented pipe sem to: %d \n", sem_value + 1);
    }
#endif
    // finally release the semaphore
    sem_post(&(ctx->pipe_sem));

    if (LOCAL_DEVICE == image_metadata.codec.device_type) {

#ifdef DEBUG_SEMAPHORES
        {
            int sem_value;
            sem_getvalue(&(ctx->local_sem), &sem_value);
            LOGW("receive_image incremented local sem to: %d \n", sem_value + 1);
        }
#endif
        sem_post(&(ctx->local_sem));
    }
    
    return CL_SUCCESS;
}

#ifdef __cplusplus
}
#endif