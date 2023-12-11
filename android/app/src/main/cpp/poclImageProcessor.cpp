
#define CL_HPP_MINIMUM_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300


#if __ANDROID__
// required for proxy device http://portablecl.org/docs/html/proxy.html
#include <rename_opencl.h>

#endif

#include <CL/cl.h>
#include <CL/cl_ext_pocl.h>
#include <string>
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <unistd.h>

#include "poclImageProcessor.h" // defines LOGTAG
#include "platform.h"
#include "sharedUtils.h"
#include "event_logger.h"
#include "yuv_compression.h"
#include "jpeg_compression.h"
#include "hevc_compression.h"
#include "testapps.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define EVAL_INTERVAL 10
#define VERBOSITY 1

static int detection_count = 1 + MAX_DETECTIONS * 6;
static int segmentation_count = MAX_DETECTIONS * MASK_W * MASK_H;
static int seg_out_count = MASK_W * MASK_H * 4; // RGBA image
static int total_out_count = detection_count + segmentation_count;
static cl_context context = nullptr;

/**
 * all the devices, order:
 * 0: local basic
 * 1: proxy
 * 2: remote basic
 * 3: remote pthread
 */
static cl_command_queue commandQueue[MAX_NUM_CL_DEVICES] = {nullptr};
static cl_program program = nullptr;
static cl_program codec_program = nullptr;
static cl_kernel dnn_kernel = nullptr;
static cl_kernel enc_y_kernel = nullptr;
static cl_kernel enc_uv_kernel = nullptr;
static cl_kernel dec_y_kernel = nullptr;
static cl_kernel dec_uv_kernel = nullptr;
static cl_kernel postprocess_kernel = nullptr;
static cl_kernel reconstruct_kernel = nullptr;

// objects related to quality evaluation
static cl_command_queue eval_command_queue = nullptr;
static cl_kernel eval_kernel = nullptr;
static cl_mem eval_img_buf = nullptr;
static cl_mem eval_out_buf = nullptr;
static cl_mem eval_out_mask_buf = nullptr;
static cl_mem eval_postprocess_buf = nullptr;
static cl_mem eval_iou_buf = nullptr;
static cl_mem tmp_out_buf[MAX_NUM_CL_DEVICES] = {nullptr};
static cl_mem tmp_postprocess_buf[MAX_NUM_CL_DEVICES] = {nullptr};
static cl_event tmp_detect_copy_event, tmp_postprocess_copy_event, eval_read_event;
static int is_eval_running = 0;
event_array_t eval_event_array;

// kernel execution loop related things
static size_t tot_pixels;
static size_t img_buf_size;
static cl_int inp_w;
static cl_int inp_h;
static cl_int rotate_cw_degrees;
//static cl_int inp_format;
//static int compression_type;

//static int compression_flags;
static int config_flags;

/**
 * the contents of the image, non compressed
 */
static cl_mem img_buf[MAX_NUM_CL_DEVICES] = {nullptr};
/**
 * the buffer containing the detections,
 * i.e. the result of calling yolo
 */
static cl_mem out_buf[MAX_NUM_CL_DEVICES] = {nullptr};
/**
 * buffer with segmentation masks.
 * also the result of calling yolo.
 * input for the postprocess kernel
 */
static cl_mem out_mask_buf[MAX_NUM_CL_DEVICES] = {nullptr};
/**
 * buffer with more a more compact representation of the
 * segmentation masks. output of the postprocess kernel.
 */
static cl_mem postprocess_buf[MAX_NUM_CL_DEVICES] = {nullptr};
static cl_mem reconstruct_buf[MAX_NUM_CL_DEVICES] = {nullptr};
static cl_uchar *host_img_buf = nullptr;
static cl_uchar *host_postprocess_buf = nullptr;
static size_t local_size;
static size_t global_size;

// related to compression kernels
static size_t enc_y_global[2], enc_uv_global[2];
static size_t dec_y_global[2], dec_uv_global[2];
static cl_mem out_enc_y_buf = nullptr, out_enc_uv_buf = nullptr;

// make these variables global, just in case
// they don't get set properly during processing.
//int device_index_copy = 0;
//int do_segment_copy = 0;
//int local_do_compression = 0;
//unsigned char enable_profiling = 0;

char *c_log_string = nullptr;
#define DATA_POINT_SIZE 22  // number of decimal digits for 2^64 + ', '
// allows for 9 datapoints plus 3 single digit configs, newline and term char
#define LOG_BUFFER_SIZE (DATA_POINT_SIZE * 9 + 9 +2)
#define MAX_EVENTS 99
int file_descriptor;
int frame_index;
// used to both log and free events
event_array_t event_array;

// variable to check if everything is ready for execution
int setup_success = 0;

yuv_codec_context_t *yuv_context = NULL;
jpeg_codec_context_t *jpeg_context = NULL;
hevc_codec_context_t *hevc_context = NULL;
ping_fillbuffer_context_t *PING_CTX = NULL;

/**
 * setup up everything needed to process jpeg images
 * @param enc_device the device that will encode
 * @param dec_device the device that will decode
 * @param quality the starting quality at which to compress
 * @param enable_resize used to enable the pocl size extension. currently jpeg images from the camera
 * do not work well with this extension
 * @return the status of how the commands executed
 */
static int
init_jpeg_codecs(cl_device_id *enc_device, cl_device_id *dec_device, cl_int quality,
                 int enable_resize) {

    int status;
    cl_program enc_program = clCreateProgramWithBuiltInKernels(context, 1, enc_device,
                                                               "pocl.compress.to.jpeg.yuv420nv21",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create enc program");

    status = clBuildProgram(enc_program, 1, enc_device, nullptr, nullptr, nullptr);
    CHECK_AND_RETURN(status, "could not build enc program");

    cl_program dec_program = clCreateProgramWithBuiltInKernels(context, 1, dec_device,
                                                               "pocl.decompress.from.jpeg.rgb888",
                                                               &status);
    CHECK_AND_RETURN(status, "could not create dec program");

    status = clBuildProgram(dec_program, 1, dec_device, nullptr, nullptr, nullptr);
    CHECK_AND_RETURN(status, "could not build dec program");

    img_buf[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, tot_pixels * 3 / 2, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the input buffer");

    // overprovision this buffer since the compressed output size can vary
    out_enc_y_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels * 3 / 2, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    // needed to indicate how big the compressed image is
    out_enc_uv_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output size buffer");

    // pocl content extension, allows for only the used part of the buffer to be transferred
    // https://registry.khronos.org/OpenCL/extensions/pocl/cl_pocl_content_size.html
    if (enable_resize > 0) {
        status = clSetContentSizeBufferPoCL(out_enc_y_buf, out_enc_uv_buf);
        CHECK_AND_RETURN(status, "could not apply content size extension");
    }

    enc_y_kernel = clCreateKernel(enc_program, "pocl.compress.to.jpeg.yuv420nv21", &status);
    CHECK_AND_RETURN(status, "failed to create enc kernel");

    dec_y_kernel = clCreateKernel(dec_program, "pocl.decompress.from.jpeg.rgb888", &status);
    CHECK_AND_RETURN(status, "failed to create dec kernel");

    status = clSetKernelArg(enc_y_kernel, 0, sizeof(cl_mem), &img_buf[1]);
    status |= clSetKernelArg(enc_y_kernel, 1, sizeof(cl_int), &inp_w);
    status |= clSetKernelArg(enc_y_kernel, 2, sizeof(cl_int), &inp_h);
    status |= clSetKernelArg(enc_y_kernel, 3, sizeof(cl_int), &quality);
    status |= clSetKernelArg(enc_y_kernel, 4, sizeof(cl_mem), &out_enc_y_buf);
    status |= clSetKernelArg(enc_y_kernel, 5, sizeof(cl_mem), &out_enc_uv_buf);
    CHECK_AND_RETURN(status, "failed to assign kernel parameters to  enc kernel");

    status = clSetKernelArg(dec_y_kernel, 0, sizeof(cl_mem), &out_enc_y_buf);
    status |= clSetKernelArg(dec_y_kernel, 1, sizeof(cl_mem), &out_enc_uv_buf);
    status |= clSetKernelArg(dec_y_kernel, 2, sizeof(cl_int), &inp_w);
    status |= clSetKernelArg(dec_y_kernel, 3, sizeof(cl_int), &inp_h);
    status |= clSetKernelArg(dec_y_kernel, 4, sizeof(cl_mem), &img_buf[2]);
    CHECK_AND_RETURN(status, "failed to assign kernel parameters to  dec kernel");

    // built-in kernels, so one dimensional
    enc_y_global[0] = 1;

    dec_y_global[0] = 1;

    clReleaseProgram(enc_program);
    clReleaseProgram(dec_program);

    return 0;

}

/**
 * setup and create the objects needed for repeated PoCL calls.
 * PoCL can be configured by setting environment variables.
 * @param width of the image
 * @param height of the image
 * @param enableProfiling enable
 * @param codec_sources the kernel sources of the codecs
 * @param src_size the size of the codec_sources
 * @return
 */
int
initPoclImageProcessor(const int width, const int height, const int init_config_flags,
                       const char *codec_sources, const size_t src_size,
                       const int fd, event_array_t **return_array,
                       event_array_t **return_eval_array) {
    cl_platform_id platform;
    cl_device_id devices[MAX_NUM_CL_DEVICES] = {nullptr};
    cl_uint devices_found;
    cl_int status;

    // This is kept as a regression test
    status = test_vec_add();
    CHECK_AND_RETURN(status, "TMP Failed running test vector addition.\n");


    config_flags = init_config_flags;

    // initial quality
    int quality = 80;

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

    cl_command_queue_properties cq_properties = 0;
    if (ENABLE_PROFILING & config_flags) {
        LOGI("enabling profiling\n");
        cq_properties = CL_QUEUE_PROFILING_ENABLE;
    }

    for (unsigned i = 0; i < devices_found; ++i) {
        commandQueue[i] = clCreateCommandQueue(context, devices[i], cq_properties, &status);
        CHECK_AND_RETURN(status, "creating command queue failed");
    }

    // Only create builtin kernel for basic and remote devices
    std::string dnn_kernel_name = "pocl.dnn.detection.u8";
    std::string postprocess_kernel_name = "pocl.dnn.segmentation.postprocess.u8";
    std::string reconstruct_kernel_name = "pocl.dnn.segmentation.reconstruct.u8";
    std::string eval_kernel_name = "pocl.dnn.eval.iou.f32";
    std::string kernel_names =
            dnn_kernel_name + ";" + postprocess_kernel_name + ";" + reconstruct_kernel_name + ";" +
            eval_kernel_name;

    int dnn_device_idx = 0;

    if (1 == devices_found) {
        program = clCreateProgramWithBuiltInKernels(context, 1, devices, kernel_names
                .c_str(), &status);
        CHECK_AND_RETURN(status, "creation of program failed");

        status = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);
        CHECK_AND_RETURN(status, "building of program failed");

    } else {
        dnn_device_idx = 2;
        // create kernels needed for remote execution
        cl_device_id inference_devices[] = {devices[0], devices[2]};

        program = clCreateProgramWithBuiltInKernels(context, 2, inference_devices,
                                                    kernel_names.c_str(), &status);
        CHECK_AND_RETURN(status, "creation of program failed");

        status = clBuildProgram(program, 2, inference_devices, nullptr, nullptr, nullptr);
        CHECK_AND_RETURN(status, "building of program failed");
    }

    eval_command_queue = clCreateCommandQueue(context, devices[dnn_device_idx], cq_properties,
                                              &status);
    CHECK_AND_RETURN(status, "creating eval command queue failed");


    eval_kernel = clCreateKernel(program, eval_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating eval kernel failed");

    eval_img_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, total_out_count * sizeof(cl_int),
                                  NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval image buffer");

    eval_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                  NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval output buffer");

    eval_out_mask_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       MASK_W * MASK_H * MAX_DETECTIONS * sizeof(cl_char),
                                       NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval segmentation mask buffer");

    eval_postprocess_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          MASK_W * MASK_H * sizeof(cl_uchar), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create eval segmentation postprocessing buffer");

    eval_iou_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_float), NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the eval iou buffer");

    dnn_kernel = clCreateKernel(program, dnn_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    postprocess_kernel = clCreateKernel(program, postprocess_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    reconstruct_kernel = clCreateKernel(program, reconstruct_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating reconstruct kernel failed");

    // set some default values;
    rotate_cw_degrees = 90;

    inp_w = width;
    inp_h = height;
    tot_pixels = inp_w * inp_h;
    // yuv420 so 6 bytes for every 4 pixels
    img_buf_size = (tot_pixels * 3) / 2;
    img_buf[0] = clCreateBuffer(context, (CL_MEM_READ_WRITE),
                                img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the image buffer");

    out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, total_out_count * sizeof(cl_int),
                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    tmp_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, total_out_count * sizeof(cl_int),
                                    NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the reference output buffer");

    out_mask_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     MASK_W * MASK_H * MAX_DETECTIONS * sizeof(cl_char),
                                     NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    postprocess_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        MASK_W * MASK_H * sizeof(cl_uchar),
                                        NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the tmp segmentation postprocessing buffer");

    tmp_postprocess_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            MASK_W * MASK_H * sizeof(cl_uchar),
                                            NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

    reconstruct_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        seg_out_count * sizeof(cl_uchar),
                                        NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation reconstructed buffer");

    // only allocate these buffers if the remote is also available
    if (devices_found > 1) {
        // remote device buffers
        if ((JPEG_COMPRESSION & config_flags) || (JPEG_IMAGE & config_flags)) {
            img_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        tot_pixels * 3, NULL, &status);
        } else {
            img_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        img_buf_size, NULL, &status);
        }
        CHECK_AND_RETURN(status, "failed to create the image buffer");

        out_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, total_out_count * sizeof(cl_int),
                                    NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the output buffer");

        tmp_out_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        total_out_count * sizeof(cl_int),
                                        NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the reference output buffer");

        out_mask_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels * MAX_DETECTIONS *
                                                                     sizeof(cl_char) / 4,
                                         NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

        postprocess_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            MASK_W * MASK_H * sizeof(cl_uchar),
                                            NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the segmentation postprocessing buffer");

        tmp_postprocess_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                MASK_W * MASK_H * sizeof(cl_uchar),
                                                NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the tmp segmentation postprocessing buffer");

        reconstruct_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            seg_out_count * sizeof(cl_uchar),
                                            NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the segmentation reconstructed buffer");
    }

    //status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &img_buf);
    status = clSetKernelArg(dnn_kernel, 1, sizeof(cl_int), &inp_w);
    status |= clSetKernelArg(dnn_kernel, 2, sizeof(cl_int), &inp_h);
    status |=
            clSetKernelArg(dnn_kernel, 3, sizeof(cl_int), &rotate_cw_degrees);
//    status |= clSetKernelArg(dnn_kernel, 4, sizeof(cl_int), &inp_format);
    //status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &out_buf);
    //status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &out_mask_buf);

    //status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &out_buf);
    //status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem), &out_mask_buf);
    //status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem), &postprocess_buf);
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    global_size = 1;
    local_size = 1;

    host_img_buf = (cl_uchar *) malloc(img_buf_size);
    host_postprocess_buf = (cl_uchar *) malloc(MASK_W * MASK_H * sizeof(cl_uchar));

    // string to write values to for logging
    c_log_string = (char *) malloc(LOG_BUFFER_SIZE * sizeof(char));

    if (devices_found > 1) {

        if (YUV_COMPRESSION & config_flags) {

            // for some reason the basic device fails to build the program
            // when building from source, therefore use proxy device (which is the mobile gpu)
            yuv_context = create_yuv_context();
            yuv_context->height = inp_h;
            yuv_context->width = inp_w;
            yuv_context->img_buf_size = img_buf_size;
            yuv_context->enc_queue = commandQueue[1];
            yuv_context->dec_queue = commandQueue[3];
            yuv_context->host_img_buf = host_img_buf;
            yuv_context->host_postprocess_buf = host_postprocess_buf;
            status = init_yuv_context(yuv_context, context, devices[1], devices[3], codec_sources,
                                      src_size);

            CHECK_AND_RETURN(status, "init of codec kernels failed");
        }
        if ((JPEG_COMPRESSION & config_flags)) {

            jpeg_context = create_jpeg_context();
            jpeg_context->height = inp_h;
            jpeg_context->width = inp_w;
            jpeg_context->img_buf_size = img_buf_size;
            jpeg_context->enc_queue = commandQueue[0];
            jpeg_context->dec_queue = commandQueue[3];
            jpeg_context->host_img_buf = host_img_buf;
            jpeg_context->host_postprocess_buf = host_postprocess_buf;
            status = init_jpeg_context(jpeg_context, context, &devices[0], &devices[3], 1);

            CHECK_AND_RETURN(status, "init of codec kernels failed");
        }
        if (JPEG_IMAGE & config_flags) {
            status = init_jpeg_codecs(&devices[0], &devices[3], quality, 0);
            CHECK_AND_RETURN(status, "init of codec kernels failed");
        }
        if (HEVC_COMPRESSION & config_flags) {

            hevc_context = create_hevc_context();
            hevc_context->height = inp_h;
            hevc_context->width = inp_w;
            hevc_context->img_buf_size = img_buf_size;
            hevc_context->enc_queue = commandQueue[0];
            hevc_context->dec_queue = commandQueue[3];
            hevc_context->host_img_buf = host_img_buf;
            hevc_context->host_postprocess_buf = host_postprocess_buf;
            status = init_hevc_context(hevc_context, context, &devices[0], &devices[3], 1);

            CHECK_AND_RETURN(status, "init of hevc codec kernels failed");
        }

    }

    for (unsigned i = 0; i < MAX_NUM_CL_DEVICES; ++i) {
        if (nullptr != devices[i]) {
            clReleaseDevice(devices[i]);
        }
    }

    // setup profiling
    if (ENABLE_PROFILING & config_flags) {
        file_descriptor = fd;
        status = dprintf(file_descriptor, CSV_HEADER);
        CHECK_AND_RETURN((status < 0), "could not write csv header");
        std::time_t t = std::time(nullptr);
        status = dprintf(file_descriptor, "-1,unix_timestamp,time_s,%ld\n", t);
        CHECK_AND_RETURN((status < 0), "could not write timestamp");
    }
    frame_index = 0;
    event_array = create_event_array(MAX_EVENTS);
    eval_event_array = create_event_array(MAX_EVENTS);
    // send this back to the quality algorithm
    *return_array = &event_array;
    *return_eval_array = &eval_event_array;

    // setup ping monitor
    status = ping_fillbuffer_init(&PING_CTX, context);
    CHECK_AND_RETURN(status, "PING failed to init");

    setup_success = 1;

    return 0;
}

/**
 * free all global variables
 * @return
 */
int
destroy_pocl_image_processor() {

    setup_success = 0;
    for (unsigned i = 0; i < MAX_NUM_CL_DEVICES; ++i) {
        if (commandQueue[i] != nullptr) {
            clReleaseCommandQueue(commandQueue[i]);
            commandQueue[i] = nullptr;
        }

        if (nullptr != img_buf[i]) {
            clReleaseMemObject(img_buf[i]);
            img_buf[i] = nullptr;
        }

        if (nullptr != out_buf[i]) {
            clReleaseMemObject(out_buf[i]);
            out_buf[i] = nullptr;
        }

        if (nullptr != tmp_out_buf[i]) {
            clReleaseMemObject(tmp_out_buf[i]);
            tmp_out_buf[i] = nullptr;
        }

        if (nullptr != out_mask_buf[i]) {
            clReleaseMemObject(out_mask_buf[i]);
            out_mask_buf[i] = nullptr;
        }

        if (nullptr != postprocess_buf[i]) {
            clReleaseMemObject(postprocess_buf[i]);
            postprocess_buf[i] = nullptr;
        }

        if (nullptr != tmp_postprocess_buf[i]) {
            clReleaseMemObject(tmp_postprocess_buf[i]);
            tmp_postprocess_buf[i] = nullptr;
        }

        if (nullptr != reconstruct_buf[i]) {
            clReleaseMemObject(reconstruct_buf[i]);
            reconstruct_buf[i] = nullptr;
        }
    }

    if (eval_command_queue != nullptr) {
        clReleaseCommandQueue(eval_command_queue);
        eval_command_queue = nullptr;
    }

    if (eval_iou_buf != nullptr) {
        clReleaseMemObject(eval_iou_buf);
        eval_iou_buf = nullptr;
    }

    if (eval_out_buf != nullptr) {
        clReleaseMemObject(eval_out_buf);
        eval_out_buf = nullptr;
    }

    if (eval_out_mask_buf != nullptr) {
        clReleaseMemObject(eval_out_mask_buf);
        eval_out_mask_buf = nullptr;
    }

    if (eval_postprocess_buf != nullptr) {
        clReleaseMemObject(eval_postprocess_buf);
        eval_postprocess_buf = nullptr;
    }

    if (nullptr != out_enc_y_buf) {
        clReleaseMemObject(out_enc_y_buf);
        out_enc_y_buf = nullptr;
    }

    if (nullptr != out_enc_uv_buf) {
        clReleaseMemObject(out_enc_uv_buf);
        out_enc_uv_buf = nullptr;
    }

    if (context != nullptr) {
        clReleaseContext(context);
        context = nullptr;
    }

    if (dnn_kernel != nullptr) {
        clReleaseKernel(dnn_kernel);
        dnn_kernel = nullptr;
    }

    if (nullptr != enc_y_kernel) {
        clReleaseKernel(enc_y_kernel);
        enc_y_kernel = nullptr;
    }

    if (nullptr != enc_uv_kernel) {
        clReleaseKernel(enc_uv_kernel);
        enc_uv_kernel = nullptr;
    }

    if (nullptr != dec_y_kernel) {
        clReleaseKernel(dec_y_kernel);
        dec_y_kernel = nullptr;
    }

    if (nullptr != dec_uv_kernel) {
        clReleaseKernel(dec_uv_kernel);
        dec_uv_kernel = nullptr;
    }

    if (postprocess_kernel != nullptr) {
        clReleaseKernel(postprocess_kernel);
        postprocess_kernel = nullptr;
    }

    if (reconstruct_kernel != nullptr) {
        clReleaseKernel(reconstruct_kernel);
        reconstruct_kernel = nullptr;
    }

    if (eval_kernel != nullptr) {
        clReleaseKernel(eval_kernel);
        eval_kernel = nullptr;
    }

    if (program != nullptr) {
        clReleaseProgram(program);
        program = nullptr;
    }

    if (nullptr != host_img_buf) {
        free(host_img_buf);
        host_img_buf = nullptr;
    }

    if (nullptr != host_postprocess_buf) {
        free(host_postprocess_buf);
        host_postprocess_buf = nullptr;
    }

    if (nullptr != c_log_string) {
        free(c_log_string);
        c_log_string = nullptr;
    }

    free_event_array(&event_array);
    free_event_array(&eval_event_array);

    destory_yuv_context(&yuv_context);
    destory_jpeg_context(&jpeg_context);
    destory_hevc_context(&hevc_context);

    ping_fillbuffer_destroy(&PING_CTX);

    return 0;
}

#if defined(PRINT_PROFILE_TIME)
void printTime(timespec start, timespec stop, const char message[256]) {

    time_t s = stop.tv_sec - start.tv_sec;
    unsigned long final = s * 1000000000;
    final += stop.tv_nsec;
    final -= start.tv_nsec;
    unsigned secs, nsecs;
    int ms = final / 1000000;
    int ns = final % 1000000;

    // char display_message[256 + 16];
    // strcpy(display_message, message);
    // strcat(display_message, ": %d ms, %d ns");
    LOGW("timing %s: %d ms, %d ns\n", message, ms, ns);
}
#endif

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
enqueue_dnn(const cl_event *wait_event, int dnn_device_idx, int out_device_idx,
            const int do_segment, cl_int *detection_array, cl_uchar *segmentation_array,
            cl_int inp_format, int save_out_buf,
            cl_event *out_tmp_detect_copy_event, cl_event *out_tmp_postprocess_copy_event,
            cl_event *out_event, cl_mem inp_buf) {

    cl_int status;

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

    // This is to prevent having two buffers accidentally due to index errors.
    cl_mem _out_buf = out_buf[dnn_device_idx];
    cl_mem _out_mask_buf = out_mask_buf[dnn_device_idx];
    cl_mem _postprocess_buf = postprocess_buf[dnn_device_idx];
    cl_mem _reconstruct_buf = reconstruct_buf[dnn_device_idx];

    status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &inp_buf);
    status |= clSetKernelArg(dnn_kernel, 4, sizeof(cl_int), &inp_format);
    status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &_out_buf);
    status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &_out_mask_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &_out_buf);
    status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem),
                             &_out_mask_buf);
    status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem),
                             &_postprocess_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

    status = clSetKernelArg(reconstruct_kernel, 0, sizeof(cl_mem),
                            &_postprocess_buf);
    status |= clSetKernelArg(reconstruct_kernel, 1, sizeof(cl_mem),
                             &_reconstruct_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to reconstruct kernel");

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "assigning kernel params");
#endif

    cl_event run_dnn_event, read_detect_event;

    status = clEnqueueNDRangeKernel(commandQueue[dnn_device_idx], dnn_kernel, 1, NULL,
                                    &global_size, &local_size, 1,
                                    wait_event, &run_dnn_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
    append_to_event_array(&event_array, run_dnn_event, VAR_NAME(dnn_event));


    status = clEnqueueReadBuffer(commandQueue[out_device_idx], _out_buf,
                                 CL_FALSE, 0,
                                 detection_count * sizeof(cl_int), detection_array, 1,
                                 &run_dnn_event, &read_detect_event);
    CHECK_AND_RETURN(status, "failed to read detection result buffer");
    append_to_event_array(&event_array, read_detect_event, VAR_NAME(read_detect_event));

    if (save_out_buf) {
        // Save detections to a temporary buffer for the quality evaluation pipeline
        size_t sz = detection_count * sizeof(cl_int);

        cl_event tmp_copy_event;
        status = clEnqueueCopyBuffer(commandQueue[dnn_device_idx], _out_buf,
                                     tmp_out_buf[dnn_device_idx],
                                     0, 0, sz, 1, &run_dnn_event, &tmp_copy_event);
        CHECK_AND_RETURN(status, "failed to copy result buffer");
        append_to_event_array(&eval_event_array, tmp_copy_event, VAR_NAME(tmp_copy_event));

        *out_tmp_detect_copy_event = tmp_copy_event;
    }

    if (do_segment) {
        cl_event run_postprocess_event, mig_seg_event, run_reconstruct_event, read_segment_event;

        // postprocess
        status = clEnqueueNDRangeKernel(commandQueue[dnn_device_idx], postprocess_kernel, 1,
                                        NULL,
                                        &global_size, &local_size, 1,
                                        &run_dnn_event, &run_postprocess_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");
        append_to_event_array(&event_array, run_postprocess_event, VAR_NAME(postprocess_event));


        if (save_out_buf) {
            // save postprocessed buffer for the quality eval pipeline
            size_t sz = MASK_W * MASK_H * sizeof(cl_uchar);

            cl_event tmp_copy_event;
            status = clEnqueueCopyBuffer(commandQueue[dnn_device_idx], _postprocess_buf,
                                         tmp_postprocess_buf[dnn_device_idx],
                                         0, 0, sz, 1, &run_postprocess_event, &tmp_copy_event);
            CHECK_AND_RETURN(status, "failed to copy result buffer");
            append_to_event_array(&eval_event_array, tmp_copy_event, VAR_NAME(tmp_copy_event));

            *out_tmp_postprocess_copy_event = tmp_copy_event;
        }

        // move postprocessed segmentation data to host device
        status = clEnqueueMigrateMemObjects(commandQueue[out_device_idx], 1,
                                            &_postprocess_buf, 0, 1,
                                            &run_postprocess_event,
                                            &mig_seg_event);
        CHECK_AND_RETURN(status, "failed to enqueue migration of postprocess buffer");
        append_to_event_array(&event_array, mig_seg_event, VAR_NAME(mig_seg_event));

        // reconstruct postprocessed data to RGBA segmentation mask
        status = clEnqueueNDRangeKernel(commandQueue[out_device_idx], reconstruct_kernel, 1,
                                        NULL,
                                        &global_size, &local_size, 1,
                                        &mig_seg_event, &run_reconstruct_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range reconstruct kernel");
        append_to_event_array(&event_array, run_reconstruct_event, VAR_NAME(reconstruct_event));

        // write RGBA segmentation mask to the result array
        status = clEnqueueReadBuffer(commandQueue[out_device_idx],
                                     _reconstruct_buf, CL_FALSE, 0,
                                     seg_out_count * sizeof(cl_uchar), segmentation_array, 1,
                                     &run_reconstruct_event, &read_segment_event);
        CHECK_AND_RETURN(status, "failed to read segmentation reconstruct result buffer");
        append_to_event_array(&event_array, read_segment_event, VAR_NAME(read_segment_event));

        *out_event = read_segment_event;
    } else {
        *out_event = read_detect_event;
    }

    return 0;
}

cl_int
enqueue_dnn_eval(const cl_event *wait_tmp_detect_copy_event,
                 const cl_event *wait_tmp_postprocess_copy_event,
                 int dnn_device_idx, int do_segment,
                 cl_float *out_iou, cl_event *result_event) {
    cl_int status;

    cl_int inp_format = YUV_SEMI_PLANAR;

    status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &eval_img_buf);
    status |= clSetKernelArg(dnn_kernel, 4, sizeof(cl_int), &inp_format);
    status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &eval_out_buf);
    status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &eval_out_mask_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to eval DNN kernel");

    status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &eval_out_buf);
    status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem), &eval_out_mask_buf);
    status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem), &eval_postprocess_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to eval postprocess kernel");

    status = clSetKernelArg(eval_kernel, 0, sizeof(cl_mem), &tmp_out_buf[dnn_device_idx]);
    status |= clSetKernelArg(eval_kernel, 1, sizeof(cl_mem), &tmp_postprocess_buf[dnn_device_idx]);
    status |= clSetKernelArg(eval_kernel, 2, sizeof(cl_mem), &eval_out_buf);
    status |= clSetKernelArg(eval_kernel, 3, sizeof(cl_mem), &eval_postprocess_buf);
    status |= clSetKernelArg(eval_kernel, 4, sizeof(cl_int), &do_segment);
    status |= clSetKernelArg(eval_kernel, 5, sizeof(cl_mem), &eval_iou_buf);
    CHECK_AND_RETURN(status, "could not assign buffers to eval kernel");

    cl_event eval_write_img_event, eval_dnn_event, eval_postprocess_event, eval_iou_event, eval_read_iou_event;

    status = clEnqueueWriteBuffer(eval_command_queue, eval_img_buf, CL_FALSE, 0,
                                  img_buf_size, host_img_buf, 0, NULL, &eval_write_img_event);
    CHECK_AND_RETURN(status, "failed to write eval img buffer");
    append_to_event_array(&eval_event_array, eval_write_img_event, VAR_NAME(eval_write_img_event));

    status = clEnqueueNDRangeKernel(eval_command_queue, dnn_kernel, 1, NULL, &global_size,
                                    &local_size, 1, &eval_write_img_event, &eval_dnn_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range eval DNN kernel");
    append_to_event_array(&eval_event_array, eval_dnn_event, VAR_NAME(eval_dnn_event));

    cl_event wait_event = eval_dnn_event;
    int num_wait_events = 2;

    if (do_segment) {
        status = clEnqueueNDRangeKernel(eval_command_queue, postprocess_kernel, 1, NULL,
                                        &global_size, &local_size, 1, &eval_dnn_event,
                                        &eval_postprocess_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range eval postprocess kernel");
        append_to_event_array(&eval_event_array, eval_postprocess_event,
                              VAR_NAME(eval_postprocess_event));

        wait_event = eval_postprocess_event;
        // wait for the last event only when segmentation is used
        num_wait_events = 3;
    }

    cl_event wait_events[] = {wait_event, *wait_tmp_detect_copy_event, *wait_tmp_postprocess_copy_event };
    status = clEnqueueNDRangeKernel(eval_command_queue, eval_kernel, 1, NULL, &global_size,
                                    &local_size, num_wait_events, wait_events, &eval_iou_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range eval kernel");
    append_to_event_array(&eval_event_array, eval_iou_event, VAR_NAME(eval_iou_event));

    status = clEnqueueReadBuffer(eval_command_queue, eval_iou_buf, CL_FALSE, 0,
                                 1 * sizeof(cl_float), out_iou, 1, &eval_iou_event,
                                 &eval_read_iou_event);
    CHECK_AND_RETURN(status, "failed to read eval iou result buffer");
    append_to_event_array(&eval_event_array, eval_read_iou_event, VAR_NAME(eval_read_iou_event));

    *result_event = eval_read_iou_event;

    return 0;
}


/**
 * function to copy raw buffers from the image to a local array and make sure the result is in
 * nv21 format.
 */
void
copy_yuv_to_array(const image_data_t image, const compression_t compression_type,
                  cl_uchar *dest_buf) {

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

    // this will be optimized out by the compiler
    const int yrow_stride = image.data.yuv.row_strides[0];
    const uint8_t *y_ptr = image.data.yuv.planes[0];
    const uint8_t *u_ptr = image.data.yuv.planes[1];
    const uint8_t *v_ptr = image.data.yuv.planes[2];
    const int ypixel_stride = image.data.yuv.pixel_strides[0];
    const int upixel_stride = image.data.yuv.pixel_strides[1];
    const int vpixel_stride = image.data.yuv.pixel_strides[2];

    // copy y plane into buffer
    for (int i = 0; i < inp_h; i++) {
        // row_stride is in bytes
        for (int j = 0; j < yrow_stride; j++) {
            dest_buf[i * yrow_stride + j] = y_ptr[(i * yrow_stride + j) * ypixel_stride];
        }
    }

    int uv_start_index = inp_h * yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for (int i = 0; i < (inp_h * inp_w) / 4; i++) {
        if (HEVC_COMPRESSION == compression_type) {
            dest_buf[uv_start_index + 1 + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 2 * i] = u_ptr[i * upixel_stride];
        } else {
            dest_buf[uv_start_index + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 1 + 2 * i] = u_ptr[i * upixel_stride];
        }
    }

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "copying image to array");
#endif

}

/**
 * directly enqueue jpegs to the remote device
 * @param input_img pointer to jpeg buffer
 * @param buf_size size of jpeg buffer
 * @param dec_index device docoding jpeg
 * @param result_event event that can be waited on
 * @return cl_success or an error
 */
cl_int
enqueue_jpeg_image(const cl_uchar *input_img, const uint64_t *buf_size, const int dec_index,
                   cl_event *result_event) {

    cl_int status;
    cl_event image_write_event, image_size_write_event, dec_event;

    status = clEnqueueWriteBuffer(commandQueue[dec_index], out_enc_uv_buf, CL_FALSE, 0, sizeof
            (uint64_t), buf_size, 0, NULL, &image_size_write_event);
    CHECK_AND_RETURN(status, "failed to write image size");
    append_to_event_array(&event_array, image_size_write_event, VAR_NAME(image_size_write_event));

    status = clEnqueueWriteBuffer(commandQueue[dec_index], out_enc_y_buf, CL_FALSE, 0, *buf_size,
                                  input_img, 1, &image_size_write_event, &image_write_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffer");
    append_to_event_array(&event_array, image_write_event, VAR_NAME(image_write_event));

    status = clEnqueueNDRangeKernel(commandQueue[dec_index], dec_y_kernel, 1, NULL, dec_y_global,
                                    NULL, 1, &image_write_event, &dec_event);
    CHECK_AND_RETURN(status, "failed to enqueue decompression kernel");
    append_to_event_array(&event_array, dec_event, VAR_NAME(dec_index));

    *result_event = dec_event;

    return 0;
}


/**
 * process the image with PoCL.
 * assumes that image format is YUV420_888
 * @param device_index
 * @param do_segment
 * @param do_compression
 * @param rotation
 * @param y_ptr
 * @param yrow_stride
 * @param ypixel_stride
 * @param u_ptr
 * @param v_ptr
 * @param uvrow_stride
 * @param uvpixel_stride
 * @param detection_array
 * @param segmentation_array
 * @return
 */
int
poclProcessImage(const int device_index, const int do_segment,
                 const compression_t compressionType,
                 const int quality, const int rotation, int32_t *detection_array,
                 uint8_t *segmentation_array, image_data_t image_data, long image_timestamp,
                 float *iou) {

    struct timespec timespec_start, timespec_stop;
    clock_gettime(CLOCK_MONOTONIC, &timespec_start);
    cl_int status;

    int is_eval_frame = 0;
    if (frame_index % EVAL_INTERVAL == 1) {
        is_eval_frame = 1;
    }

    if (!setup_success) {
        LOGE("poclProcessImage called but setup did not complete successfully\n");
        return 1;
    }
    const compression_t compression_type = compressionType;
    // make sure that the input is actually a valid compression type
    assert(CHECK_COMPRESSION_T(compression_type));

    // check that this compression type is enabled
    assert(compression_type & config_flags);

    // check that no compression is passed to local device
    assert((0 == device_index) ? (NO_COMPRESSION == compression_type) : 1);

    // make local copies so that they don't change during execution
    // when the user presses a button.
    const int device_index_copy = device_index;
    const int do_segment_copy = do_segment;

    if (rotation != rotate_cw_degrees) {
        rotate_cw_degrees = rotation;
        status =
                clSetKernelArg(dnn_kernel, 3, sizeof(cl_int), &rotate_cw_degrees);
        CHECK_AND_RETURN(status, "failed to set rotation");
    }

    // this is done at the beginning so that the quality algorithm has
    // had the option to use the events
    release_events(&event_array);

    // even though inp_format is assigned image_format_t,
    // the type is set to cl_int since underlying enum types
    // can vary and we want a known size on both the client and server.
    cl_int inp_format;
    cl_event dnn_wait_event, dnn_read_event;

    reset_event_array(&event_array);

    uint64_t size = 0;

    // used to pass the buffer with the output contents to the dnn
    cl_mem inp_buf = NULL;

    // the local device does not support other compression types, but this this function with
    // local devices should only be called with no compression, so other paths will not be
    // reached. There is also an assert to make sure of this.
    if (NO_COMPRESSION == compressionType) {
        size = img_buf_size;
        // normal execution
        inp_format = YUV_SEMI_PLANAR;
        // copy the yuv image data over to the host_img_buf and make sure it's semiplanar.
        copy_yuv_to_array(image_data, compression_type, host_img_buf);
        status = clEnqueueWriteBuffer(commandQueue[device_index_copy], img_buf[device_index_copy],
                                      CL_FALSE, 0,
                                      img_buf_size, host_img_buf,
                                      0, NULL, &dnn_wait_event);
        CHECK_AND_RETURN(status, "failed to write image to ocl buffers");
        append_to_event_array(&event_array, dnn_wait_event, "write_img_event");
        inp_buf = img_buf[device_index_copy];
    } else if (YUV_COMPRESSION == compression_type) {
        size = img_buf_size;
        inp_format = yuv_context->output_format;
        copy_yuv_to_array(image_data, compression_type, yuv_context->host_img_buf);
        status = enqueue_yuv_compression(yuv_context, &event_array, &dnn_wait_event);
        inp_buf = yuv_context->out_buf;
        CHECK_AND_RETURN(status, "could not enqueue yuv compression work");
    } else if (JPEG_COMPRESSION == compression_type) {
        inp_format = jpeg_context->output_format;
        jpeg_context->quality = quality;
        copy_yuv_to_array(image_data, compression_type, jpeg_context->host_img_buf);
        status = enqueue_jpeg_compression(jpeg_context, &event_array, &dnn_wait_event);
        inp_buf = jpeg_context->out_buf;
        CHECK_AND_RETURN(status, "could not enqueue jpeg compression");
    } else if (HEVC_COMPRESSION == compression_type) {
        inp_format = hevc_context->output_format;
        copy_yuv_to_array(image_data, compression_type, hevc_context->host_img_buf);

        cl_event configure_event = NULL;
        // note: if you want to configure the codec,
        // set the variables you want in the context and
        // set isconfigured to 0
        if(1 != hevc_context->codec_configured) {
            // todo: remove these defaults once dynamic configuration is implemented
            hevc_context->i_frame_interval = 2;
            hevc_context->framerate = 5;
            hevc_context->bitrate = 640*480;
            configure_hevc_codec(hevc_context, &event_array, &configure_event);
        }

        status = enqueue_hevc_compression(hevc_context, &event_array, &configure_event, &dnn_wait_event);
        inp_buf = hevc_context->out_buf;
        CHECK_AND_RETURN(status, "could not enqueue hevc compression");
    } else {
        // process jpeg images
        size = image_data.data.jpeg.capacity;
        inp_format = RGB;
        status = enqueue_jpeg_image(image_data.data.jpeg.data,
                                    (const uint64_t *) &image_data.data.jpeg.capacity,
                                    3, &dnn_wait_event);
        CHECK_AND_RETURN(status, "could not enqueue jpeg image");
        inp_buf = img_buf[device_index_copy];
    }

    // Testing ping with CL calls to fill a buffer
    cl_event fill_event;
    status = ping_fillbuffer_run(PING_CTX, commandQueue[device_index], &event_array, &fill_event);
    CHECK_AND_RETURN(status, "PING failed to run");

    status = enqueue_dnn(&dnn_wait_event, device_index_copy, 0, do_segment_copy, detection_array,
                         segmentation_array, inp_format, is_eval_frame, &tmp_detect_copy_event,
                         &tmp_postprocess_copy_event, &dnn_read_event, inp_buf);
    CHECK_AND_RETURN(status, "could not enqueue dnn kernels");

    int is_eval_ready = 0;
    if (is_eval_running) {
        // eval is running, check if it's ready
        cl_int eval_status;
        status = clGetEventInfo(eval_read_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                                &eval_status, NULL);
        CHECK_AND_RETURN(status, "could not get eval event info");

        if (eval_status == CL_COMPLETE) {
            // eval finished
            is_eval_running = 0;
            is_eval_ready = 1;

            if (VERBOSITY >= 1) {
                LOGI("=== Frame %3d: EVAL IOU: finished\n", frame_index);
            }
        } else {
            // eval not finished
            if (VERBOSITY >= 1) {
                LOGI("=== Frame %3d: EVAL IOU: still running...\n", frame_index);
            }
        }
    } else if (is_eval_frame) {
        // eval is not running and it's time to start a new eval run
        if (VERBOSITY >= 1) {
            LOGI("=== Frame %3d: EVAL IOU: started eval\n", frame_index);
        }

        status = enqueue_dnn_eval(&tmp_detect_copy_event, &tmp_postprocess_copy_event,
                                  device_index_copy, do_segment_copy, iou, &eval_read_event);
        CHECK_AND_RETURN(status, "could not enqueue eval kernels");

        is_eval_running = 1;
    } else {
        if (VERBOSITY >= 1) {
            LOGI("=== Frame %3d: EVAL IOU: nothing\n", frame_index);
        }
    }

    if (VERBOSITY >= 1) {
        LOGI("=== Frame %3d, EVAL IOU = %f, no. detections: %d\n", frame_index, *iou,
             detection_array[0]);
    }

    cl_event wait_events[] = { dnn_read_event, fill_event };
    status = clWaitForEvents(2, wait_events);
    CHECK_AND_RETURN(status, "could not wait for final events");

#if defined(PRINT_PROFILE_TIME)
    timespec timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_start, timespec_b, "total cl stuff");
#endif

    if (ENABLE_PROFILING & config_flags) {
        cl_mem sz_buf = nullptr;
        cl_command_queue sz_queue = nullptr;

        if (JPEG_COMPRESSION == compression_type) {
            sz_buf = jpeg_context->size_buf;
            sz_queue = jpeg_context->enc_queue;
        } else if (HEVC_COMPRESSION == compression_type) {
            sz_buf = hevc_context->size_buf;
            sz_queue = hevc_context->enc_queue;
        }

        // read size buffer
        if (sz_buf != nullptr && sz_queue != nullptr) {
            cl_event sz_read_event;
            status = clEnqueueReadBuffer(sz_queue, sz_buf,
                                         CL_FALSE, 0, sizeof(cl_ulong),
                                         &size, 1, &dnn_wait_event, &sz_read_event);
            CHECK_AND_RETURN(status, "could not read size buffer");

            status = clWaitForEvents(1, &sz_read_event);
            CHECK_AND_RETURN(status, "failed to wait for sz read event");
        }


        status = print_events(file_descriptor, frame_index, &event_array);
        CHECK_AND_RETURN(status, "failed to print events");

        dprintf(file_descriptor, "%d,frame,timestamp,%ld\n", frame_index,
                image_timestamp);  // this should match device timestamp in camera log
        dprintf(file_descriptor, "%d,frame,is_eval,%d\n", frame_index, is_eval_frame);
        dprintf(file_descriptor, "%d,device,index,%d\n", frame_index, device_index_copy);
        dprintf(file_descriptor, "%d,config,segment,%d\n", frame_index, do_segment_copy);
        dprintf(file_descriptor, "%d,compression,name,%s\n", frame_index,
                get_compression_name(compression_type));
        dprintf(file_descriptor, "%d,compression,quality,%d\n", frame_index, quality);
        dprintf(file_descriptor, "%d,compression,size,%llu\n", frame_index, size);
        if (is_eval_ready) {
            dprintf(file_descriptor, "%d,dnn,iou,%f\n", frame_index, *iou);
        }
    }

    if (is_eval_ready) {
        if (ENABLE_PROFILING & config_flags) {
            status = print_events(file_descriptor, frame_index, &eval_event_array);
            CHECK_AND_RETURN(status, "failed to print eval events");
        }

        release_events(&eval_event_array);
        reset_event_array(&eval_event_array);
    }

    clock_gettime(CLOCK_MONOTONIC, &timespec_stop);
#if defined(PRINT_PROFILE_TIME)
    printTime(timespec_b, timespec_stop, "printing messages");
#endif

    if (ENABLE_PROFILING & config_flags) {
        dprintf(file_descriptor, "%d,frame_time,start_ns,%lu\n", frame_index,
                (timespec_start.tv_sec * 1000000000) + timespec_start.tv_nsec);
        dprintf(file_descriptor, "%d,frame_time,stop_ns,%lu\n", frame_index,
                (timespec_stop.tv_sec * 1000000000) + timespec_stop.tv_nsec);
    }

    frame_index++;
    return 0;
}

char *
get_c_log_string_pocl() {
    return c_log_string;
}

const char *
get_compression_name(const compression_t compression_id) {
    switch (compression_id) {
        case NO_COMPRESSION:
            return "none";
        case YUV_COMPRESSION:
            return "yuv";
        case JPEG_COMPRESSION:
            return "jpeg";
        case HEVC_COMPRESSION:
            return "hevc";
        case JPEG_IMAGE:
            return "jpeg_image";
        default:
            return "unknown";
    }
}

#ifdef __cplusplus
}
#endif
