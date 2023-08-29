

#define CL_HPP_MINIMUM_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <rename_opencl.h>
#include <CL/cl.h>
#include "event_logger.h"
#include "poclImageProcessor.h"
#include <string>
#include <stdlib.h>
#include <vector>
#include "sharedUtils.h"
#include <assert.h>

#ifdef __ANDROID__

#include <android/log.h>

#endif

#ifdef __cplusplus
extern "C" {
#endif


#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

static int detection_count = 1 + MAX_DETECTIONS * 6;
static int segmentation_count = MAX_DETECTIONS * MASK_W * MASK_H;
static int seg_postprocess_count = 4 * MASK_W * MASK_H; // RGBA image
static int total_out_count = detection_count + segmentation_count;

static int colors[] = {-1651865, -6634562, -5921894,
                       -9968734, -1277957, -2838283,
                       -9013359, -9634954, -470042,
                       -8997255, -4620585, -2953862,
                       -3811878, -8603498, -2455171,
                       -5325920, -6757258, -8214427,
                       -5903423, -4680978, -4146958,
                       -602947, -5396049, -9898511,
                       -8346466, -2122577, -2304523,
                       -4667802, -222837, -4983945,
                       -234790, -8865559, -4660525,
                       -3744578, -8720427, -9778035,
                       -680538, -7942224, -7162754,
                       -2986121, -8795194, -2772629,
                       -4820488, -9401960, -3443339,
                       -1781041, -4494168, -3167240,
                       -7629631, -6685500, -6901785,
                       -2968136, -3953703, -4545430,
                       -6558846, -2631687, -5011272,
                       -4983118, -9804322, -2593374,
                       -8473686, -4006938, -7801488,
                       -7161859, -4854121, -5654350,
                       -817410, -8013957, -9252928,
                       -2240041, -3625560, -6381719,
                       -4674608, -5704237, -8466309,
                       -1788449, -7283030, -5781889,
                       -4207444, -8225948};

#define MAX_NUM_CL_DEVICES 4

// todo: increase these values to match the values in dct.cl
#define BLK_W 1
#define BLK_H 1
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

// kernel execution loop related things
static size_t tot_pixels;
static size_t img_buf_size;
static cl_int inp_w;
static cl_int inp_h;
static cl_int rotate_cw_degrees;
static cl_int inp_format;
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
static cl_uchar *host_img_buf = nullptr;
static size_t local_size;
static size_t global_size;

// related to compression kernels
static size_t enc_y_global[2], enc_uv_global[2];
static size_t dec_y_global[2], dec_uv_global[2];
static cl_mem out_enc_y_buf = nullptr, out_enc_uv_buf = nullptr;

// make these variables global, just in case
// they don't get set properly during processing.
int device_index_copy = 0;
int do_segment_copy = 0;
int local_do_compression = 0;
unsigned char enable_profiling = 0;

char *c_log_string = nullptr;
#define DATA_POINT_SIZE 22  // number of decimal digits for 2^64 + ', '
// allows for 9 datapoints plus 3 single digit configs, newline and term char
#define LOG_BUFFER_SIZE (DATA_POINT_SIZE * 9 + 9 +2)
#define MAX_EVENTS 9
int file_descriptor;
int frame_index;
// used to both log and free events
event_array_t event_array;

// names of compression types, used for logging
static const char *compression_names[] = {
        "none",
        "yuv"
};

// variable to check if everything is ready for execution
int setup_success = 0;


/**
 * @note height and width are rotated since the camera is rotated 90 degrees with respect to the
 * screen
 * @param env
 * @param jAssetManager
 * @param devices
 * @return
 */
static int
init_codecs(cl_device_id enc_device, cl_device_id dec_device, const char *source, const size_t
src_size) {

    int status;

    cl_device_id codec_devices[] = {enc_device, dec_device};

    codec_program = clCreateProgramWithSource(context, 1, &source,
                                              &src_size,
                                              &status);
    CHECK_AND_RETURN(status, "creation of codec program failed");

    status = clBuildProgram(codec_program, 2, codec_devices, nullptr, nullptr, nullptr);
    CHECK_AND_RETURN(status, "building codec program failed");

    // proxy device buffers
    // input for compression
    // important that it is read only since both enc kernels read from it.
    img_buf[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, img_buf_size,
                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    out_enc_y_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_y_buf");
    out_enc_uv_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels / 2, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create out_enc_uv_buf");

    enc_y_kernel = clCreateKernel(codec_program, "encode_y", &status);
    CHECK_AND_RETURN(status, "creating encode_y kernel failed");
    int narg = 0;
    status = clSetKernelArg(enc_y_kernel, narg++, sizeof(cl_mem), &img_buf[1]);
    status |= clSetKernelArg(enc_y_kernel, narg++, sizeof(int), &inp_w);
    status |= clSetKernelArg(enc_y_kernel, narg++, sizeof(int), &inp_h);
    status |= clSetKernelArg(enc_y_kernel, narg++, sizeof(cl_mem), &out_enc_y_buf);
    CHECK_AND_RETURN(status, "setting encode_y kernel args failed");

    enc_uv_kernel = clCreateKernel(codec_program, "encode_uv", &status);
    CHECK_AND_RETURN(status, "creating encode_uv kernel failed");
    narg = 0;
    status = clSetKernelArg(enc_uv_kernel, narg++, sizeof(cl_mem), &img_buf[1]);
    status |= clSetKernelArg(enc_uv_kernel, narg++, sizeof(int), &inp_h);
    status |= clSetKernelArg(enc_uv_kernel, narg++, sizeof(int), &inp_w);
    status |= clSetKernelArg(enc_uv_kernel, narg++, sizeof(cl_mem), &out_enc_uv_buf);
    CHECK_AND_RETURN(status, "setting encode_uv kernel args failed");

    dec_y_kernel = clCreateKernel(codec_program, "decode_y", &status);
    CHECK_AND_RETURN(status, "creating decode_y kernel failed");
    narg = 0;
    status = clSetKernelArg(dec_y_kernel, narg++, sizeof(cl_mem), &out_enc_y_buf);
    status |= clSetKernelArg(dec_y_kernel, narg++, sizeof(int), &inp_w);
    status |= clSetKernelArg(dec_y_kernel, narg++, sizeof(int), &inp_h);
    status |= clSetKernelArg(dec_y_kernel, narg++, sizeof(cl_mem), &img_buf[2]);
    CHECK_AND_RETURN(status, "setting decode_y kernel args failed");

    dec_uv_kernel = clCreateKernel(codec_program, "decode_uv", &status);
    CHECK_AND_RETURN(status, "creating decode_uv kernel failed");
    narg = 0;
    status = clSetKernelArg(dec_uv_kernel, narg++, sizeof(cl_mem), &out_enc_uv_buf);
    status |= clSetKernelArg(dec_uv_kernel, narg++, sizeof(int), &inp_h);
    status |= clSetKernelArg(dec_uv_kernel, narg++, sizeof(int), &inp_w);
    status |= clSetKernelArg(dec_uv_kernel, narg++, sizeof(cl_mem), &img_buf[2]);
    CHECK_AND_RETURN(status, "setting decode_uv kernel args failed");

    // set global work group sizes for codec kernels
    enc_y_global[0] = (size_t) (inp_w / BLK_W);
    enc_y_global[1] = (size_t) (inp_h / BLK_H);

    enc_uv_global[0] = (size_t) (inp_h / BLK_H) / 2;
    enc_uv_global[1] = (size_t) (inp_w / BLK_W) / 2;

    dec_y_global[0] = (size_t) (inp_w / BLK_W);
    dec_y_global[1] = (size_t) (inp_h / BLK_H);

    dec_uv_global[0] = (size_t) (inp_h / BLK_H) / 2;
    dec_uv_global[1] = (size_t) (inp_w / BLK_W) / 2;

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
initPoclImageProcessor(const int width, const int height, const bool enableProfiling,
                       const char *codec_sources, const size_t src_size, const int fd) {
    cl_platform_id platform;
    cl_device_id devices[MAX_NUM_CL_DEVICES] = {nullptr};
    cl_uint devices_found;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_NUM_CL_DEVICES, devices,
                            &devices_found);
    CHECK_AND_RETURN(status, "getting device id failed");
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "Platform has %d devices", devices_found);
#endif
    assert(devices_found > 0);

    // some info
    char result_array[256];
    for (unsigned i = 0; i < devices_found; ++i) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);
#endif
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 256 * sizeof(char), result_array, NULL);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_VERSION: %s", result_array);
#endif
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, 256 * sizeof(char), result_array, NULL);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DRIVER_VERSION: %s", result_array);
#endif
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);
#endif
    }

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    context = clCreateContext(cps, devices_found, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    enable_profiling = enableProfiling;
    cl_command_queue_properties cq_properties = 0;
    if (enable_profiling) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "enabling profiling");
#endif
        cq_properties = CL_QUEUE_PROFILING_ENABLE;
    }

    for (unsigned i = 0; i < devices_found; ++i) {
        commandQueue[i] = clCreateCommandQueue(context, devices[i], cq_properties, &status);
        CHECK_AND_RETURN(status, "creating command queue failed");
    }

    // Only create builtin kernel for basic and remote devices
    std::string dnn_kernel_name = "pocl.dnn.detection.u8";
    std::string postprocess_kernel_name = "pocl.dnn.segmentation_postprocess.u8";
    std::string kernel_names = dnn_kernel_name + ";" + postprocess_kernel_name;

    if (1 == devices_found) {
        program = clCreateProgramWithBuiltInKernels(context, 1, devices, kernel_names
                .c_str(), &status);
        CHECK_AND_RETURN(status, "creation of program failed");

        status = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);
        CHECK_AND_RETURN(status, "building of program failed");

    } else {
        // create kernels needed for remote execution
        cl_device_id inference_devices[] = {devices[0], devices[2]};

        program = clCreateProgramWithBuiltInKernels(context, 2, inference_devices,
                                                    kernel_names.c_str(), &status);
        CHECK_AND_RETURN(status, "creation of program failed");

        status = clBuildProgram(program, 2, inference_devices, nullptr, nullptr, nullptr);
        CHECK_AND_RETURN(status, "building of program failed");
    }

    dnn_kernel = clCreateKernel(program, dnn_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    postprocess_kernel = clCreateKernel(program, postprocess_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    // set some default values;
    rotate_cw_degrees = 90;
    // 0 - RGB
    // 1 - YUV420 NV21 Android (interleaved U/V)
    // 2 - YUV420 (U/V separate)
    inp_format = 1;

    inp_w = width;
    inp_h = height;
    tot_pixels = inp_w * inp_h;
    // yuv420 so 6 bytes for every 4 pixels
    img_buf_size = (tot_pixels * 3) / 2;
    img_buf[0] = clCreateBuffer(context, (CL_MEM_READ_ONLY),
                                img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the image buffer");

    out_buf[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    out_mask_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     tot_pixels * MAX_DETECTIONS * sizeof(cl_char) / 4,
                                     NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    // RGBA:
    postprocess_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        tot_pixels * 4 * sizeof(cl_char) / 4,
                                        NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation preprocessing buffer");

    // only allocate these buffers if the remote is also available
    if (devices_found > 1) {
        // remote device buffers
        img_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    img_buf_size, NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the image buffer");

        out_buf[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                    NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the output buffer");

        out_mask_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels * MAX_DETECTIONS *
                                                                     sizeof(cl_char) / 4,
                                         NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

        postprocess_buf[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            tot_pixels * 4 * sizeof(cl_char) / 4,
                                            NULL, &status);
        CHECK_AND_RETURN(status, "failed to create the segmentation preprocessing buffer");
    }

    //status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &img_buf);
    status = clSetKernelArg(dnn_kernel, 1, sizeof(cl_int), &inp_w);
    status |= clSetKernelArg(dnn_kernel, 2, sizeof(cl_int), &inp_h);
    status |=
            clSetKernelArg(dnn_kernel, 3, sizeof(cl_int), &rotate_cw_degrees);
    status |= clSetKernelArg(dnn_kernel, 4, sizeof(cl_int), &inp_format);
    //status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &out_buf);
    //status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &out_mask_buf);

    //status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &out_buf);
    //status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem), &out_mask_buf);
    //status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem), &postprocess_buf);
    CHECK_AND_RETURN(status, "could not assign dnn kernel args");

    global_size = 1;
    local_size = 1;

    // buffer to copy image data to
    host_img_buf = (cl_uchar *) malloc(img_buf_size);
    // string to write values to for logging
    c_log_string = (char *) malloc(LOG_BUFFER_SIZE * sizeof(char));

    if (devices_found > 1) {

        status = init_codecs(devices[1], devices[3], codec_sources, src_size);
        CHECK_AND_RETURN(status, "init of codec kernels failed");
    }

    for (unsigned i = 0; i < MAX_NUM_CL_DEVICES; ++i) {
        if (nullptr != devices[i]) {
            clReleaseDevice(devices[i]);
        }
    }

    // setup profiling
    if (enableProfiling) {
        file_descriptor = fd;
        status = dprintf(file_descriptor, CSV_HEADER);
        CHECK_AND_RETURN((status < 0), "could not write csv header");
    }
    frame_index = 0;
    event_array = create_event_array(MAX_EVENTS);

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

        if (nullptr != out_mask_buf[i]) {
            clReleaseMemObject(out_mask_buf[i]);
            out_mask_buf[i] = nullptr;
        }

        if (nullptr != postprocess_buf[i]) {
            clReleaseMemObject(postprocess_buf[i]);
            postprocess_buf[i] = nullptr;
        }
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

    if (program != nullptr) {
        clReleaseProgram(program);
        program = nullptr;
    }

    if (nullptr != host_img_buf) {
        free(host_img_buf);
        host_img_buf = nullptr;
    }

    if (nullptr != c_log_string) {
        free(c_log_string);
        c_log_string = nullptr;
    }

    free_event_array(&event_array);

    return 0;
}

#if defined(PRINT_PROFILE_TIME)
void printTime(timespec start, timespec stop, char message[256]) {

    time_t s = stop.tv_sec - start.tv_sec;
    unsigned long final = s * 1000000000;
    final += stop.tv_nsec;
    final -= start.tv_nsec;
    unsigned secs, nsecs;
    int ms = final / 1000000;
    int ns = final % 1000000;

    char display_message[256 + 16];
    strcpy(display_message, message);
    strcat(display_message, ": %d ms, %d ns");
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing", display_message, ms, ns);
#endif
}
#endif

/**
 * Function to enqueue the opencl commands related to object detection.
 * @param wait_event event to wait on before starting these commands
 * @param device_index index of device doing object detection
 * @param do_segment bool to enable/disable segmentation
 * @param detection_array array of bounding boxes of detected objects
 * @param segmentation_array bitmap of segments of detected objects
 * @param events struct of events of these commands
 * @return 0 if everything went well, otherwise a cl error number
 */
cl_int
enqueue_dnn(const cl_event *wait_event, const int device_index, const int do_segment, cl_int
*detection_array, cl_char *segmentation_array) {

    cl_int status;

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

    status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &img_buf[device_index]);
    status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &out_buf[device_index]);
    status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &out_mask_buf[device_index]);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &out_buf[device_index]);
    status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem),
                             &out_mask_buf[device_index]);
    status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem),
                             &postprocess_buf[device_index]);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "assigning kernel params");
#endif

    cl_event run_dnn_event, read_detection_event;

    status = clEnqueueNDRangeKernel(commandQueue[device_index], dnn_kernel, 1, NULL,
                                    &global_size, &local_size, 1,
                                    wait_event, &run_dnn_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");
    append_to_event_array(&event_array, run_dnn_event, VAR_NAME(dnn_event));

    status = clEnqueueReadBuffer(commandQueue[device_index], out_buf[device_index],
                                 CL_FALSE, 0,
                                 detection_count * sizeof(cl_int), detection_array, 1,
                                 &run_dnn_event, &read_detection_event);
    CHECK_AND_RETURN(status, "failed to read detection result buffer");
    append_to_event_array(&event_array, read_detection_event, VAR_NAME(detect_event));

    if (do_segment) {
        cl_event run_postprocess_event, read_segment_event;

        status = clEnqueueNDRangeKernel(commandQueue[device_index], postprocess_kernel, 1,
                                        NULL,
                                        &global_size, &local_size, 1,
                                        &run_dnn_event, &run_postprocess_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");
        append_to_event_array(&event_array, run_postprocess_event, VAR_NAME(postprocess_event));

        status = clEnqueueReadBuffer(commandQueue[device_index],
                                     postprocess_buf[device_index], CL_FALSE, 0,
                                     seg_postprocess_count * sizeof(cl_char), segmentation_array, 1,
                                     &run_postprocess_event, &read_segment_event);
        CHECK_AND_RETURN(status, "failed to read segmentation postprocess result buffer");
        append_to_event_array(&event_array, read_segment_event, VAR_NAME(segment_event));

    }

    return 0;
}

/**
 * function to copy raw buffers from the image to a local array and make sure the result is in
 * nv21 format.
 * @param y_ptr
 * @param yrow_stride
 * @param ypixel_stride
 * @param u_ptr
 * @param v_ptr
 * @param uvpixel_stride
 * @param dest_buf
 */
void
copy_yuv_to_array(const cl_char *y_ptr, const int yrow_stride, const int ypixel_stride,
                  const cl_char *u_ptr, const cl_char *v_ptr, const int uvpixel_stride,
                  cl_uchar *dest_buf) {

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

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

        dest_buf[uv_start_index + 2 * i] = v_ptr[i * uvpixel_stride];

        dest_buf[uv_start_index + 1 + 2 * i] = u_ptr[i * uvpixel_stride];
    }

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "copying image to array");
#endif

}

/**
 * enqueue the required opencl commands needed for YUV compression.
 * @param input_img the array holding the image data
 * @param buf_size the size size of the input_img
 * @param enc_index the index of the compressing device
 * @param dec_index the index of the decompressing device
 * @param events the struct to log events to
 * @param result_event a return event that can be used to wait for compression to be done.
 * @return
 */
cl_int
enqueue_yuv_compression(const cl_uchar *input_img, const size_t buf_size, const int enc_index,
                        const int dec_index, cl_event *result_event) {
    cl_int status;
    cl_event enc_image_event, enc_y_event, enc_uv_event,
            dec_y_event, dec_uv_event;

    status = clEnqueueWriteBuffer(commandQueue[enc_index], img_buf[enc_index], CL_FALSE, 0,
                                  buf_size, input_img, 0, NULL, &enc_image_event);
    CHECK_AND_RETURN(status, "failed to write image to enc buffers");
    append_to_event_array(&event_array, enc_image_event, VAR_NAME(enc_image_event));

    status = clEnqueueNDRangeKernel(commandQueue[enc_index], enc_y_kernel, 2, NULL, enc_y_global,
                                    NULL, 1, &enc_image_event, &enc_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_y_kernel");
    append_to_event_array(&event_array, enc_y_event, VAR_NAME(enc_y_event));

    status = clEnqueueNDRangeKernel(commandQueue[enc_index], enc_uv_kernel, 2, NULL, enc_uv_global,
                                    NULL, 1, &enc_image_event, &enc_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue enc_uv_kernel");
    append_to_event_array(&event_array, enc_uv_event, VAR_NAME(enc_uv_event));

    status = clEnqueueNDRangeKernel(commandQueue[dec_index], dec_y_kernel, 2, NULL, dec_y_global,
                                    NULL, 1, &enc_y_event, &dec_y_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_y_kernel");
    append_to_event_array(&event_array, dec_y_event, VAR_NAME(dec_y_event));

    // we have to wait for both since dec_y and dec_uv write to the same buffer and there is
    // no guarantee what happens if both dec_y and dec_uv write at the same time.
    cl_event dec_uv_wait_events[] = {enc_uv_event, dec_y_event};
    status = clEnqueueNDRangeKernel(commandQueue[dec_index], dec_uv_kernel, 2, NULL, dec_uv_global,
                                    NULL, 2, dec_uv_wait_events, &dec_uv_event);
    CHECK_AND_RETURN(status, "failed to enqueue dec_uv_kernel");
    append_to_event_array(&event_array, dec_uv_event, VAR_NAME(dec_uv_event));

    // move the intermediate buffers back to the phone after decompression.
    // Since we don't care about the contents, the latest state of the buffer is not moved
    // back from the remote.
    // https://man.opencl.org/clEnqueueMigrateMemObjects.html
    cl_mem migrate_bufs[] = {out_enc_y_buf, out_enc_uv_buf};
    status = clEnqueueMigrateMemObjects(commandQueue[enc_index], 2, migrate_bufs,
                                        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1,
                                        &dec_uv_event, NULL);
    CHECK_AND_RETURN(status, "failed to migrate buffers back");

    // set the event that other ocl commands can wait for
    *result_event = dec_uv_event;

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
poclProcessYUVImage(const int device_index, const int do_segment, const int do_compression,
                    const int rotation, const int8_t *y_ptr, const int yrow_stride,
                    const int ypixel_stride, const int8_t *u_ptr, const int8_t *v_ptr,
                    const int uvrow_stride, const int uvpixel_stride, int32_t *detection_array,
                    int8_t *segmentation_array) {

    if (!setup_success) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG,
                            "poclProcessYUVImage called but setup did not complete successfully");
#endif
        return 1;
    }

    struct timespec timespec_start, timespec_stop;
    clock_gettime(CLOCK_MONOTONIC, &timespec_start);

    // make local copies so that they don't change during execution
    // when the user presses a button.
    device_index_copy = device_index;
    do_segment_copy = do_segment;
    local_do_compression = do_compression;

    cl_int status;

    // todo: remove this variable, since the phone is always kept in portrait mode.
    rotate_cw_degrees = rotation;

    // copy the yuv image data over to the host_img_buf and make sure it's semiplanar.
    copy_yuv_to_array(y_ptr, yrow_stride, ypixel_stride, u_ptr, v_ptr, uvpixel_stride,
                      host_img_buf);

    cl_event dnn_wait_event;
    reset_event_array(&event_array);

    // do compression
    if (2 == device_index_copy && (1 == local_do_compression)) {

        status = enqueue_yuv_compression(host_img_buf, img_buf_size, 1, 3,
                                         &dnn_wait_event);
        CHECK_AND_RETURN(status, "could enqueue yuv compression work");

    } else {
        // normal execution
        status = clEnqueueWriteBuffer(commandQueue[device_index_copy], img_buf[device_index_copy],
                                      CL_FALSE, 0,
                                      img_buf_size, host_img_buf,
                                      0, NULL, &dnn_wait_event);
        CHECK_AND_RETURN(status, "failed to write image to ocl buffers");
        append_to_event_array(&event_array, dnn_wait_event, "enc_image_event");

    }

    status = enqueue_dnn(&dnn_wait_event, device_index_copy, do_segment_copy, detection_array,
                         segmentation_array);
    CHECK_AND_RETURN(status, "could not enqueue dnn kernels");

    status = clFinish(commandQueue[device_index_copy]);
    CHECK_AND_RETURN(status, "failed to clfinish");

#if defined(PRINT_PROFILE_TIME)
    timespec timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_start, timespec_b, "total cl stuff");
#endif

    if (enable_profiling) {

        print_events(file_descriptor, frame_index, &event_array);

        dprintf(file_descriptor, "%d,device,index,%d\n", frame_index, device_index_copy);
        dprintf(file_descriptor, "%d,config,segment,%d\n", frame_index, do_segment_copy);
        dprintf(file_descriptor, "%d,compression,name,%s\n", frame_index,
                compression_names[local_do_compression]);

    }

    release_events(&event_array);

//    __android_log_print(ANDROID_LOG_DEBUG, "DETECTION", "%d %d %d %d %d %d %d", result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6]);

    clock_gettime(CLOCK_MONOTONIC, &timespec_stop);
#if defined(PRINT_PROFILE_TIME)
    printTime(timespec_b, timespec_stop, "printing messages");
#endif

    if (enable_profiling) {
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

#ifdef __cplusplus
}
#endif

