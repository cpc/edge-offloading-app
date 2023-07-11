

//#include <rename_opencl.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <rename_opencl.h>
#include <CL/cl.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <string>
#include <stdlib.h>
#include <vector>
#include <jni.h>
#include <android/bitmap.h>
#include "sharedUtils.h"
#include <assert.h>

#include <fstream>


#ifdef __cplusplus
extern "C" {
#endif

#define LOGTAG "poclimageprocessor"
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
// allows for 6 datapoints plus newline and term char
#define LOG_BUFFER_SIZE (DATA_POINT_SIZE * 10 +2)

// enable this to print timing to logs
//#define PRINT_PROFILE_TIME

// variable to check if everything is ready for execution
int setup_success = 0;

// Global variables for smuggling our blob into PoCL so we can pretend it is a builtin kernel.
// Please don't ever actually do this in production code.
const char *pocl_onnx_blob = NULL;
uint64_t pocl_onnx_blob_size = 0;

bool smuggleONNXAsset(JNIEnv *env, jobject jAssetManager, const char *filename) {
    AAssetManager *assetManager = AAssetManager_fromJava(env, jAssetManager);
    if (assetManager == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager", "Failed to get asset manager");
        return false;
    }

    AAsset *a = AAssetManager_open(assetManager, "yolov8n-seg.onnx", AASSET_MODE_STREAMING);
    auto num_bytes = AAsset_getLength(a);
    char *tmp = new char[num_bytes + 1];
    tmp[num_bytes] = 0;
    int read_bytes = AAsset_read(a, tmp, num_bytes);
    AAsset_close(a);
    if (read_bytes != num_bytes) {
        delete[] tmp;
        __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager", "Failed to read asset contents");
        return false;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "NDK_Asset_Manager", "ONNX blob read successfully");

    // Smuggling in progress
    pocl_onnx_blob = tmp;
    pocl_onnx_blob_size = num_bytes;
    return true;
}

void destroySmugglingEvidence() {
    char *tmp = (char *) pocl_onnx_blob;
    pocl_onnx_blob_size = 0;
    pocl_onnx_blob = nullptr;
    delete[] tmp;
}

// Read contents of files
static char *
read_file(JNIEnv *env, jobject jAssetManager, const char *filename, size_t *bytes_read) {

    AAssetManager *asset_manager = AAssetManager_fromJava(env, jAssetManager);
    if (asset_manager == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG "read_file", "Failed to get asset manager "
                                                                   "to read file");
        return nullptr;
    }

    AAsset *a_asset = AAssetManager_open(asset_manager, filename, AASSET_MODE_STREAMING);

    const size_t asset_size = AAsset_getLength(a_asset);

    char *contents = (char *) malloc(asset_size);

    *bytes_read = AAsset_read(a_asset, contents, asset_size);

    if (asset_size != *bytes_read) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG "read_file", "Failed to read file contents");
        free(contents);
        return nullptr;
    }

    return contents;

}

/**
 * @note height and width are rotated since the camera is rotated 90 degrees with respect to the
 * screen
 * @param env
 * @param jAssetManager
 * @param devices
 * @return
 */
static int
init_codecs(JNIEnv *env, jobject jAssetManager, cl_device_id *devices) {

    int status;

    cl_device_id codec_devices[] = {devices[1], devices[3]};

    // create codec kernels from source files.
    size_t src_size;
    const char *source = read_file(env, jAssetManager, "kernels/copy.cl", &src_size);
    if (nullptr == source) {
        return -1;
    }
    codec_program = clCreateProgramWithSource(context, 1, &source,
                                              &src_size,
                                              &status);
    CHECK_AND_RETURN(status, "creation of codec program failed");
    free((void *) source);

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
 * @param env
 * @param clazz
 * @return
 */
JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_initPoclImageProcessor(JNIEnv *env,
                                                                              jclass clazz,
                                                                              jboolean enableProfiling,
                                                                              jobject jAssetManager,
                                                                              jint width,
                                                                              jint height) {
    cl_platform_id platform;
    cl_device_id devices[MAX_NUM_CL_DEVICES] = {nullptr};
    cl_uint devices_found;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_NUM_CL_DEVICES, devices,
                            &devices_found);
    CHECK_AND_RETURN(status, "getting device id failed");
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "Platform has %d devices", devices_found);
    assert(devices_found > 0);

    // some info
    char result_array[256];
    for (unsigned i = 0; i < devices_found; ++i) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 256 * sizeof(char), result_array, NULL);
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_VERSION: %s", result_array);

        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, 256 * sizeof(char), result_array, NULL);
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DRIVER_VERSION: %s", result_array);

        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);
    }

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    context = clCreateContext(cps, devices_found, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    enable_profiling = enableProfiling;
    cl_command_queue_properties cq_properties = 0;
    if (enable_profiling) {
        __android_log_print(ANDROID_LOG_INFO, LOGTAG, "enabling profiling");
        cq_properties = CL_QUEUE_PROFILING_ENABLE;
    }
    
    for (unsigned i = 0; i < devices_found; ++i) {
        commandQueue[i] = clCreateCommandQueue(context, devices[i], cq_properties, &status);
        CHECK_AND_RETURN(status, "creating command queue failed");
    }

    bool smuggling_ok = smuggleONNXAsset(env, jAssetManager, "yolov8n-seg.onnx");
    assert(smuggling_ok);

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

    destroySmugglingEvidence();

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
    img_buf_size = (tot_pixels * 3) / 2 ;
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

    if(devices_found > 1){
        status = init_codecs(env, jAssetManager, devices);
        CHECK_AND_RETURN(status, "init of codec kernels failed");
    }

    for (unsigned i = 0; i < MAX_NUM_CL_DEVICES; ++i) {
        if (nullptr != devices[i]) {
            clReleaseDevice(devices[i]);
        }
    }

    setup_success = 1;
    return 0;
}

/**
 * release everything related to PoCL
 * @param env
 * @param clazz
 * @return
 */
JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_destroyPoclImageProcessor(JNIEnv *env,
                                                                                 jclass clazz) {

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

        if (nullptr != out_buf[i]){
            clReleaseMemObject(out_buf[i]);
            out_buf[i] = nullptr;
        }

        if (nullptr != out_mask_buf[i]){
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

    return 0;
}

cl_ulong getEventRuntime(cl_event event){
    int status;
    cl_ulong event_start, event_end;

    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                                     &event_start, NULL);
    CHECK_AND_RETURN(status, "could not read event start date");

    status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                                     &event_end, NULL);
    CHECK_AND_RETURN(status, "could not read event end date");

    // useful for debugging
//    __android_log_print(ANDROID_LOG_WARN, LOGTAG, "timestampe values: %ld, %ld", event_end, event_start );

    return event_end - event_start;
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
    __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing", display_message, ms, ns);
}
#endif

/**
 * process the image with PoCL.
 *  assumes that image format is YUV420_888
 * @param env
 * @param clazz
 * @param width
 * @param height
 * @param y
 * @param yrow_stride
 * @param ypixel_stride
 * @param u
 * @param v
 * @param uvrow_stride
 * @param uvpixel_stride
 * @param result
 * @return
 */
JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclProcessYUVImage(JNIEnv *env,
                                                                           jclass clazz,
                                                                           jint device_index,
                                                                           jint do_segment,
                                                                           jint doCompression,
                                                                           jint rotation, jobject y,
                                                                           jint yrow_stride,
                                                                           jint ypixel_stride,
                                                                           jobject u, jobject v,
                                                                           jint uvrow_stride,
                                                                           jint uvpixel_stride,
                                                                           jintArray detection_result,
                                                                           jbyteArray segmentation_result) {

    if (!setup_success) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG,
                            "poclProcessYUVImage called but setup did not complete successfully");
        return 1;
    }

    // make local copies so that they don't change during execution
    // when the user presses a button.
    device_index_copy = device_index;
    do_segment_copy = do_segment;
    local_do_compression = doCompression;

#if defined(PRINT_PROFILE_TIME)
    struct timespec timespec_a, timespec_b;
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
#endif

    cl_int status;

    status = clSetKernelArg(dnn_kernel, 0, sizeof(cl_mem), &img_buf[device_index_copy]);
    status |= clSetKernelArg(dnn_kernel, 5, sizeof(cl_mem), &out_buf[device_index_copy]);
    status |= clSetKernelArg(dnn_kernel, 6, sizeof(cl_mem), &out_mask_buf[device_index_copy]);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &out_buf[device_index_copy]);
    status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem),
                             &out_mask_buf[device_index_copy]);
    status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem),
                             &postprocess_buf[device_index_copy]);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "assigning kernel params");
#endif

    // todo: look into if iscopy=true works on android
    cl_int *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    cl_char *segmentation_array = env->GetByteArrayElements(segmentation_result, JNI_FALSE);

    cl_char *y_ptr = (cl_char *) env->GetDirectBufferAddress(y);
    cl_char *u_ptr = (cl_char *) env->GetDirectBufferAddress(u);
    cl_char *v_ptr = (cl_char *) env->GetDirectBufferAddress(v);

    // todo: remove this variable, since the phone is always kept in portrait mode.
    rotate_cw_degrees = rotation;

    // copy y plane into buffer
    for(int i = 0; i < inp_h; i++){
        // row_stride is in bytes
        for(int j = 0; j < yrow_stride ; j++){
            host_img_buf[i*yrow_stride + j] = y_ptr[(i*yrow_stride + j)*ypixel_stride];
        }
    }

    int uv_start_index = inp_h * yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for (int i = 0; i < (inp_h * inp_w) / 4; i++) {

        host_img_buf[uv_start_index + 2 * i] = v_ptr[i * uvpixel_stride];

        host_img_buf[uv_start_index + 1 + 2 * i] = u_ptr[i * uvpixel_stride];
    }
#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
    printTime(timespec_b, timespec_a, "copying image to array");
#endif

    // compression:
    // 1. write raw image to buffer
    // 2. compress on proxy device
    // 3. write compressed image buffer to remote
    // 4. decompress image on remote pthread
    // 5. run yolo on remote basic
    // 6. read detections back

    // milestone 2:
    // 7. post process on remote basic
    // 8. compress segments
    // 9. read segments back
    // 10. decompress segments
    // 11.? scale segments


    // the dnn_kernel will wait for this event to be done
    cl_event dnn_wait_event;

    cl_event enc_image_event;
    cl_event enc_y_event, enc_uv_event;
    cl_event dec_y_event, dec_uv_event;

    // do compression
    if (2 == device_index_copy && (1 == local_do_compression)) {

        status = clEnqueueWriteBuffer(commandQueue[1], img_buf[1], CL_FALSE, 0,
                                      img_buf_size, host_img_buf, 0, NULL, &enc_image_event);
        CHECK_AND_RETURN(status, "failed to write image to enc buffers");


        status = clEnqueueNDRangeKernel(commandQueue[1], enc_y_kernel, 2, NULL, enc_y_global,
                                        NULL, 1, &enc_image_event, &enc_y_event);
        CHECK_AND_RETURN(status, "failed to enqueue enc_y_kernel");

        status = clEnqueueNDRangeKernel(commandQueue[1], enc_uv_kernel, 2, NULL, enc_uv_global,
                                        NULL, 1, &enc_image_event, &enc_uv_event);
        CHECK_AND_RETURN(status, "failed to enqueue enc_uv_kernel");

        status = clEnqueueNDRangeKernel(commandQueue[3], dec_y_kernel, 2, NULL, dec_y_global,
                                        NULL, 1, &enc_y_event, &dec_y_event);
        CHECK_AND_RETURN(status, "failed to enqueue dec_y_kernel");

        // we have to wait for both since dec_y and dec_uv write to the same buffer and there is
        // no guarantee what happens if both dec_y and dec_uv write at the same time.
        cl_event dec_uv_wait_events[] = {enc_uv_event, dec_y_event};
        status = clEnqueueNDRangeKernel(commandQueue[3], dec_uv_kernel, 2, NULL, dec_uv_global,
                                        NULL, 2, dec_uv_wait_events, &dec_uv_event);
        CHECK_AND_RETURN(status, "failed to enqueue dec_uv_kernel");

        // move the intermediate buffers back to the phone after decompression.
        // Since we don't care about the contents, the latest state of the buffer is not moved
        // back from the remote.
        // https://man.opencl.org/clEnqueueMigrateMemObjects.html
        cl_mem migrate_bufs[] = {out_enc_y_buf, out_enc_uv_buf};
        status = clEnqueueMigrateMemObjects(commandQueue[1], 2, migrate_bufs,
                                            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1,
                                            &dec_uv_event, NULL);
        CHECK_AND_RETURN(status, "failed to migrate buffers back");

        dnn_wait_event = dec_uv_event;

    } else {
        // normal execution
        status = clEnqueueWriteBuffer(commandQueue[device_index_copy], img_buf[device_index_copy],
                                      CL_FALSE, 0,
                                      img_buf_size, host_img_buf,
                                      0, NULL, &enc_image_event);
        CHECK_AND_RETURN(status, "failed to write image to ocl buffers");

        dnn_wait_event = enc_image_event;
    }

    cl_event ndrange_event;
    status = clEnqueueNDRangeKernel(commandQueue[device_index_copy], dnn_kernel, 1, NULL,
                                    &global_size, &local_size, 1,
                                    &dnn_wait_event, &ndrange_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");

    cl_event read_event;
    status = clEnqueueReadBuffer(commandQueue[device_index_copy], out_buf[device_index_copy],
                                 CL_FALSE, 0,
                                 detection_count * sizeof(cl_int), detection_array, 1,
                                 &ndrange_event, &read_event);
    CHECK_AND_RETURN(status, "failed to read detection result buffer");

    cl_event postprocess_event = NULL, segment_event = NULL;
    if (do_segment_copy) {

        status = clEnqueueNDRangeKernel(commandQueue[device_index_copy], postprocess_kernel, 1,
                                        NULL,
                                        &global_size, &local_size, 1,
                                        &ndrange_event, &postprocess_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");

        status = clEnqueueReadBuffer(commandQueue[device_index_copy],
                                     postprocess_buf[device_index_copy], CL_FALSE, 0,
                                     seg_postprocess_count * sizeof(cl_char), segmentation_array, 1,
                                     &postprocess_event, &segment_event);
        CHECK_AND_RETURN(status, "failed to read segmentation postprocess result buffer");

    }

    status = clFinish(commandQueue[device_index_copy]);
    CHECK_AND_RETURN(status, "failed to clfinish");

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "total cl stuff");
#endif

    if (enable_profiling) {

        char formated_string[DATA_POINT_SIZE + 1];
        // clear the string
        strcpy(c_log_string, "\0");
        int chars_used;

        cl_ulong diff = getEventRuntime(enc_image_event);
        chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
        strncat(c_log_string, formated_string, chars_used);

#if defined(PRINT_PROFILE_TIME)
        __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing",
                            "writing took this long: %llu ms, %llu ns",
                            (diff / 1000000), diff % 1000000);
#endif

        diff = getEventRuntime(ndrange_event);
        chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
        strncat(c_log_string, formated_string, chars_used);

#if defined(PRINT_PROFILE_TIME)
        __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing",
                            "detections took this long: %llu ms, %llu ns",
                            (diff / 1000000), diff % 1000000);
#endif

        diff = getEventRuntime(read_event);
        chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
        strncat(c_log_string, formated_string, chars_used);

#if defined(PRINT_PROFILE_TIME)
        __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing",
                            "reading back took this long: %llu ms, %llu ns",
                            (diff / 1000000), diff % 1000000);
#endif

        if (do_segment_copy) {

            diff = getEventRuntime(postprocess_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);

#if defined(PRINT_PROFILE_TIME)
            __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing",
                                "post process took this long: %llu ms, %llu "
                                "ns", (diff / 1000000), diff % 1000000);
#endif

            diff = getEventRuntime(postprocess_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);

#if defined(PRINT_PROFILE_TIME)
            __android_log_print(ANDROID_LOG_WARN, LOGTAG "timing",
                                "reading segs took this long: %llu ms, "
                                "%llu ns", (diff / 1000000), diff % 1000000);
#endif

        }

        if (2 == device_index_copy && (1 == local_do_compression)) {
            diff = getEventRuntime(enc_y_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);

            diff = getEventRuntime(enc_uv_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);

            diff = getEventRuntime(dec_y_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);

            diff = getEventRuntime(dec_uv_event);
            chars_used = snprintf(formated_string, DATA_POINT_SIZE, "%llu, ", diff);
            strncat(c_log_string, formated_string, chars_used);
        }

        // finally add a new line to the end of it
        strncat(c_log_string, "\n", 1);
    }

    if (2 == device_index_copy && (1 == local_do_compression)) {
        clReleaseEvent(enc_y_event);
        clReleaseEvent(enc_uv_event);
        clReleaseEvent(dec_y_event);
        clReleaseEvent(dec_uv_event);
    }

    clReleaseEvent(enc_image_event);
    clReleaseEvent(ndrange_event);
    clReleaseEvent(read_event);

    if (NULL != postprocess_event) {
        clReleaseEvent(postprocess_event);
    }

    if (NULL != segment_event) {
        clReleaseEvent(segment_event);
    }

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_a);
    printTime(timespec_b, timespec_a, "printing messages");
#endif

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result, segmentation_array, JNI_FALSE);
//    __android_log_print(ANDROID_LOG_DEBUG, "DETECTION", "%d %d %d %d %d %d %d", result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6]);

#if defined(PRINT_PROFILE_TIME)
    clock_gettime(CLOCK_MONOTONIC, &timespec_b);
    printTime(timespec_a, timespec_b, "releasing buffers");
#endif

    return 0;
}

/**
 * return a string that contains log lines that can be written to a file.
 * This is a workaround to the fact that JNI makes strings immutable.
 * @param env
 * @param clazz
 * @return
 */
JNIEXPORT jstring JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getProfilingStats(JNIEnv *env,
                                                                         jclass clazz) {

    return env->NewStringUTF(c_log_string);
}

#ifdef __cplusplus
}
#endif

extern "C"
void pocl_remote_get_traffic_stats(uint64_t *out_buf, int server_num);

extern "C"
JNIEXPORT jobject JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getRemoteTrafficStats(JNIEnv *env,
                                                                             jclass clazz) {
    jclass c = env->FindClass("org/portablecl/poclaisademo/TrafficMonitor$DataPoint");
    assert (c != nullptr);

    jmethodID datapoint_constructor = env->GetMethodID(c, "<init>", "(JJJJJJ)V");
    assert (datapoint_constructor != nullptr);

    uint64_t buf[6];
    pocl_remote_get_traffic_stats(buf, 0);

    return env->NewObject(c, datapoint_constructor, (jlong) buf[0], (jlong) buf[1], (jlong) buf[2],
                          (jlong) buf[3], (jlong) buf[4], (jlong) buf[5]);
}
