

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

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

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
static int do_segment = 1;

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

#define NUM_CL_DEVICES 3
static cl_context context = NULL;
static cl_command_queue commandQueue[NUM_CL_DEVICES] = {nullptr};
static cl_program program = NULL;
static cl_kernel dnn_kernel = NULL;
static cl_kernel postprocess_kernel = NULL;

// kernel execution loop related things
static size_t tot_pixels;
static size_t img_buf_size;
static cl_int inp_w;
static cl_int inp_h;
static cl_int rotate_cw_degrees;
static cl_int inp_format;
static cl_mem img_buf[3] = {nullptr, nullptr, nullptr};
static cl_mem out_buf[3] = {nullptr, nullptr, nullptr};
static cl_mem out_mask_buf[3] = {nullptr, nullptr, nullptr};
static cl_mem postprocess_buf[3] = {nullptr, nullptr, nullptr};
static cl_uchar* host_img_buf = nullptr;
static size_t local_size;
static size_t global_size;

// Global variables for smuggling our blob into PoCL so we can pretend it is a builtin kernel.
// Please don't ever actually do this in production code.
const char *pocl_onnx_blob = NULL;
uint64_t pocl_onnx_blob_size = 0;

bool smuggleONNXAsset(JNIEnv *env, jobject jAssetManager, const char *filename)
{
    AAssetManager *assetManager = AAssetManager_fromJava(env, jAssetManager);
    if (assetManager == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager", "Failed to get asset manager");
        return false;
    }

    AAsset *a = AAssetManager_open(assetManager, "yolov8n-seg.onnx", AASSET_MODE_STREAMING);
    auto num_bytes = AAsset_getLength(a);
    char * tmp = new char[num_bytes+1];
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
    char * tmp = (char *)pocl_onnx_blob;
    pocl_onnx_blob_size = 0;
    pocl_onnx_blob = nullptr;
    delete[] tmp;
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
                                                                              jclass clazz, jobject jAssetManager,
                                                                              jint width, jint height) {
    cl_platform_id platform;
    cl_device_id devices[NUM_CL_DEVICES];
    cl_uint devices_found;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, NUM_CL_DEVICES, devices, &devices_found);
    CHECK_AND_RETURN(status, "getting device id failed");
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "Platform has %d devices", devices_found);
    assert(devices_found == NUM_CL_DEVICES);

    // some info
    char result_array[256];
    for (unsigned i = 0; i < NUM_CL_DEVICES; ++i) {
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

    context = clCreateContext(cps, NUM_CL_DEVICES, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    for (unsigned i = 0; i < NUM_CL_DEVICES; ++i) {
        commandQueue[i] = clCreateCommandQueue(context, devices[i], 0, &status);
        CHECK_AND_RETURN(status, "creating command queue failed");
    }

    bool smuggling_ok = smuggleONNXAsset(env, jAssetManager, "yolov8n-seg.onnx");
    assert(smuggling_ok);

    cl_device_id inference_devices[] = {devices[0], devices[2]};
    // Only create builtin kernel for basic and remote devices
    std::string dnn_kernel_name = "pocl.dnn.detection.u8";
    std::string postprocess_kernel_name = "pocl.dnn.segmentation_postprocess.u8";
    std::string kernel_names = dnn_kernel_name + ";" + postprocess_kernel_name;
    program = clCreateProgramWithBuiltInKernels(context, 2, inference_devices, kernel_names.c_str(), &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    status = clBuildProgram(program, 2, inference_devices, nullptr, nullptr, nullptr);
    CHECK_AND_RETURN(status, "building of program failed");

    dnn_kernel = clCreateKernel(program, dnn_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating dnn kernel failed");

    postprocess_kernel = clCreateKernel(program, postprocess_kernel_name.c_str(), &status);
    CHECK_AND_RETURN(status, "creating postprocess kernel failed");

    destroySmugglingEvidence();

    for (unsigned i = 0; i < NUM_CL_DEVICES; ++i) {
        clReleaseDevice(devices[i]);
    }

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

    out_mask_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, tot_pixels * MAX_DETECTIONS * sizeof(cl_char) / 4,
                                         NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    // RGBA:
    postprocess_buf[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tot_pixels * 4 * sizeof(cl_char) / 4,
                                         NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation preprocessing buffer");

    img_buf[2] = clCreateBuffer(context, (CL_MEM_READ_ONLY),
                                img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the image buffer");

    out_buf[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
                                NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    out_mask_buf[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tot_pixels * MAX_DETECTIONS *
                                                                 sizeof(cl_char) / 4,
                                     NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    postprocess_buf[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tot_pixels * 4 * sizeof(cl_char) / 4,
                                         NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation preprocessing buffer");

    // buffer to copy image data to
    host_img_buf = (cl_uchar*) malloc(img_buf_size);

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

    global_size = 1;
    local_size = 1;
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

    for (unsigned i = 0; i < NUM_CL_DEVICES; ++i) {
        if (commandQueue != nullptr) {
            clReleaseCommandQueue(commandQueue[i]);
            commandQueue[i] = nullptr;
        }

        if (nullptr != img_buf[i]){
            clReleaseMemObject(img_buf[i]);
            img_buf[i] = nullptr;
        }

        if (nullptr != out_buf[i]){
            clReleaseMemObject(out_buf[i]);
            out_buf[i] = nullptr;
        }

        if (nullptr != out_mask_buf){
            clReleaseMemObject(out_mask_buf[i]);
            out_mask_buf[i] = nullptr;
        }

        if (nullptr != postprocess_buf){
            clReleaseMemObject(postprocess_buf[i]);
            postprocess_buf[i] = nullptr;
        }
    }

    if (context != nullptr) {
        clReleaseContext(context);
        context = nullptr;
    }

    if (dnn_kernel != nullptr) {
        clReleaseKernel(dnn_kernel);
        dnn_kernel = nullptr;
    }

    if (postprocess_kernel != nullptr) {
        clReleaseKernel(postprocess_kernel);
        postprocess_kernel = nullptr;
    }

    if (program != nullptr) {
        clReleaseProgram(program);
        program = nullptr;
    }
    
    if(nullptr != host_img_buf){
        free(host_img_buf);
        host_img_buf = nullptr;
    }

    return 0;
}

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
                                                                           jint rotation, jobject y,
                                                                           jint yrow_stride,
                                                                           jint ypixel_stride,
                                                                           jobject u, jobject v,
                                                                           jint uvrow_stride,
                                                                           jint uvpixel_stride,
                                                                           jintArray detection_result,
                                                                           jbyteArray segmentation_result) {

    if (!dnn_kernel) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG, "poclProcessYUVImage called but DNN kernel does not exist!");
        return 1;
    }

    if (!postprocess_kernel) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG, "poclProcessYUVImage called but postprocess kernel does not exist!");
        return 1;
    }

    cl_int	status;

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_buf[device_index]);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_buf[device_index]);
    status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &out_mask_buf[device_index]);
    CHECK_AND_RETURN(status, "could not assign buffers to DNN kernel");

    status = clSetKernelArg(postprocess_kernel, 0, sizeof(cl_mem), &out_buf[device_index]);
    status |= clSetKernelArg(postprocess_kernel, 1, sizeof(cl_mem), &out_mask_buf[device_index]);
    status |= clSetKernelArg(postprocess_kernel, 2, sizeof(cl_mem), &postprocess_buf[device_index]);
    CHECK_AND_RETURN(status, "could not assign buffers to postprocess kernel");

    // todo: look into if iscopy=true works on android
    cl_int * detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    cl_char * segmentation_array = env->GetByteArrayElements(segmentation_result, JNI_FALSE);

    cl_char *y_ptr = (cl_char *) env->GetDirectBufferAddress(y);
    cl_char *u_ptr = (cl_char *) env->GetDirectBufferAddress(u);
    cl_char *v_ptr = (cl_char *) env->GetDirectBufferAddress(v);

    rotate_cw_degrees = rotation;

    // copy y plane into buffer
    for(int i = 0; i < inp_h; i++){
        // row_stride is in bytes
        for(int j = 0; j < yrow_stride ; j++){
            host_img_buf[i*yrow_stride + j] = y_ptr[(i*yrow_stride + j)*ypixel_stride];
        }
    }

    int uv_start_index = inp_h*yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for(int i = 0; i < (inp_h * inp_w)/4; i++){

        host_img_buf[uv_start_index + 2*i] = v_ptr[i*uvpixel_stride];

        host_img_buf[uv_start_index + 1 + 2*i] = u_ptr[i*uvpixel_stride];
    }

    cl_event write_event;
    status = clEnqueueWriteBuffer(commandQueue[device_index], img_buf[device_index],
                                  CL_FALSE, 0,
                                  img_buf_size, host_img_buf,
                                  0, NULL, &write_event );
    CHECK_AND_RETURN(status, "failed to write image to ocl buffers");

    cl_event ndrange_event;
    status = clEnqueueNDRangeKernel(commandQueue[device_index], dnn_kernel, 1, NULL,
                           &global_size, &local_size, 1,
                           &write_event, &ndrange_event);
    CHECK_AND_RETURN(status, "failed to enqueue ND range DNN kernel");

    status = clEnqueueReadBuffer(commandQueue[device_index], out_buf[device_index], CL_FALSE, 0,
                                 detection_count * sizeof(cl_int),detection_array, 1,
                                 &ndrange_event, NULL);
    CHECK_AND_RETURN(status, "failed to read detection result buffer");

    if (do_segment) {
        cl_event postprocess_event;
        status = clEnqueueNDRangeKernel(commandQueue[device_index], postprocess_kernel, 1, NULL,
                               &global_size, &local_size, 1,
                               &ndrange_event, &postprocess_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range postprocess kernel");

        status = clEnqueueReadBuffer(commandQueue[device_index], postprocess_buf[device_index], CL_FALSE, 0,
                                     seg_postprocess_count * sizeof(cl_char), segmentation_array, 1,
                                     &postprocess_event, NULL);
        CHECK_AND_RETURN(status, "failed to read segmentation postprocess result buffer");

        clReleaseEvent(postprocess_event);
    }

    clReleaseEvent(write_event);
    clReleaseEvent(ndrange_event);
    status = clFinish(commandQueue[device_index]);
    CHECK_AND_RETURN(status, "failed to clfinish");

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result, segmentation_array, JNI_FALSE);
//    __android_log_print(ANDROID_LOG_DEBUG, "DETECTION", "%d %d %d %d %d %d %d", result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6]);

    return 0;
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
