

//#include <rename_opencl.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <rename_opencl.h>
#include <CL/cl.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <string.h>
#include <stdlib.h>
#include <jni.h>
#include <android/bitmap.h>
#include "sharedUtils.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOGTAG "poclimageprocessor"
#define MAX_DETECTIONS 10

int out_count = 1 + MAX_DETECTIONS * 6;

#define NUM_CL_DEVICES 3
static cl_context context = NULL;
static cl_command_queue commandQueue[NUM_CL_DEVICES] = {nullptr};
static cl_program program = NULL;
static cl_kernel kernel = NULL;

// kernel execution loop related things
static size_t tot_pixels;
static size_t img_buf_size;
static cl_int inp_w;
static cl_int inp_h;
static cl_int rotate_cw_degrees;
static cl_int inp_format;
static cl_mem img_buf;
static cl_mem out_buf;
static cl_mem out_mask_buf;
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
    char* kernel_name =  "pocl.dnn.detection.u8";
    program = clCreateProgramWithBuiltInKernels(context, 2, inference_devices, kernel_name, &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    status = clBuildProgram(program, 2, inference_devices, nullptr, nullptr, nullptr);
    CHECK_AND_RETURN(status, "building of program failed");

    kernel = clCreateKernel(program, kernel_name, &status);
    CHECK_AND_RETURN(status, "creating kernel failed");

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
    img_buf = clCreateBuffer(context, (CL_MEM_READ_ONLY),
                                    img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the image buffer");

    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_count * sizeof(cl_int),
                                    NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    out_mask_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tot_pixels * MAX_DETECTIONS * sizeof(cl_char) / 4,
                                         NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

    // buffer to copy image data to
    host_img_buf = (cl_uchar*) malloc(img_buf_size);

    int arg_idx = 0;
    status = clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &img_buf);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &inp_w);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &inp_h);
    status |=
            clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &rotate_cw_degrees);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &inp_format);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &out_buf);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &out_mask_buf);

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
    }

    if (context != nullptr) {
        clReleaseContext(context);
        context = nullptr;
    }

    if (kernel != nullptr) {
        clReleaseKernel(kernel);
        kernel = nullptr;
    }

    if (program != nullptr) {
        clReleaseProgram(program);
        program = nullptr;
    }

    if (nullptr != img_buf){
        clReleaseMemObject(img_buf);
        img_buf = nullptr;
    }

    if (nullptr != out_buf){
        clReleaseMemObject(out_buf);
        out_buf = nullptr;
    }

    if (nullptr != out_mask_buf){
        clReleaseMemObject(out_mask_buf);
        out_mask_buf = nullptr;
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
                                                                           jintArray result) {

    if (!kernel) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG, "poclProcessYUVImage called but OpenCL kernel does not exist!");
        return 1;
    }

    cl_int	status;
    // todo: look into if iscopy=true works on android
    cl_int * result_array = env->GetIntArrayElements(result, JNI_FALSE);

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
    status = clEnqueueWriteBuffer(commandQueue[device_index], img_buf,
                                  CL_FALSE, 0,
                                  img_buf_size, host_img_buf,
                                  0, NULL, &write_event );
    CHECK_AND_RETURN(status, "failed to write image to ocl buffers");

    cl_event ndrange_event;
    status = clEnqueueNDRangeKernel(commandQueue[device_index], kernel, 1, NULL,
                           &global_size, &local_size, 1,
                           &write_event,&ndrange_event );
    CHECK_AND_RETURN(status, "failed to enqueue ND range kernel");

    status = clEnqueueReadBuffer(commandQueue[device_index], out_buf, CL_FALSE, 0,
                                 out_count * sizeof(cl_int),result_array, 1,
                                 &ndrange_event, NULL);
    CHECK_AND_RETURN(status, "failed to read result buffer");

    clFinish(commandQueue[device_index]);

    // commit the results back
    env->ReleaseIntArrayElements(result,result_array, JNI_FALSE);
//    __android_log_print(ANDROID_LOG_DEBUG, "DETECTION", "%d %d %d %d %d %d %d", result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6]);

    return 0;
}

#ifdef __cplusplus
}
#endif
