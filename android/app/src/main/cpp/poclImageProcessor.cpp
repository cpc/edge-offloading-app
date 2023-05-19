

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

static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;

const char *pocl_onnx_blob = NULL;
size_t pocl_onnx_blob_size = 0;

/**
 * setup and create the objects needed for repeated PoCL calls.
 * PoCL can be configured by setting environment variables.
 * @param env
 * @param clazz
 * @return
 */
JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_initPoclImageProcessor(JNIEnv *env,
                                                                              jclass clazz, jobject jAssetManager) {

    // TODO: this is racy
    if (pocl_onnx_blob == nullptr) {
        AAssetManager *assetManager = AAssetManager_fromJava(env, jAssetManager);
        if (assetManager == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager", "Failed to get asset manager");
            return -1;
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
            return -1;
        }
        pocl_onnx_blob = tmp;
        pocl_onnx_blob_size = num_bytes;
        __android_log_print(ANDROID_LOG_DEBUG, "NDK_Asset_Manager", "ONNX blob read successfully");
    }

    cl_platform_id platform;
    cl_device_id device;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    CHECK_AND_RETURN(status, "getting device id failed");

    // some info
    char result_array[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);

    clGetDeviceInfo(device, CL_DEVICE_VERSION, 256 * sizeof(char), result_array, NULL);
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_VERSION: %s", result_array);

    clGetDeviceInfo(device, CL_DRIVER_VERSION, 256 * sizeof(char), result_array, NULL);
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DRIVER_VERSION: %s", result_array);

    clGetDeviceInfo(device, CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
    __android_log_print(ANDROID_LOG_INFO, LOGTAG, "CL_DEVICE_NAME: %s", result_array);

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    context = clCreateContext(cps, 1, &device, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    commandQueue = clCreateCommandQueue(context, device, 0, &status);
    CHECK_AND_RETURN(status, "creating command queue failed");

    char* kernel_name =  "pocl.dnn.detection.u8";
    program = clCreateProgramWithBuiltInKernels(context, 1, &device, kernel_name ,
                                                &status);
    CHECK_AND_RETURN(status, "creation of program failed");

    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "building of program failed");

    kernel = clCreateKernel(program, kernel_name, &status);
    CHECK_AND_RETURN(status, "creating kernel failed");

    clReleaseDevice(device);

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

    if (commandQueue != NULL) {
        clReleaseCommandQueue(commandQueue);
    }

    if (context != NULL) {
        clReleaseContext(context);
    }

    if (kernel != NULL) {
        clReleaseKernel(kernel);
    }

    if (program != NULL) {
        clReleaseProgram(program);
    }

    if (pocl_onnx_blob != nullptr) {
        delete[] pocl_onnx_blob;
        pocl_onnx_blob = nullptr;
        pocl_onnx_blob_size = 0;
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
                                                                           jclass clazz, jint width,
                                                                           jint height, jobject y,
                                                                           jint yrow_stride,
                                                                           jint ypixel_stride,
                                                                           jobject u, jobject v,
                                                                           jint uvrow_stride,
                                                                           jint uvpixel_stride,
                                                                           jintArray result) {

    cl_int	status;
    // todo: look into if iscopy=true works on android
    cl_int * result_array = env->GetIntArrayElements(result, 0);

    int tot_pixels = width * height;

    cl_char *y_ptr = (cl_char *) env->GetDirectBufferAddress(y);
    cl_char *u_ptr = (cl_char *) env->GetDirectBufferAddress(u);
    cl_char *v_ptr = (cl_char *) env->GetDirectBufferAddress(v);

//    __android_log_print(ANDROID_LOG_INFO, "native test", " y address %p", y_ptr);
//    __android_log_print(ANDROID_LOG_INFO, "native test", " u address %p", u_ptr);
//    __android_log_print(ANDROID_LOG_INFO, "native test", " v address %p", v_ptr);

    // yuv420 so 6 bytes for every 4 pixels
    int img_buf_size = (tot_pixels * 3) / 2 ;
    cl_mem img_buf = clCreateBuffer(context, (CL_MEM_READ_ONLY),
                                    img_buf_size, NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the image buffer");

    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_count * sizeof(cl_int),
                                    NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the output buffer");

    cl_char * host_img_ptr = (cl_char *) clEnqueueMapBuffer(commandQueue, img_buf, CL_TRUE,
                                                            CL_MAP_WRITE, 0,
                                                            img_buf_size,
                                                            0, NULL,
                                                            NULL,
                                                            &status);
    CHECK_AND_RETURN(status, "failed to map the image buffer");

    // copy y plane into buffer
    for(int i = 0; i < height; i++){
        // row_stride is in bytes
        for(int j = 0; j < yrow_stride ; j++){
            host_img_ptr[i*j] = y_ptr[i*j];
        }
    }

    int uv_start_index = height*yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for(int i = 0; i < (height * width)/4; i++){

        host_img_ptr[uv_start_index + i] = v_ptr[i*uvpixel_stride];

        host_img_ptr[uv_start_index + 1 + i] = u_ptr[i*uvpixel_stride];
    }

    status = clEnqueueUnmapMemObject(commandQueue, img_buf, host_img_ptr, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "failed to unmap the image buffer");

    // todo: call clsetkernelarg for each of the variables
//
//    status = clSetKernelArg(clKernel, 0, sizeof(cl_int), (void *)&n);
//    status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&A_obj);

    size_t local_size = 1;
    size_t global_size = 1;

//    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
//                           &global_size, &local_size, 0,
//                           NULL,NULL );
//    CHECK_AND_RETURN(status, "failed to enqueue ND range kernel");
//
//    status = clEnqueueReadBuffer(commandQueue, out_buf, CL_TRUE, 0,
//                                 out_count * sizeof(cl_int),result_array, 0,
//                                 NULL, NULL);
//    CHECK_AND_RETURN(status, "failed to read result buffer");

    // a test array
    result_array[0] = 1;
    result_array[1] = 18; // sheep
    result_array[2] = 1063948516;
    result_array[3] = 100;
    result_array[4] = 200;
    result_array[5] = 250;
    result_array[6] = 300;

    // commit the results back
    env->ReleaseIntArrayElements(result,result_array, 0);

    clReleaseMemObject(img_buf);
    clReleaseMemObject(out_buf);

    return 0;
}

#ifdef __cplusplus
}
#endif
