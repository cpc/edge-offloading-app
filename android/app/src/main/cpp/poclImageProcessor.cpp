

//#include <rename_opencl.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <rename_opencl.h>
#include <CL/cl.h>
#include <CL/cl_ext_pocl.h>
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
static clCreateProgramWithBuiltInOnnxKernelsPOCL_fn ext_clCreateProgramWithBuiltInOnnxKernelsPOCL = nullptr;

cl_program createClProgramFromOnnxAsset(JNIEnv *env, jobject jAssetManager, const char *filename, cl_int num_devices, cl_device_id *devices, const char *kernel_names, cl_int *status)
{

    AAssetManager *assetManager = AAssetManager_fromJava(env, jAssetManager);
    if (assetManager == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager", "Failed to get asset manager");
        return nullptr;
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
        return nullptr;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "NDK_Asset_Manager", "ONNX blob read successfully");

    size_t blob_size = num_bytes;
    cl_program p = ext_clCreateProgramWithBuiltInOnnxKernelsPOCL(context, num_devices, devices, kernel_names, (const char **)&tmp, &blob_size, status);

    delete[] tmp;

    return p;
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
                                                                              jclass clazz, jobject jAssetManager) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    ext_clCreateProgramWithBuiltInOnnxKernelsPOCL =
            reinterpret_cast<clCreateProgramWithBuiltInOnnxKernelsPOCL_fn>(
                    clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithBuiltInOnnxKernelsPOCL"));


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
    program = createClProgramFromOnnxAsset(env, jAssetManager, "yolov8n-seg.onnx", 1, &device, kernel_name, &status);
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

    if (commandQueue != nullptr) {
        clReleaseCommandQueue(commandQueue);
        commandQueue = nullptr;
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
                                                                           jint height, jint rotation,
                                                                           jobject y,
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

    cl_mem out_mask_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tot_pixels * MAX_DETECTIONS * sizeof(cl_char) / 4,
                                    NULL, &status);
    CHECK_AND_RETURN(status, "failed to create the segmentation mask buffer");

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
            host_img_ptr[i*yrow_stride + j] = y_ptr[(i*yrow_stride + j)*ypixel_stride];
        }
    }

    int uv_start_index = height*yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for(int i = 0; i < (height * width)/4; i++){

        host_img_ptr[uv_start_index + 2*i] = v_ptr[i*uvpixel_stride];

        host_img_ptr[uv_start_index + 1 + 2*i] = u_ptr[i*uvpixel_stride];
    }

    status = clEnqueueUnmapMemObject(commandQueue, img_buf, host_img_ptr, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "failed to unmap the image buffer");

    int rotate_cw_degrees = rotation;
    int inp_format = 1; // 0 - RGB
    // 1 - YUV420 NV21 Android (interleaved U/V)
    // 2 - YUV420 (U/V separate)
    int inp_w = width;
    int inp_h = height;

    int arg_idx = 0;
    status = clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &img_buf);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &inp_w);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &inp_h);
    status |=
        clSetKernelArg(kernel, arg_idx++, sizeof(int), &rotate_cw_degrees);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &inp_format);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &out_buf);
    status |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &out_mask_buf);
    size_t local_size = 1;
    size_t global_size = 1;

    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                           &global_size, &local_size, 0,
                           NULL,NULL );
    CHECK_AND_RETURN(status, "failed to enqueue ND range kernel");

    status = clEnqueueReadBuffer(commandQueue, out_buf, CL_TRUE, 0,
                                 out_count * sizeof(cl_int),result_array, 0,
                                 NULL, NULL);
    CHECK_AND_RETURN(status, "failed to read result buffer");

    // commit the results back
    env->ReleaseIntArrayElements(result,result_array, JNI_FALSE);
    __android_log_print(ANDROID_LOG_DEBUG, "DETECTION", "%d %d %d %d %d %d %d", result_array[0],result_array[1],result_array[2],result_array[3],result_array[4],result_array[5],result_array[6]);

    clReleaseMemObject(img_buf);
    clReleaseMemObject(out_buf);
    clReleaseMemObject(out_mask_buf);

    return 0;
}

#ifdef __cplusplus
}
#endif
