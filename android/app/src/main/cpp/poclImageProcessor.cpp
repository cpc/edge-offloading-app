

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

    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_out_count * sizeof(cl_int),
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
                                                                           jintArray detection_result,
                                                                           jbyteArray segmentation_result) {

    if (!kernel) {
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG, "poclProcessYUVImage called but OpenCL kernel does not exist!");
        return 1;
    }

    cl_int	status;
    // todo: look into if iscopy=true works on android
    cl_int * detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    cl_char * segmentation_array = env->GetByteArrayElements(segmentation_result, JNI_FALSE);
    std::vector<cl_char> segmentation_out(MAX_DETECTIONS * MASK_W * MASK_H);

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
                                 detection_count * sizeof(cl_int),detection_array, 1,
                                 &ndrange_event, NULL);
    CHECK_AND_RETURN(status, "failed to read detection result buffer");

    if (do_segment) {
        status = clEnqueueReadBuffer(commandQueue[device_index], out_mask_buf, CL_FALSE, 0,
                                     segmentation_count * sizeof(cl_char),segmentation_out.data(), 1,
                                     &ndrange_event, NULL);
        CHECK_AND_RETURN(status, "failed to read segmentation result buffer");
    }

    clReleaseEvent(write_event);
    clReleaseEvent(ndrange_event);
    clFinish(commandQueue[device_index]);

    if (do_segment) {
        int num_detections = detection_array[0];
        cv::Mat color_mask = cv::Mat::zeros(MASK_H, MASK_W, CV_8UC4);

        for (int i = 0; i < num_detections; ++i) {
            cl_int class_id = detection_array[1 + 6 * i];
            int color_int = colors[class_id];
            cl_uchar* channels = reinterpret_cast<cl_uchar*>(&color_int);
            cv::Scalar color = cv::Scalar(channels[0], channels[1], channels[2], channels[3]) / 2;

            __android_log_print(ANDROID_LOG_DEBUG, "SEGMENTATION", "prebox: %d %d %d %d",
                detection_array[1 + 6 * i + 2],
                detection_array[1 + 6 * i + 3],
                detection_array[1 + 6 * i + 4],
                detection_array[1 + 6 * i + 5]);

            int box_x = (int)((float)(detection_array[1 + 6 * i + 2]) / 480 * MASK_W);
            int box_y = (int)((float)(detection_array[1 + 6 * i + 3]) / 640 * MASK_H);
            int box_w = (int)((float)(detection_array[1 + 6 * i + 4]) / 480 * MASK_W);
            int box_h = (int)((float)(detection_array[1 + 6 * i + 5]) / 640 * MASK_H);

            box_x = std::min(std::max(box_x, 0), MASK_W);
            box_y = std::min(std::max(box_y, 0), MASK_H);
            box_w = std::min(box_w, MASK_W - box_x);
            box_h = std::min(box_h, MASK_H - box_y);

            __android_log_print(ANDROID_LOG_DEBUG, "SEGMENTATION", "box: %d %d %d %d",
                box_x, box_y, box_w, box_h);

            if (box_w > 0 && box_h > 0) {
                cv::Mat raw_mask(MASK_H, MASK_W, CV_8UC1,
                                 reinterpret_cast<cl_uchar*>(segmentation_out.data() + i * MASK_W * MASK_H));
                cv::Rect roi(box_x, box_y, box_w, box_h);
                cv::Mat raw_mask_roi = cv::Mat::zeros(MASK_H, MASK_W, CV_8UC1);
                raw_mask(roi).copyTo(raw_mask_roi(roi));
                color_mask.setTo(color, raw_mask_roi);
            }
        }

        memcpy(segmentation_array, reinterpret_cast<cl_char*>(color_mask.data), MASK_W*MASK_H*4);
    }


//    if (num_detections > 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, "SEGMENTATION", "Num detections: %d", num_detections);
//        for (int y = 0; y < MASK_H; ++y) {
//            for (int x = 0; x < MASK_W; ++x) {
//                int i = y * MASK_W + x;
//                __android_log_print(ANDROID_LOG_DEBUG, "SEGMENTATION", "(%3d, %3d): %4d %4d",
//                                    x, y, segmentation_out[i], *reinterpret_cast<cl_uchar *>(&segmentation_out[i]));
//            }
//        }
//
//        while (true) {};
//    }

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
