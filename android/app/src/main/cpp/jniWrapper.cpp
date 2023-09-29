//
// Created by rabijl on 8/22/23.
//

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <assert.h>
//#include <file_descriptor_jni.h>
#include <jni.h>
#include "poclImageProcessor.h"
#include <stdlib.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif

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

    AAsset *a = AAssetManager_open(assetManager, filename, AASSET_MODE_STREAMING);
    auto num_bytes = AAsset_getLength(a);
    char *tmp = new char[num_bytes + 1];
    tmp[num_bytes] = 0;
    int read_bytes = AAsset_read(a, tmp, num_bytes);
    AAsset_close(a);
    if (read_bytes != num_bytes) {
        delete[] tmp;
        __android_log_print(ANDROID_LOG_ERROR, "NDK_Asset_Manager",
                            "Failed to read asset contents");
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
 * setup and create the objects needed for repeated PoCL calls.
 * PoCL can be configured by setting environment variables.
 * @param env
 * @param clazz
 * @return
 */
JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_initPoclImageProcessor(JNIEnv *env,
                                                                              jclass clazz,
                                                                              jint config_flags,
                                                                              jobject jAssetManager,
                                                                              jint width,
                                                                              jint height,
                                                                              jint fd) {

    bool smuggling_ok = smuggleONNXAsset(env, jAssetManager, "yolov8n-seg.onnx");
    assert(smuggling_ok);

    size_t src_size;
    char *codec_sources = read_file(env, jAssetManager, "kernels/copy.cl", &src_size);
    assert((nullptr != codec_sources) && "could not read sources");

    jint res = initPoclImageProcessor(width, height, config_flags, codec_sources, src_size, fd);

    destroySmugglingEvidence();

    free(codec_sources);

    return res;
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

    return destroy_pocl_image_processor();
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
                                                                           jint do_segment,
                                                                           jint do_compression,
                                                                           jint quality,
                                                                           jint rotation,
                                                                           jintArray detection_result,
                                                                           jbyteArray segmentation_result,
                                                                           jobject plane0,
                                                                           jint row_stride0,
                                                                           jint pixel_stride0,
                                                                           jobject plane1,
                                                                           jint row_stride1,
                                                                           jint pixel_stride1,
                                                                           jobject plane2,
                                                                           jint row_stride2,
                                                                           jint pixel_stride2,
                                                                           jlong image_timestamp) {
    // todo: look into if iscopy=true works on android
    int32_t *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    // pocl returns segmentation result in uint8_t, but jbyte is int8_t
    uint8_t *segmentation_array = reinterpret_cast<uint8_t *>(env->GetByteArrayElements(
            segmentation_result, JNI_FALSE));

    image_data_t image_data;
    image_data.type = YUV_DATA_T;
    image_data.data.yuv.planes[0] = (uint8_t *) env->GetDirectBufferAddress(plane0);
    image_data.data.yuv.planes[1] = (uint8_t *) env->GetDirectBufferAddress(plane1);
    image_data.data.yuv.planes[2] = (uint8_t *) env->GetDirectBufferAddress(plane2);
    image_data.data.yuv.pixel_strides[0] = pixel_stride0;
    image_data.data.yuv.pixel_strides[1] = pixel_stride1;
    image_data.data.yuv.pixel_strides[2] = pixel_stride2;
    image_data.data.yuv.row_strides[0] = row_stride0;
    image_data.data.yuv.row_strides[1] = row_stride1;
    image_data.data.yuv.row_strides[2] = row_stride2;

    int res = poclProcessImage(device_index, do_segment, (compression_t) do_compression, quality,
                               rotation, detection_array, segmentation_array, image_data,
                               image_timestamp);

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result,
                                  reinterpret_cast<jbyte *>(segmentation_array), JNI_FALSE);

    return res;
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_poclProcessJPEGImage(JNIEnv *env,
                                                                            jclass clazz,
                                                                            jint device_index,
                                                                            jint do_segment,
                                                                            jint do_compression,
                                                                            jint quality,
                                                                            jint rotation,
                                                                            jintArray detection_result,
                                                                            jbyteArray segmentation_result,
                                                                            jobject data,
                                                                            jint size,
                                                                            jlong image_timestamp) {

    int32_t *detection_array = env->GetIntArrayElements(detection_result, JNI_FALSE);
    // pocl returns segmentation result in uint8_t, but jbyte is int8_t
    uint8_t *segmentation_array = reinterpret_cast<uint8_t *>(env->GetByteArrayElements(
            segmentation_result, JNI_FALSE));

    image_data_t image_data;
    image_data.type = JPEG_DATA_T;
    image_data.data.jpeg.data = (uint8_t *) env->GetDirectBufferAddress(data);
    image_data.data.jpeg.capacity = size;

    int res = poclProcessImage(device_index, do_segment, (compression_t) do_compression, quality,
                               rotation, detection_array, segmentation_array, image_data,
                               image_timestamp);

    // commit the results back
    env->ReleaseIntArrayElements(detection_result, detection_array, JNI_FALSE);
    env->ReleaseByteArrayElements(segmentation_result,
                                  reinterpret_cast<jbyte *>(segmentation_array), JNI_FALSE);

    return res;
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
    return env->NewStringUTF(get_c_log_string_pocl());
}

/**
 * Function to return the header for the csv when logging.
 * @param env
 * @param clazz
 * @return string with names of each profiling stat
 */
JNIEXPORT jstring JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getCSVHeader(JNIEnv *env, jclass clazz) {
    return env->NewStringUTF(CSV_HEADER);
}

/**
 * return a byte array with the results that can be directly written to a file instead of a
 * string of which the bytes will need to be gotten for the streamwriter.
 * @param env
 * @param clazz
 * @return new jbytearray with a log line
 */
JNIEXPORT jbyteArray JNICALL
Java_org_portablecl_poclaisademo_JNIPoclImageProcessor_getProfilingStatsbytes(JNIEnv *env,
                                                                              jclass clazz) {
    char *c_log_string = get_c_log_string_pocl();
    auto c_str_leng = (jsize) strlen(c_log_string);
    jbyteArray res = env->NewByteArray(c_str_leng);
    env->SetByteArrayRegion(res, 0, c_str_leng, (jbyte *) c_log_string);
    return res;
}

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

#ifdef __cplusplus
}
#endif