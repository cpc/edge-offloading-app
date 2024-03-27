//
// Created by rabijl on 3.4.2023.
//


#include <android/log.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <android/log.h>
#include <assert.h>
#include "jni_utils.h"


#ifdef __cplusplus
extern "C" {
#endif

#define LOGTAG "jni_utils"

void
pocl_remote_get_traffic_stats(uint64_t *out_buf, int server_num);

/**
 * set the environment variables for native code
 * @param env
 * @param thiz
 * @param key
 * @param value
 */
JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_JNIutils_setNativeEnv(JNIEnv *env, jclass thiz, jstring key,
                                                       jstring value) {

    char *c_key = (char *) env->GetStringUTFChars(key, 0);
    char *c_value = (char *) env->GetStringUTFChars(value, 0);
    __android_log_print(ANDROID_LOG_INFO, "Native utils", "setting env variable: %s : %s", c_key,
                        c_value);

    setenv(c_key, c_value, 1);

}

bool
put_asset_in_local_storage(JNIEnv *env, jobject jAssetManager, const char *file_name) {

    // TODO: make sure that there is enough local storage for the onnx file
    // copy the asset to the local storage so that opencv can find the onnx file
    char write_file[128];
    sprintf(write_file, "/data/user/0/org.portablecl.poclaisademo/files/%s", file_name);
    if (access(write_file, F_OK) == 0) {
        __android_log_print(ANDROID_LOG_INFO, "jniWrapperV2", "%s exists", file_name);
        return true;
    }

    FILE *write_ptr = fopen(write_file, "w");
    if (NULL == write_ptr) {
        __android_log_print(ANDROID_LOG_ERROR, "jniWrapperV2",
                            "fopen errno: %s", strerror(errno));
        return false;
    }

    AAssetManager *assetManager = AAssetManager_fromJava(env, jAssetManager);
    if (assetManager == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "jniWrapperV2", "Failed to get asset manager");
        return false;
    }

    AAsset *a = AAssetManager_open(assetManager, file_name, AASSET_MODE_STREAMING);
    auto num_bytes = AAsset_getLength(a);
    char *asset_data = (char *) malloc((num_bytes + 1) * sizeof(char));
    asset_data[num_bytes] = 0;
    int read_bytes = AAsset_read(a, asset_data, num_bytes);
    AAsset_close(a);

    if (read_bytes != num_bytes) {
        fclose(write_ptr);
        free(asset_data);
        __android_log_print(ANDROID_LOG_ERROR, "jniWrapperV2",
                            "Failed to read asset contents");
        return false;
    }

    fwrite(asset_data, 1, num_bytes, write_ptr);
    fclose(write_ptr);
    free(asset_data);
    __android_log_print(ANDROID_LOG_DEBUG, "jniWrapperV2", "wrote %s to local storage", file_name);
    return true;

}

// Read contents of files
char *
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

#undef LOGTAG

#ifdef __cplusplus
}
#endif
