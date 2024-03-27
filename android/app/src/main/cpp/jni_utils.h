//
// Created by rabijl on 27.3.2024.
//

#ifndef POCL_AISA_DEMO_JNI_UTILS_H
#define POCL_AISA_DEMO_JNI_UTILS_H


#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#ifdef __cplusplus
extern "C" {
#endif

bool
put_asset_in_local_storage(JNIEnv *env, jobject jAssetManager, const char *file_name);

// Read contents of files
char *
read_file(JNIEnv *env, jobject jAssetManager, const char *filename, size_t *bytes_read);

#ifdef __cplusplus
}
#endif


#endif //POCL_AISA_DEMO_JNI_UTILS_H