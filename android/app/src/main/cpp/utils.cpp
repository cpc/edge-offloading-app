//
// Created by rabijl on 3.4.2023.
//

#include <jni.h>
#include <android/log.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

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

    char * c_key = (char*) env->GetStringUTFChars(key, 0);
    char * c_value = (char*) env->GetStringUTFChars(value, 0);
    __android_log_print(ANDROID_LOG_INFO, "Native utils", "setting env variable: %s : %s", c_key,
                        c_value);

    setenv(c_key, c_value, 1);

}

#ifdef __cplusplus
}
#endif