//
// Created by rabijl on 29.3.2023.
//



#ifndef POCL_AISA_DEMO_VECTORADDEXAMPLE_H
#define POCL_AISA_DEMO_VECTORADDEXAMPLE_H

#include <jni.h>

#define CL_TARGET_OPENCL_VERSION 200

#ifdef __cplusplus
extern "C" {
#endif


JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_initCL(JNIEnv *env, jobject thiz);

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_vectorAddCL(JNIEnv *env, jobject thiz, jint n,
                                                          jfloatArray a, jfloatArray b,
                                                          jfloatArray c);

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_destroyCL(JNIEnv *env, jobject thiz);

JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_MainActivity_setenv(JNIEnv *env, jobject thiz, jstring key,
                                                     jstring value);

#ifdef __cplusplus
}
#endif

#endif //
