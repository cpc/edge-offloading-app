//
// Created by rabijl on 31.3.2023.
//

#ifndef POCL_AISA_DEMO_POCLREMOTEEXAMPLE_H
#define POCL_AISA_DEMO_POCLREMOTEEXAMPLE_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_poclRemoteVectorAdd(JNIEnv *env, jobject thiz, jint n,
                                                                  jfloatArray a, jfloatArray b,
                                                                  jfloatArray c);

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_destroyPoCL(JNIEnv *env, jobject thiz);

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_initPoCL(JNIEnv *env, jobject thiz);

JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_MainActivity_setPoCLEnv(JNIEnv *env, jobject thiz, jstring key,
                                                         jstring value);


#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLREMOTEEXAMPLE_H



