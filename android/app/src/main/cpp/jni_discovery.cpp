#include <jni.h>
#include "CL/cl.h"
#include <string>
#include <android/log.h>
#include <pthread.h>

//
// Created by rabijl on 17.4.2024.
//

#define str(x) #x

#define CHECK(x)                                                    \
    do{                                                             \
        cl_int err = (x);                                           \
        if(err != CL_SUCCESS)                                       \
        {                                                           \
            __android_log_print(ANDROID_LOG_INFO, "DISC",           \
            "CL error %d in " __FILE__ ":%d: %s\n",err,             \
            __LINE__, str(x));                                      \
            return err;                                             \
        }                                                           \
    }while(0)


cl_int (*func)(char * const, cl_uint, void *);


cl_int add_device(char *parameter, cl_uint mode){

    // mode => 0:add - 1:reconnect
    cl_platform_id *platforms = NULL;
    cl_uint numPlatforms = 0;
    char funcName[] = "clAddReconnectDevicePOCL";
    char device_driver_name[] = "remote";

    CHECK(clGetPlatformIDs(0, NULL, &numPlatforms));
    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    CHECK(clGetPlatformIDs(numPlatforms, platforms, NULL));

    func = (cl_int (*)(char * const, cl_uint, void *)) clGetExtensionFunctionAddressForPlatform(platforms[0], funcName);

    CHECK(func(parameter, mode, device_driver_name));

    free(platforms);
    return CL_SUCCESS;
}

extern "C"
JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_Discovery_addDevice(JNIEnv *env, jclass clazz, jstring key,
                                                     jint mode) {
    char *parameter = (char *) env->GetStringUTFChars(key, 0);

//    add_device(parameter, (cl_uint)mode);
    __android_log_print(ANDROID_LOG_INFO, "DISC", "from discovery service: %s \n", parameter);
    env->ReleaseStringUTFChars(key, parameter);
}